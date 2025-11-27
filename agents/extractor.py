from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are given the full text of a scientific research paper. Your task is to extract a structured UPS-IR representation of its content. Output must be a single valid JSON object with exactly the following top-level keys: metadata, problem, model, dataset, and experiments. Do not output any extra text outside the JSON. Follow these instructions carefully:\n\n"
    "metadata: Include key bibliographic and contextual fields. For example: title, authors (as a list), year, venue or conference, keywords (if given), DOI or arXiv ID (if available). Ensure consistent field naming (e.g. use snake_case for keys). Provide each field's content as a string or appropriate JSON type. If a field is not present, set it to null or an empty list.\n\n"
    "problem: Summarize the research problem and objectives. Include fields such as objective (the main goal or hypothesis), task (e.g. \"image classification\", \"regression\" etc.), and motivation or background if relevant. Mention the core problem statement or research question the paper addresses. Be concise but complete, and avoid any unrelated details.\n\n"
    "model: Describe the proposed model or method. Include:\n\n"
    "architecture: a brief description of the model architecture (e.g. \"Transformer-based encoder-decoder\", \"CNN with residual blocks\", etc.).\n\n"
    "components or submodules: list and describe major sub-modules or layers (e.g. embedding layer, attention module, decoder head, loss function).\n\n"
    "formulas: an array of mathematical equations used by the model; each should be a LaTeX string. For example, include loss functions or important equations from the paper. Optionally, you can parse each formula into variables (e.g. variables: {{\"x\": \"input image\", \"y\": \"label\", ...}}) under each formula entry.\n\n"
    "pseudocode or algorithm: if the paper provides algorithmic steps or pseudocode, include a concise version here as text.\n\n"
    "flow or execution_flow: if applicable, list the sequence of module calls or data flow between components (e.g. [\"tokenizer\", \"encoder\", \"decoder\", \"classifier\"]). This should clarify how data moves through the model.\n\n"
    "dataset: Provide details about the data. Include fields such as name (if the dataset has a specific name), num_samples (total number of data points), num_classes (for classification tasks), and features or input_dimensions if specified. Describe any data preprocessing steps (e.g. normalization, augmentation) and how the data is split (e.g. train/validation/test ratios or counts). If multiple datasets are used, list each as a separate entry in an array with the above details.\n\n"
    "experiments: Describe the experimental setup and results. Include:\n\n"
    "training_strategy: e.g. optimizer used, learning rate schedule, number of epochs, batch size.\n\n"
    "hyperparameters: any key hyperparameters with their values (learning rate, momentum, weight decay, etc.).\n\n"
    "evaluation_protocol: metrics used to evaluate (e.g. accuracy, F1 score), validation scheme (cross-validation, hold-out test set), and baseline comparisons if mentioned.\n\n"
    "results: key quantitative results (e.g. test accuracy or error rates) for the main experiments.\n\n"
    "hardware: computational resources used (e.g. GPU type, number of GPUs) if specified.\n\n"
    "reproducibility: any details that support reproducibility (e.g. random seed, code repository link).\n\n"
    "procedure: briefly outline the experiment workflow (data loading -> training -> evaluation).\n\n"
    "General requirements: Follow UPS-IR design principles: make the JSON compact and machine-parseable with clear, consistent naming. Preserve LaTeX formatting in the formulas fields (do not convert or simplify them). Make sure every field is explained and filled appropriately; if information is missing in the paper, use null or empty arrays/objects rather than omit the field. Use consistent identifiers so cross-references are clear (for example, if the experiments section mentions a dataset, it should match the name given in dataset). Avoid redundancy: do not repeat the same information in multiple sections. Return only the JSON object as the final answer.\n\n"
    "User clarifications (may be \"None\"):\n"
    "{clarifications}\n\n"
    "Paper content:\n\"\"\"\n{text}\n\"\"\""

)

CLARIFICATION_PROMPT = (
    "You are assisting an information extraction pipeline.\n"
    "Review the academic paper text below and determine if crucial UPS-IR attributes are missing or ambiguous.\n"
    "If everything is clear, respond with {{\"needs_clarification\": false}}.\n"
    "Otherwise respond with {{\"needs_clarification\": true, \"questions\": [\"question1\", ...]}} for at most three critical questions.\n"
    "Questions should focus on missing frameworks, datasets, metrics, or other essential UPS-IR details.\n\n"
    "Paper content:\n\"\"\"\n{text}\n\"\"\""
)


class _PatchedChatOpenAI(ChatOpenAI):
    """Work around providers that may return raw strings instead of OpenAI-style objects."""

    def _create_chat_result(self, response, generation_info): # type: ignore
        if isinstance(response, str):
            message = AIMessage(content=response)
            generation = ChatGeneration(message=message, text=response)
            return ChatResult(generations=[generation], llm_output=generation_info or {})
        return super()._create_chat_result(response, generation_info)



@dataclass
class ExtractorConfig:
    """Configuration for the ExtractorAgent."""

    output_path: Path = Path("output/extracted_info.json")
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    ask_user_when_unsure: bool = True

# ab
class ExtractorAgent:
    """
    Call an LLLM via LangChain to convert raw text into structured UPS-IR fields.
    """

    def __init__(self, config: Optional[ExtractorConfig] = None, llm: Optional[ChatOpenAI] = None):
        self.config = config or ExtractorConfig()
        self.config.output_path = self.config.output_path.resolve()
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.llm = llm or _PatchedChatOpenAI(model=self.config.model_name, temperature=self.config.temperature)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self._chain = self.prompt | self.llm | StrOutputParser()
        self._clarification_prompt = ChatPromptTemplate.from_template(CLARIFICATION_PROMPT)
        self._clarification_chain = self._clarification_prompt | self.llm | StrOutputParser()

    def run(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if state is None or "text" not in state:
            raise ValueError("ExtractorAgent requires 'text' in the incoming state.")

        clarifications = ""
        if self.config.ask_user_when_unsure:
            clarifications = self._gather_user_clarifications(state["text"])

        raw_response = self._chain.invoke(
            {
                "text": state["text"],
                "clarifications": clarifications or "None",
            }
        )
        logger.debug("ExtractorAgent raw response: %s", raw_response)

        info = self._try_parse_json(raw_response)
        state["info"] = info

        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        state.setdefault("artifacts", {})
        state["artifacts"]["extracted_json"] = str(self.config.output_path)

        return state

    @staticmethod
    def _try_parse_json(content: str) -> Dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = ExtractorAgent._strip_fences(cleaned)

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object at the top level.")
            return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"ExtractorAgent received non-JSON output: {exc}\nContent:\n{cleaned}") from exc

    @staticmethod
    def _strip_fences(block: str) -> str:
        lines = block.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _gather_user_clarifications(self, text: str) -> str:
        questions = self._identify_questions(text)
        if not questions:
            return ""

        responses: List[Dict[str, str]] = []
        print("\nExtractorAgent The following information may be uncertain; please supplement each item individually (press Enter to skip):")
        for idx, question in enumerate(questions, 1):
            answer = input(f"[{idx}/{len(questions)}] {question}\n> ").strip()
            if answer:
                responses.append({"question": question, "answer": answer})

        if not responses:
            return ""

        return json.dumps(responses, ensure_ascii=False)

    def _identify_questions(self, text: str) -> List[str]:
        try:
            raw = self._clarification_chain.invoke({"text": text})
        except Exception as exc:
            logger.warning("Clarification chain failed: %s", exc)
            return []

        try:
            payload = self._try_parse_json(raw)
        except ValueError:
            logger.warning("Clarification chain did not return JSON.")
            return []

        if not payload.get("needs_clarification"):
            return []

        questions_payload = payload.get("questions") or []
        questions: List[str] = []
        for item in questions_payload:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    questions.append(stripped)
        return questions
