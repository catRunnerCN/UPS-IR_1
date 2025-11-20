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
    "You are a scientific information extraction assistant.\n"
    "Your task is to extract structured UPS-IR data from the paper text below and output a valid JSON.\n"
    "Strictly follow the schema below. Do not invent or omit fields. Only extract if clearly supported by the text.\n\n"

    "Cross-reference constraints:\n"
    "- All IDs must be explicit: t1, m1, d1, e1, etc.\n"
    "- All references between methods, datasets, equations, and experiments must use IDs from their respective sections.\n"
    "- If a dataset or equation is used in an experiment or method, it must also appear in methods[*].uses_* and relations[*].\n\n"

    "Schema:\n"
    "{{\n"
    "  \"meta\": {{\"title\": str, \"authors\": [str], \"venue\": str, \"year\": int}},\n\n"

    "  \"tasks\": [\n"
    "    {{\"id\": \"t1\", \"name\": str}}, ...\n"
    "  ],\n\n"

    "  \"methods\": [\n"
    "    {{\n"
    "      \"id\": \"m1\", \"name\": str,\n"
    "      \"uses_datasets\": [\"d1\"],\n"
    "      \"uses_equations\": [\"e1\"],\n"
    "      \"components\": [str],            // e.g. [\"encoder\", \"attention\"]\n"
    "      \"inherits_from\": [str],         // e.g. [\"Transformer\"]\n"
    "      \"framework\": str                // e.g. \"PyTorch\"\n"
    "    }}, ...\n"
    "  ],\n\n"

    "  \"datasets\": [\n"
    "    {{\n"
    "      \"id\": \"d1\", \"name\": str,\n"
    "      \"description\": str,             // e.g. \"parallel corpus for translation\"\n"
    "      \"modality\": str,                // one of: image, text, audio, graph\n"
    "      \"split\": {{\"train\": int, \"test\": int, \"val\": int}}\n"
    "    }}, ...\n"
    "  ],\n\n"

    "  \"equations\": [\n"
    "    {{\n"
    "      \"id\": \"e1\", \"latex\": str, \"units\": str,\n"
    "      \"defines\": str,                // e.g. \"loss\", \"attention\"\n"
    "      \"type\": str,                   // e.g. \"embedding\", \"loss\", \"transition\"\n"
    "      \"variables\": [str]             // e.g. [\"Q\", \"K\", \"V\", \"d_k\"]\n"
    "    }}, ...\n"
    "  ],\n\n"

    "  \"experiments\": [\n"
    "    {{\n"
    "      \"method\": \"m1\", \"dataset\": \"d1\",\n"
    "      \"hyperparameters\": {{\"lr\": float, \"batch_size\": int}},\n"
    "      \"setup\": str,                  // e.g. \"8×A100, 80GB\"\n"
    "      \"metrics\": [{{\"name\": str, \"value\": float}}],\n"
    "      \"repeat\": int,                 // number of times averaged\n"
    "      \"compare_to\": [\"m2\"]         // baseline methods by ID\n"
    "    }}, ...\n"
    "  ],\n\n"

    "  \"relations\": [\n"
    "    {{\"from\": str, \"to\": str, \"type\": str}}, ...\n"
    "  ],\n\n"

    "  \"references\": [\n"
    "    {{\"target\": str, \"type\": str}}, ...\n"
    "  ],\n\n"

    "  \"schema_version\": \"1.1\",\n"
    "  \"history\": [\n"
    "    {{\"by\": str, \"date\": str, \"changes\": str}}\n"
    "  ]\n"
    "}}\n\n"

    "Instructions:\n"
    "- Output must be valid JSON only\n"
    "- Do not paraphrase; use phrases from the text verbatim when possible\n"
    "- Do not invent content not present in the text\n"
    "- Leave values empty only if the paper provides no clue\n"
    "- If the user supplies clarifications, integrate them exactly as stated\n\n"

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


class ExtractorAgent:
    """
    Call an LLM via LangChain to convert raw text into structured UPS-IR fields.
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
