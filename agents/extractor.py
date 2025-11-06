from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are an expert scientific information extraction assistant. Your task is to extract structured UPS-IR information "
    "from the following academic paper text. Output MUST be a valid JSON object strictly following the schema below.\n\n"
    "Additional requirements for referenced fields:"
    "1. methods[*].uses_datasets must specify an ID from the datasets list (e.g., d1)."
    "2. methods[*].uses_equations must specify an ID from the equations list (e.g., eq1)."
    "3. experiments[*].method and experiments[*].dataset must also reference corresponding list IDs.\n"
    "4. relations[*].from / relations[*].to may only use IDs from the tasks, methods, datasets, or equations lists.\n\n"
    
    "Use the following format:\n"
    "{\n"
    '  "meta": {\n'
    '    "title": str,\n'
    '    "authors": [str],\n'
    '    "venue": str,\n'
    '    "year": int\n'
    '  },\n'
    '  "tasks": [\n'
    '    {"id": "t1", "name": str}, ...\n'
    '  ],\n'
    '  "methods": [\n'
    '    {"id": "m1", "name": str, "uses_datasets": [str], "uses_equations": [str]}, ...\n'
    '  ],\n'
    '  "datasets": [\n'
    '    {"id": "d1", "name": str, "split": {"train": int}}, ...\n'
    '  ],\n'
    '  "equations": [\n'
    '    {"id": "e1", "latex": str, "units": str}, ...\n'
    '  ],\n'
    '  "experiments": [\n'
    '    {"method": "m1", "dataset": "d1", "metrics": [{"name": str, "value": float}]}, ...\n'
    '  ],\n'
    '  "relations": [\n'
    '    {"from": str, "to": str, "type": str}, ...\n'
    '  ]\n'
    '}\n\n'

    "Rules:\n"
    "- All IDs must follow the format 't1', 'm1', 'd1', 'e1' etc.\n"
    "- Do not invent content not present in the text.\n"
    "- Avoid paraphrasing; prefer exact phrases from the text.\n"
    "- Only output the final JSON. No explanation or commentary.\n\n"

    "Thesis content:\n\"\"\"\n{text}\n\"\"\""
)



@dataclass
class ExtractorConfig:
    """Configuration for the ExtractorAgent."""

    output_path: Path = Path("output/extracted_info.json")
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0


class ExtractorAgent:
    """
    Call an LLM via LangChain to convert raw text into structured UPS-IR fields.
    """

    def __init__(self, config: Optional[ExtractorConfig] = None, llm: Optional[ChatOpenAI] = None):
        self.config = config or ExtractorConfig()
        self.config.output_path = self.config.output_path.resolve()
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.llm = llm or ChatOpenAI(model=self.config.model_name, temperature=self.config.temperature)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self._chain = self.prompt | self.llm | StrOutputParser()

    def run(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if state is None or "text" not in state:
            raise ValueError("ExtractorAgent requires 'text' in the incoming state.")

        raw_response = self._chain.invoke({"text": state["text"]})
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
