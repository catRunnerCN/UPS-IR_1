from __future__ import annotations

import json
import logging
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "请从以下论文文本中抽取关键信息并以 JSON 格式输出：\n"
    "- meta: {{title, authors, venue, year}}\n"
    "- tasks: [id, name]\n"
    "- methods: [id, name, uses_datasets, uses_equations]\n"
    "- datasets: [id, name, split]\n"
    "- equations: [id, latex, units]\n"
    "- experiments: [method, dataset, metrics]\n"
    "- relations: [{{from, to, type}}]\n\n"
    "对引用字段的额外要求：\n"
    "1. methods[*].uses_datasets 必须填写 datasets 列表中的 id（例如 d1）。\n"
    "2. methods[*].uses_equations 必须填写 equations 列表中的 id（例如 eq1）。\n"
    "3. experiments[*].method 和 experiments[*].dataset 也必须引用对应列表的 id。\n"
    "4. relations[*].from / relations[*].to 只能使用 tasks、methods、datasets、equations 列表中的 id。\n\n"
    "论文内容：\"\"\"{text}\"\"\""
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

        if llm is not None:
            self.llm = llm
        else:
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
            if not base_url.rstrip("/").endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            api_key = os.getenv("OPENAI_API_KEY")
            self.llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                base_url=base_url,
                api_key=api_key,
                timeout=60,
                max_retries=2,
            )
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

        def _parse(payload: str) -> Dict[str, Any]:
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object at the top level.")
            return parsed

        try:
            return _parse(cleaned)
        except json.JSONDecodeError as exc:
            if "Invalid \\escape" in str(exc):
                repaired = ExtractorAgent._escape_invalid_backslashes(cleaned)
                if repaired != cleaned:
                    try:
                        return _parse(repaired)
                    except json.JSONDecodeError:
                        pass
            raise ValueError(f"ExtractorAgent received non-JSON output: {exc}\nContent:\n{cleaned}") from exc

    @staticmethod
    def _strip_fences(block: str) -> str:
        lines = block.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _escape_invalid_backslashes(payload: str) -> str:
        """
        Double-escape lone backslashes that would break JSON decoding.
        Keeps valid JSON escapes (e.g. \\n, \\t, \\u) unchanged.
        """

        pattern = r"(?<!\\)\\(?![\"\\/bfnrtu])"
        return re.sub(pattern, r"\\\\", payload)
