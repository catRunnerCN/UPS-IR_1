from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


logger = logging.getLogger(__name__)


@dataclass
class SynthesizerConfig:
    """Configuration for SynthesizerAgent."""

    output_dir: Path = Path("UPS-IR_Output")
    pretty: bool = True


class SynthesizerAgent:
    """
    Persist UPS-IR sections into dedicated JSON files for downstream use.
    """

    def __init__(self, config: SynthesizerConfig | None = None):
        self.config = config or SynthesizerConfig()
        self.config.output_dir = self.config.output_dir.resolve()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if "ups_ir" not in state:
            raise ValueError("SynthesizerAgent requires 'ups_ir' in the incoming state.")

        ups_ir = state["ups_ir"]
        indent = 2 if self.config.pretty else None

        for key, value in ups_ir.items():
            output_path = self.config.output_dir / f"{key}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=indent)
            logger.info("SynthesizerAgent wrote %s", output_path)

        state.setdefault("artifacts", {})
        state["artifacts"]["synthesized_dir"] = str(self.config.output_dir)

        return state
