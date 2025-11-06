from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

"""
ReaderAgent: 读取 Markdown，并可选调用图片描述增强。

为兼容不同项目结构，避免在模块导入期就强依赖 picture 模块，
在需要增强时再进行相对导入：`from .picture import describe_images_in_markdown`。
"""


logger = logging.getLogger(__name__)


@dataclass
class ReaderConfig:
    """Configuration parameters for the ReaderAgent."""

    md_path: Path
    annotate_images: bool = True
    annotated_output: Optional[Path] = None
    encoding: str = "utf-8"


class ReaderAgent:
    """
    Load Markdown content (optionally enriching image descriptions via picture.py)
    and emit the raw text into the shared pipeline state.
    """

    def __init__(self, config: ReaderConfig):
        self.config = config
        self.config.md_path = self.config.md_path.resolve()
        if self.config.annotated_output:
            self.config.annotated_output = self.config.annotated_output.resolve()

    def run(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = state or {}

        if not self.config.md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {self.config.md_path}")

        active_md_path = self._prepare_markdown()
        logger.info("ReaderAgent ingesting markdown from %s", active_md_path)

        text = active_md_path.read_text(encoding=self.config.encoding)
        state["text"] = text
        state.setdefault("artifacts", {})
        state["artifacts"]["markdown_path"] = str(active_md_path)

        return state

    def _prepare_markdown(self) -> Path:
        """
        Optionally annotate images via picture.py; otherwise return original path.
        """
        if not self.config.annotate_images:
            return self.config.md_path

        annotated_path = self.config.annotated_output or Path("output/annotated.md").resolve()
        annotated_path.parent.mkdir(parents=True, exist_ok=True)

        # 延迟导入，且使用相对导入以适配 agents/picture.py 的位置
        try:
            from .picture import describe_images_in_markdown  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "未找到图片增强模块（agents/picture.py）。如果不需要图片增强，请使用 --skip-annotation；"
                "如需增强，请将 picture.py 放置到 agents/ 目录或修正导入路径。"
            ) from exc

        describe_images_in_markdown(
            input_md_path=str(self.config.md_path),
            output_md_path=str(annotated_path),
        )

        return annotated_path
