from __future__ import annotations

from typing import Any, Dict, TypedDict


class PipelineState(TypedDict, total=False):
    text: str
    info: Dict[str, Any]
    ups_ir: Dict[str, Any]
    artifacts: Dict[str, str]


def initial_state() -> PipelineState:
    return PipelineState()
