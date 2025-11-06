from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Iterable


logger = logging.getLogger(__name__)


@dataclass
class StructurerConfig:
    """Configuration for StructurerAgent."""

    output_path: Path = Path("UPS-IR.json")
    version: str = "1.0"


class StructurerAgent:
    """
    Transform the extractor output into a normalized UPS-IR JSON representation.
    """

    def __init__(self, config: Optional[StructurerConfig] = None):
        self.config = config or StructurerConfig()
        self.config.output_path = self.config.output_path.resolve()
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if state is None or "info" not in state:
            raise ValueError("StructurerAgent requires 'info' in the incoming state.")

        info = state["info"]
        ups_ir = self._build_structure(info)

        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(ups_ir, f, ensure_ascii=False, indent=2)

        logger.info("StructurerAgent wrote UPS-IR JSON to %s", self.config.output_path)

        state["ups_ir"] = ups_ir
        state.setdefault("artifacts", {})
        state["artifacts"]["ups_ir_json"] = str(self.config.output_path)

        return state

    def _build_structure(self, info: Dict[str, Any]) -> Dict[str, Any]:
        meta = info.get("meta") or {}
        current_timestamp = datetime.now(timezone.utc).isoformat()

        ups_ir = {
            "version": self.config.version,
            "generated_at": current_timestamp,
            "meta": {
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "venue": meta.get("venue", ""),
                "year": meta.get("year"),
            },
            "tasks": self._normalize_entities(info.get("tasks", [])),
            "methods": self._normalize_methods(info.get("methods", [])),
            "datasets": self._normalize_entities(info.get("datasets", [])),
            "equations": self._normalize_entities(info.get("equations", [])),
            "experiments": self._normalize_experiments(info.get("experiments", [])),
            "relations": self._normalize_relations(info.get("relations", [])),
        }

        return ups_ir

    def _normalize_entities(self, items: Any) -> list[Dict[str, Any]]:
        if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
            return []

        normalized: list[Dict[str, Any]] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            normalized.append(item)
        return normalized

    def _normalize_methods(self, items: Any) -> list[Dict[str, Any]]:
        methods = self._normalize_entities(items)
        for method in methods:
            for field in ("uses_datasets", "uses_equations"):
                values = method.get(field)
                if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
                    method[field] = [str(value) for value in values]
                elif values is None:
                    method.pop(field, None)
                else:
                    method[field] = [str(values)]
        return methods

    def _normalize_experiments(self, items: Any) -> list[Dict[str, Any]]:
        if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
            return []

        normalized: list[Dict[str, Any]] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            if "method" in item and item["method"] is not None:
                item["method"] = str(item["method"])
            if "dataset" in item and item["dataset"] is not None:
                item["dataset"] = str(item["dataset"])

            metrics = item.get("metrics")
            if isinstance(metrics, dict):
                item["metrics"] = [{"name": str(k), "value": metrics[k]} for k in metrics]
            elif isinstance(metrics, Iterable) and not isinstance(metrics, (str, bytes)):
                normalized_metrics: list[Dict[str, Any]] = []
                for metric in metrics:
                    if isinstance(metric, dict):
                        normalized_metrics.append(metric)
                item["metrics"] = normalized_metrics
            elif metrics is None:
                item.pop("metrics", None)
            else:
                item["metrics"] = [{"value": metrics}]

            normalized.append(item)
        return normalized

    def _normalize_relations(self, items: Any) -> list[Dict[str, Any]]:
        if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
            return []

        normalized: list[Dict[str, Any]] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            for field in ("from", "to"):
                if field in item and item[field] is not None:
                    item[field] = str(item[field])
            normalized.append(item)
        return normalized
