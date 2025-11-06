from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Set

from jsonschema import Draft202012Validator, ValidationError


logger = logging.getLogger(__name__)


UPS_IR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "version",
        "generated_at",
        "meta",
        "tasks",
        "methods",
        "datasets",
        "equations",
        "experiments",
        "relations",
    ],
    "properties": {
        "version": {"type": "string"},
        "generated_at": {"type": "string"},
        "meta": {
            "type": "object",
            "required": ["title", "authors", "venue", "year"],
            "properties": {
                "title": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "venue": {"type": "string"},
                "year": {"type": ["integer", "string", "null"]},
            },
            "additionalProperties": True,
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                "additionalProperties": True,
            },
        },
        "methods": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "uses_datasets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "uses_equations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "additionalProperties": True,
            },
        },
        "datasets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "split": {"type": ["string", "null"]},
                },
                "additionalProperties": True,
            },
        },
        "equations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "latex"],
                "properties": {
                    "id": {"type": "string"},
                    "latex": {"type": "string"},
                    "units": {"type": ["string", "null"]},
                },
                "additionalProperties": True,
            },
        },
        "experiments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["method"],
                "properties": {
                    "method": {"type": "string"},
                    "dataset": {"type": ["string", "null"]},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "object"},
                        "default": [],
                    },
                },
                "additionalProperties": True,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to", "type"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "type": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": True,
}


@dataclass
class ReferenceIndex:
    methods: Set[str]
    datasets: Set[str]
    equations: Set[str]
    tasks: Set[str]

    @classmethod
    def from_ups_ir(cls, ups_ir: Dict[str, Any]) -> "ReferenceIndex":
        return cls(
            methods=cls._collect(ups_ir.get("methods", []), extra=("name",)),
            datasets=cls._collect(ups_ir.get("datasets", []), extra=("name",)),
            equations=cls._collect(ups_ir.get("equations", []), extra=("latex",)),
            tasks=cls._collect(ups_ir.get("tasks", []), extra=("name",)),
        )

    @staticmethod
    def _collect(items: Any, extra: Iterable[str]) -> Set[str]:
        values: Set[str] = set()
        if not isinstance(items, list):
            return values

        for entry in items:
            if not isinstance(entry, dict):
                continue
            for key in ("id", *tuple(extra)):
                value = entry.get(key)
                if value is None or isinstance(value, (list, dict)):
                    continue
                values.add(str(value))
        return values

    def _contains(self, collection: Set[str], value: Any) -> bool:
        if value is None or isinstance(value, (list, dict)):
            return False
        return str(value) in collection

    def is_method(self, value: Any) -> bool:
        return self._contains(self.methods, value)

    def is_dataset(self, value: Any) -> bool:
        return self._contains(self.datasets, value)

    def is_equation(self, value: Any) -> bool:
        return self._contains(self.equations, value)

    def is_task(self, value: Any) -> bool:
        return self._contains(self.tasks, value)

    def is_known(self, value: Any) -> bool:
        return any(
            (
                self.is_method(value),
                self.is_dataset(value),
                self.is_equation(value),
                self.is_task(value),
            )
        )


class VerifierAgent:
    """
    Validate UPS-IR output against a JSON schema and perform cross-reference checks.
    The cross-reference checks accept either entity IDs or their human-readable names.
    """

    def __init__(self) -> None:
        self.validator = Draft202012Validator(UPS_IR_SCHEMA)

    def run(self, state: Dict[str, Any]) -> bool:
        if "ups_ir" not in state:
            raise ValueError("VerifierAgent requires 'ups_ir' in the incoming state.")

        ups_ir = state["ups_ir"]

        try:
            self.validator.validate(ups_ir)
        except ValidationError as exc:
            raise ValueError(f"UPS-IR schema validation failed: {exc.message}") from exc

        self._check_integrity(ups_ir)
        logger.info("VerifierAgent validation successful.")
        return True

    def _check_integrity(self, ups_ir: Dict[str, Any]) -> None:
        refs = ReferenceIndex.from_ups_ir(ups_ir)

        methods = list(self._iter_dicts(ups_ir.get("methods"), "methods"))
        experiments = list(self._iter_dicts(ups_ir.get("experiments"), "experiments"))
        relations = list(self._iter_dicts(ups_ir.get("relations"), "relations"))

        self._validate_method_links(methods, refs)
        self._validate_experiments(experiments, refs)
        self._validate_relations(relations, refs)

    def _iter_dicts(self, items: Any, label: str) -> Iterator[Dict[str, Any]]:
        if items is None:
            return iter(())
        if not isinstance(items, list):
            raise ValueError(f"{label} should be a list.")
        for entry in items:
            if not isinstance(entry, dict):
                raise ValueError(f"Entries in {label} must be objects.")
            yield entry

    def _entity_handle(self, entity: Dict[str, Any]) -> str:
        return str(entity.get("id") or entity.get("name") or "<unknown>")

    def _validate_method_links(self, methods: Iterable[Dict[str, Any]], refs: ReferenceIndex) -> None:
        for method in methods:
            handle = self._entity_handle(method)
            for dataset in method.get("uses_datasets", []) or []:
                if not refs.is_dataset(dataset):
                    raise ValueError(f"Method {handle} references unknown dataset '{dataset}'.")
            for equation in method.get("uses_equations", []) or []:
                if not refs.is_equation(equation):
                    raise ValueError(f"Method {handle} references unknown equation '{equation}'.")

    def _validate_experiments(self, experiments: Iterable[Dict[str, Any]], refs: ReferenceIndex) -> None:
        for experiment in experiments:
            method = experiment.get("method")
            if not refs.is_method(method):
                raise ValueError(f"Experiment references unknown method '{method}'.")
            dataset = experiment.get("dataset")
            if dataset and not refs.is_dataset(dataset):
                raise ValueError(f"Experiment references unknown dataset '{dataset}'.")

    def _validate_relations(self, relations: Iterable[Dict[str, Any]], refs: ReferenceIndex) -> None:
        for relation in relations:
            source = relation.get("from")
            target = relation.get("to")
            relation_type = relation.get("type")

            if not refs.is_known(source):
                raise ValueError(f"Relation source '{source}' is not a known entity.")
            if not refs.is_known(target):
                raise ValueError(f"Relation target '{target}' is not a known entity.")
            if not relation_type:
                raise ValueError("Relation type is required.")
