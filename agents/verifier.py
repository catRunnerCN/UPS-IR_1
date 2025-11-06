from __future__ import annotations

import logging
from typing import Any, Dict, Set

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
            #"additionalProperties": True,
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                #"additionalProperties": True,
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
                #"additionalProperties": True,
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
                #"additionalProperties": True,
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
                #"additionalProperties": True,
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
                #"additionalProperties": True,
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
                #"additionalProperties": True,
            },
        },
    },
    #"additionalProperties": True,
}


class VerifierAgent:
    """
    Validate UPS-IR output against a JSON schema and perform cross-reference checks.
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
        method_ids = self._collect_ids(ups_ir.get("methods", []), "methods")
        dataset_ids = self._collect_ids(ups_ir.get("datasets", []), "datasets")
        equation_ids = self._collect_ids(ups_ir.get("equations", []), "equations")
        task_ids = self._collect_ids(ups_ir.get("tasks", []), "tasks")

        self._validate_method_links(ups_ir.get("methods", []), dataset_ids, equation_ids)
        self._validate_experiments(ups_ir.get("experiments", []), method_ids, dataset_ids)
        self._validate_relations(ups_ir.get("relations", []), method_ids, dataset_ids, equation_ids, task_ids)

    def _collect_ids(self, items: Any, label: str) -> Set[str]:
        ids: Set[str] = set()
        if not isinstance(items, list):
            raise ValueError(f"{label} should be a list.")
        for entry in items:
            if not isinstance(entry, dict):
                raise ValueError(f"Entries in {label} must be objects.")
            identifier = entry.get("id")
            if not identifier:
                raise ValueError(f"Missing 'id' in {label} entry: {entry}")
            if identifier in ids:
                raise ValueError(f"Duplicate id '{identifier}' detected in {label}.")
            ids.add(identifier)
        return ids

    def _validate_method_links(self, methods: Any, dataset_ids: Set[str], equation_ids: Set[str]) -> None:
        for method in methods:
            for dataset in method.get("uses_datasets", []) or []:
                if dataset not in dataset_ids:
                    raise ValueError(f"Method {method.get('id')} references unknown dataset '{dataset}'.")
            for equation in method.get("uses_equations", []) or []:
                if equation not in equation_ids:
                    raise ValueError(f"Method {method.get('id')} references unknown equation '{equation}'.")

    def _validate_experiments(self, experiments: Any, method_ids: Set[str], dataset_ids: Set[str]) -> None:
        for experiment in experiments:
            method = experiment.get("method")
            if method not in method_ids:
                raise ValueError(f"Experiment references unknown method '{method}'.")
            dataset = experiment.get("dataset")
            if dataset and dataset not in dataset_ids:
                raise ValueError(f"Experiment references unknown dataset '{dataset}'.")

    def _validate_relations(
        self,
        relations: Any,
        method_ids: Set[str],
        dataset_ids: Set[str],
        equation_ids: Set[str],
        task_ids: Set[str],
    ) -> None:
        valid_targets = method_ids | dataset_ids | equation_ids | task_ids
        for relation in relations:
            source = relation.get("from")
            target = relation.get("to")
            relation_type = relation.get("type")

            if source not in valid_targets:
                raise ValueError(f"Relation source '{source}' is not a known entity.")
            if target not in valid_targets:
                raise ValueError(f"Relation target '{target}' is not a known entity.")
            if not relation_type:
                raise ValueError("Relation type is required.")
