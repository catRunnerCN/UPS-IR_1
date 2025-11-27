from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class FaithfulnessConfig:
    """Configuration for FaithfulnessVerifierAgent."""

    output_path: Path = Path("output/extracted_info.json")
    report_path: Path = Path("output/faithfulness_report.json")
    strict: bool = False

#q s
@dataclass
class RulePack:
    """Simple rule definition for domain-sensitive checks."""

    name: str
    keywords: List[str]
    validator: Callable[[Dict[str, Any]], bool]
    message: str


class FaithfulnessVerifierAgent:
    """
    Validate and enrich UPS-IR JSON emitted by the extractor to ensure cross-field consistency.
    """

    def __init__(self, config: Optional[FaithfulnessConfig] = None):
        self.config = config or FaithfulnessConfig()
        self.config.output_path = self.config.output_path.resolve()
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_path = self.config.report_path.resolve()
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)

        self._rule_packs: List[RulePack] = [
            RulePack(
                name="diffusion",
                keywords=["diffusion"],
                validator=lambda info: any(
                    "score" in (formula.get("latex", "").lower())
                    or "noise" in (formula.get("latex", "").lower())
                    for formula in info.get("model", {}).get("formulas", [])
                ),
                message="Diffusion papers should include a noise-schedule or score-matching style equation.",
            ),
            RulePack(
                name="reinforcement_learning",
                keywords=["reinforcement", "policy", "rl"],
                validator=lambda info: any(
                    "reward" in str(metric.get("name", "")).lower()
                    or "return" in str(metric.get("name", "")).lower()
                    for exp in info.get("experiments", [])
                    for metric in exp.get("results", []) or []
                    if isinstance(metric, dict)
                ),
                message="Reinforcement learning experiments must report at least one reward/return metric.",
            ),
            RulePack(
                name="gnn",
                keywords=["graph", "gnn"],
                validator=lambda info: any(
                    "graph" in (component.get("name", "").lower())
                    for component in info.get("model", {}).get("components", [])
                ),
                message="GNN models must include at least one graph-structured component (e.g., message passing, graph encoder).",
            ),
        ]

    def run(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if state is None or "info" not in state:
            raise ValueError("FaithfulnessVerifierAgent requires 'info' in the incoming state.")

        info = self._ensure_schema(state["info"])
        issues: List[Dict[str, str]] = []

        self._normalize_metadata(info)
        dataset_names = self._normalize_datasets(info, issues)
        component_names = self._normalize_model(info, issues)
        self._normalize_problem(info)
        self._normalize_experiments(info, dataset_names, component_names, issues)
        self._apply_rule_packs(info, issues)

        self._emit_outputs(info, issues)

        state["info"] = info
        state.setdefault("artifacts", {})
        state["artifacts"]["extracted_json"] = str(self.config.output_path)
        state["artifacts"]["faithfulness_report"] = str(self.config.report_path)

        if self.config.strict and any(issue["severity"] == "error" for issue in issues):
            raise ValueError("Faithfulness verification failed. See report for details.")

        logger.info(
            "FaithfulnessVerifierAgent completed with %s issues.",
            len(issues),
        )
        return state

    def _ensure_schema(self, raw_info: Any) -> Dict[str, Any]:
        info = raw_info if isinstance(raw_info, dict) else {}
        info.setdefault("metadata", {})
        info.setdefault("problem", {})
        info.setdefault("model", {})
        info.setdefault("dataset", [])
        info.setdefault("experiments", [])

        if not isinstance(info["metadata"], dict):
            info["metadata"] = {}
        if not isinstance(info["problem"], dict):
            info["problem"] = {}
        if not isinstance(info["model"], dict):
            info["model"] = {}

        dataset_value = info["dataset"]
        if isinstance(dataset_value, dict):
            info["dataset"] = [dataset_value]
        elif not isinstance(dataset_value, list):
            info["dataset"] = []

        experiments_value = info["experiments"]
        if isinstance(experiments_value, dict):
            info["experiments"] = [experiments_value]
        elif not isinstance(experiments_value, list):
            info["experiments"] = []

        return info

    def _normalize_metadata(self, info: Dict[str, Any]) -> None:
        metadata = info["metadata"]
        authors = metadata.get("authors")
        if isinstance(authors, str):
            authors_list = [a.strip() for a in re.split(r",|;| and ", authors) if a.strip()]
        elif isinstance(authors, list):
            authors_list = [str(a).strip() for a in authors if str(a).strip()]
        else:
            authors_list = []
        metadata["authors"] = authors_list

        metadata.setdefault("title", None)
        metadata.setdefault("year", None)
        metadata.setdefault("venue", None)
        metadata.setdefault("keywords", [])
        metadata.setdefault("doi", None)
        metadata.setdefault("arxiv_id", None)

    def _normalize_problem(self, info: Dict[str, Any]) -> None:
        problem = info["problem"]
        problem.setdefault("objective", None)
        problem.setdefault("task", None)
        problem.setdefault("motivation", None)

    def _normalize_datasets(self, info: Dict[str, Any], issues: List[Dict[str, str]]) -> set[str]:
        normalized: List[Dict[str, Any]] = []
        seen_names: set[str] = set()

        for idx, raw in enumerate(info["dataset"]):
            if isinstance(raw, dict):
                dataset = dict(raw)
            else:
                dataset = {"name": str(raw)}

            name = dataset.get("name")
            if not name:
                name = f"dataset_{idx + 1}"
                dataset["name"] = name
                self._add_issue(
                    issues,
                    "warning",
                    "dataset",
                    f"Dataset #{idx + 1} is missing a name; generated {name} automatically.",
                )
            if name in seen_names:
                suffix = len(seen_names) + 1
                new_name = f"{name}_{suffix}"
                dataset["name"] = new_name
                name = new_name
                self._add_issue(
                    issues,
                    "warning",
                    "dataset",
                    f"Duplicate dataset name detected; renamed to {new_name}.",
                )
            seen_names.add(name)

            dataset.setdefault("num_samples", None) # type: ignore
            dataset.setdefault("num_classes", None)
            dataset.setdefault("features", None)
            dataset.setdefault("input_dimensions", None)
            dataset.setdefault("preprocessing", None)
            dataset.setdefault("split", {})

            normalized.append(dataset)

        info["dataset"] = normalized
        return seen_names

    def _normalize_model(self, info: Dict[str, Any], issues: List[Dict[str, str]]) -> set[str]:
        model = info["model"]
        architecture = model.get("architecture")
        if isinstance(architecture, list):
            architecture = ", ".join(str(x) for x in architecture)
        model["architecture"] = architecture or "unspecified"

        raw_components = model.get("components") or model.get("submodules") or []
        components: List[Dict[str, Any]] = []
        slug_counts: Dict[str, int] = {}
        component_names: set[str] = set()

        if not isinstance(raw_components, list):
            raw_components = [raw_components]

        for idx, entry in enumerate(raw_components):
            component = self._normalize_component(entry, idx, slug_counts)
            components.append(component)
            component_names.add(component["name"])

        if not components and architecture:
            component = self._normalize_component(architecture, 0, slug_counts)
            component["description"] = component.get("description") or "Auto-derived from architecture."
            components.append(component)
            component_names.add(component["name"])
            self._add_issue(
                issues,
                "warning",
                "model",
                "Component list missing; created one entry based on the architecture field.",
            )

        model["components"] = components

        flow = model.get("flow") or model.get("execution_flow") or []
        if isinstance(flow, str):
            flow = [step.strip() for step in re.split(r"->|,|>", flow) if step.strip()]
        elif not isinstance(flow, list):
            flow = []
        model["flow"] = flow

        dag = self._build_dag(flow, components)
        model["dag"] = dag

        formulas = model.get("formulas") or []
        if not isinstance(formulas, list):
            formulas = [formulas]
        normalized_formulas: List[Dict[str, Any]] = []
        for idx, formula in enumerate(formulas):
            normalized_formulas.append(self._normalize_formula_entry(formula, idx, issues))
        model["formulas"] = normalized_formulas

        component_names.update(flow)
        return component_names

    def _normalize_component(
        self,
        component: Any,
        index: int,
        slug_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        if isinstance(component, dict):
            name = component.get("name") or component.get("id") or f"component_{index + 1}"
            description = component.get("description")
        else:
            name = str(component) if component is not None else f"component_{index + 1}"
            description = None

        comp_id = self._slugify(name, slug_counts)
        return {"id": comp_id, "name": name, "description": description}

    def _normalize_formula_entry(
        self,
        entry: Any,
        index: int,
        issues: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if isinstance(entry, dict):
            formula = dict(entry)
        else:
            formula = {"latex": str(entry)}

        formula.setdefault("id", f"equation_{index + 1}")
        latex = formula.get("latex", "")
        formula["latex"] = latex
        formula.setdefault("variables", {})

        ast_tree = self._build_formula_ast(latex)
        formula["ast"] = ast_tree

        consistency = self._check_unit_balance(latex, formula["variables"])
        if consistency is not None:
            formula["unit_consistent"] = consistency
            if not consistency:
                self._add_issue(
                    issues,
                    "warning",
                    "formula",
                    f"{formula['id']} has inconsistent units; please double-check variable metadata.",
                )

        return formula
#a
    def _normalize_experiments(
        self,
        info: Dict[str, Any],
        dataset_names: set[str],
        component_names: set[str],
        issues: List[Dict[str, str]],
    ) -> None:
        problem_task = info.get("problem", {}).get("task")
        architecture = info.get("model", {}).get("architecture")

        normalized: List[Dict[str, Any]] = []
        for idx, raw in enumerate(info["experiments"]):
            entry = dict(raw) if isinstance(raw, dict) else {"description": str(raw)}
            entry.setdefault("name", entry.get("id") or f"experiment_{idx + 1}")

            datasets_ref = entry.get("dataset")
            dataset_refs = self._ensure_list(datasets_ref)

            if not dataset_refs and dataset_names:
                fallback = next(iter(dataset_names))
                entry["dataset"] = fallback
                self._add_issue(
                    issues,
                    "warning",
                    "experiments",
                    f"{entry['name']} does not specify a dataset; defaulted to {fallback}.",
                )
            else:
                for ref in dataset_refs:
                    if ref not in dataset_names:
                        dataset_names.add(ref)
                        placeholder = {
                            "name": ref,
                            "num_samples": None,
                            "num_classes": None,
                            "features": None,
                            "input_dimensions": None,
                            "preprocessing": None,
                            "split": {},
                        }
                        info["dataset"].append(placeholder)
                        self._add_issue(
                            issues,
                            "warning",
                            "experiments",
                            f"{entry['name']} referenced unknown dataset {ref}; added a placeholder entry automatically.",
                        )

            if "model" not in entry or not entry["model"]:
                entry["model"] = architecture
            if "task" not in entry or not entry["task"]:
                entry["task"] = problem_task

            comp_refs = entry.get("components")
            comp_list = self._ensure_list(comp_refs)
            if comp_list:
                filtered = [ref for ref in comp_list if ref in component_names]
                if len(filtered) != len(comp_list):
                    if not filtered and component_names:
                        filtered = [next(iter(component_names))]
                    entry["components"] = filtered
                    self._add_issue(
                        issues,
                        "warning",
                        "experiments",
                        f"{entry['name']} referenced unknown components; removed invalid entries.",
                    )
            elif component_names:
                primary = next(iter(component_names))
                entry["components"] = [primary]
                self._add_issue(
                    issues,
                    "info",
                    "experiments",
                    f"{entry['name']} did not specify components; linked to {primary}.",
                )

            if dataset_refs:
                cleaned = [ref for ref in dataset_refs if ref in dataset_names]
                if len(cleaned) == 1:
                    entry["dataset"] = cleaned[0]
                elif len(cleaned) > 1:
                    entry["dataset"] = cleaned

            entry.setdefault("training_strategy", None)
            entry.setdefault("hyperparameters", {})
            entry.setdefault("evaluation_protocol", None)
            entry.setdefault("results", [])
            entry.setdefault("hardware", None)
            entry.setdefault("reproducibility", None)
            entry.setdefault("procedure", None)

            metrics = entry.get("results")
            if isinstance(metrics, dict):
                entry["results"] = [
                    {"name": k, "value": metrics[k]} for k in metrics
                ]
                self._add_issue(
                    issues,
                    "info",
                    "experiments",
                    f"Converted {entry['name']} results from a dict to a list representation.",
                )
            elif not isinstance(metrics, list):
                entry["results"] = [{"name": "metric", "value": metrics}]

            normalized.append(entry)

        info["experiments"] = normalized

    def _apply_rule_packs(self, info: Dict[str, Any], issues: List[Dict[str, str]]) -> None:
        searchable_text = " ".join(
            filter(
                None,
                [
                    str(info.get("problem", {}).get("objective") or ""),
                    str(info.get("problem", {}).get("task") or ""),
                    str(info.get("model", {}).get("architecture") or ""),
                ],
            )
        ).lower()

        for pack in self._rule_packs:
            if any(keyword in searchable_text for keyword in pack.keywords):
                try:
                    passed = pack.validator(info)
                except Exception as exc:  # pragma: no cover - defensive
                    passed = False
                    logger.warning("Rule pack %s raised %s", pack.name, exc)
                if not passed:
                    self._add_issue(
                        issues,
                        "warning",
                        "rule_pack",
                        f"{pack.name} validation failed: {pack.message}",
                    )

    def _emit_outputs(self, info: Dict[str, Any], issues: List[Dict[str, str]]) -> None:
        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        report = {
            "status": "passed" if not issues else "completed_with_warnings",
            "issue_count": len(issues),
            "issues": issues,
        }
        with open(self.config.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def _add_issue(
        self,
        issues: List[Dict[str, str]],
        severity: str,
        category: str,
        detail: str,
    ) -> None:
        issues.append({"severity": severity, "category": category, "detail": detail})

    def _ensure_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]

    def _slugify(self, value: str, counters: Dict[str, int]) -> str:
        base = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "node"
        count = counters.get(base, 0)
        counters[base] = count + 1
        return base if count == 0 else f"{base}_{count}"

    def _build_dag(self, flow: List[str], components: List[Dict[str, Any]]) -> Dict[str, Any]:
        nodes = []
        edges = []
        slug_counts: Dict[str, int] = {}
        name_to_id: Dict[str, str] = {}

        for component in components:
            comp_id = component.get("id") or self._slugify(component.get("name", "component"), slug_counts)
            component["id"] = comp_id
            nodes.append(
                {
                    "id": comp_id,
                    "name": component.get("name"),
                    "description": component.get("description"),
                }
            )
            name_to_id[component.get("name")] = comp_id

        for idx, step in enumerate(flow):
            if step not in name_to_id:
                comp_id = self._slugify(step, slug_counts)
                name_to_id[step] = comp_id
                nodes.append({"id": comp_id, "name": step, "description": None})
            if idx > 0:
                prev_step = flow[idx - 1]
                edges.append({"from": name_to_id[prev_step], "to": name_to_id[step]})

        return {"nodes": nodes, "edges": edges}

    def _build_formula_ast(self, latex: str) -> Dict[str, Any]:
        sanitized = latex.strip()
        if not sanitized:
            return {"symbol": ""}
        return self._parse_expression(sanitized)

    def _parse_expression(self, expr: str) -> Dict[str, Any]:
        operators = ["=", "+", "-", "\\times", "\\cdot", "*", "/", "^"]
        for operator in operators:
            parts = self._split_outside(expr, operator)
            if parts:
                return {
                    "op": operator.replace("\\", ""),
                    "args": [self._parse_expression(part) for part in parts],
                }
        return {"symbol": expr.strip()}

    def _split_outside(self, expr: str, delimiter: str) -> Optional[List[str]]:
        depth = 0
        start = 0
        parts: List[str] = []
        i = 0
        length = len(expr)
        delim_len = len(delimiter)

        while i < length:
            ch = expr[i]
            if ch in "{[(":
                depth += 1
            elif ch in "}])":
                depth = max(depth - 1, 0)
            if depth == 0 and expr.startswith(delimiter, i):
                parts.append(expr[start:i].strip())
                i += delim_len
                start = i
                continue
            i += 1

        if not parts:
            return None
        parts.append(expr[start:].strip())
        return [part for part in parts if part]

    def _check_unit_balance(self, latex: str, variables: Dict[str, Any]) -> Optional[bool]:
        if "=" not in latex or not variables:
            return None
        left, right = latex.split("=", 1)
        left_units = self._collect_units(left, variables)
        right_units = self._collect_units(right, variables)
        return left_units == right_units

    def _collect_units(self, expr: str, variables: Dict[str, Any]) -> set[str]:
        cleaned = re.sub(r"\\([a-zA-Z]+)", r"\1", expr)
        cleaned = cleaned.replace("{", " ").replace("}", " ")
        names = re.findall(r"[A-Za-z]+", cleaned)
        units: set[str] = set()
        for name in names:
            meta = variables.get(name)
            if isinstance(meta, dict):
                unit = meta.get("unit")
                if unit:
                    units.add(str(unit))
        return units
