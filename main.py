from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from state import PipelineState, initial_state
import os


def set_api_keys_inline() -> None:
    """Inline set API credentials for local testing only.

    WARNING: Do not commit real keys in shared repos. This is per-user testing.
    """
    # User-provided testing endpoint and key
    OPENAI_BASE_URL = "https://api.openai.com/v1"
    OPENAI_API_KEY = "your OPENAI API KEY here"

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    # Prefer newer var; keep legacy for compatibility across libs
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
    os.environ.setdefault("OPENAI_API_BASE", OPENAI_BASE_URL)

    def _mask(v: str | None) -> str:
        return ("****" + v[-4:]) if v and len(v) >= 4 else "(unset)"

    print("OPENAI_BASE_URL:", os.environ.get("OPENAI_BASE_URL", "(unset)"))
    print("OPENAI_API_KEY:", _mask(os.environ.get("OPENAI_API_KEY")))


def configure_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "agent_pipeline.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UPS-IR LangChain/LangGraph pipeline runner")
    parser.add_argument("md_path", type=Path, help="Path to the source Markdown file")
    parser.add_argument(
        "--agents",
        type=str,
        default="reader,extractor,faithfulness",
        help="Comma-separated list indicating which agents to run in order.",
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip the picture description augmentation step in ReaderAgent.",
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Execute the pipeline via LangGraph's StateGraph instead of sequential calls.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name for ExtractorAgent (LangChain ChatOpenAI).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for ExtractorAgent LLM.",
    )

    return parser.parse_args(list(argv))


def create_agents(agent_names: list[str], args: argparse.Namespace) -> Dict[str, object]:
    agents: Dict[str, object] = {}

    if "reader" in agent_names:
        from agents.reader import ReaderAgent, ReaderConfig  # lazy import

        agents["reader"] = ReaderAgent(
            ReaderConfig(
                md_path=args.md_path,
                annotate_images=not args.skip_annotation,
                annotated_output=Path("output") / "paper_with_images.md",
            )
        )

    if "extractor" in agent_names:
        from agents.extractor import ExtractorAgent, ExtractorConfig  # lazy import

        agents["extractor"] = ExtractorAgent(
            ExtractorConfig(
                output_path=Path("output") / "extracted_info.json",
                model_name=args.model,
                temperature=args.temperature,
            )
        )

    if "structurer" in agent_names:
        from agents.structurer import StructurerAgent, StructurerConfig  # lazy import

        agents["structurer"] = StructurerAgent(
            StructurerConfig(output_path=Path("UPS-IR.json"))
        )

    if "verifier" in agent_names:
        from agents.verifier import VerifierAgent  # lazy import

        agents["verifier"] = VerifierAgent()

    if "faithfulness" in agent_names:
        from agents.faithfulness import FaithfulnessVerifierAgent, FaithfulnessConfig  # lazy import

        agents["faithfulness"] = FaithfulnessVerifierAgent(
            FaithfulnessConfig(
                output_path=Path("output") / "extracted_info.json",
                report_path=Path("output") / "faithfulness_report.json",
            )
        )

    if "synthesizer" in agent_names:
        from agents.synthesizer import SynthesizerAgent, SynthesizerConfig  # lazy import

        agents["synthesizer"] = SynthesizerAgent(
            SynthesizerConfig(output_dir=Path("UPS-IR_Output"))
        )

    return agents


def parse_agent_sequence(agent_string: str) -> List[str]:
    sequence = [item.strip().lower() for item in agent_string.split(",") if item.strip()]
    valid_order = ["reader", "extractor", "faithfulness", "structurer", "verifier", "synthesizer"]

    for agent_name in sequence:
        if agent_name not in valid_order:
            raise ValueError(f"Unsupported agent '{agent_name}'. Valid options: {', '.join(valid_order)}")

    # Preserve global order but allow omissions or reordering within the valid list.
    return sequence


def run_sequential(agent_names: List[str], agents: Dict[str, object]) -> PipelineState:
    state: PipelineState = initial_state()

    for name in agent_names:
        agent = agents[name]
        if name == "verifier":
            verified = agent.run(state)  # type: ignore[attr-defined]
            state.setdefault("artifacts", {})
            state["artifacts"]["verified"] = str(bool(verified))
            if not verified:
                logging.error("Verification failed. Halting pipeline before synthesis.")
                break
        else:
            state = agent.run(state)  # type: ignore[attr-defined]

    if "structurer" in agent_names and "ups_ir" not in state:
        logging.warning("UPS-IR structure not present in state after sequential run.")

    return state


def run_via_graph(agent_names: List[str], agents: Dict[str, object]) -> PipelineState:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is not installed. Install it or run without --use-graph.") from exc

    if not agent_names:
        raise ValueError("At least one agent must be specified when using LangGraph.")

    builder: StateGraph = StateGraph(PipelineState)

    for name in agent_names:
        if name not in agents:
            raise ValueError(f"Agent '{name}' was not created.")

        agent = agents[name]

        if name == "verifier":
            def _make_verifier_node(verifier_agent):
                def _node(state: PipelineState) -> PipelineState:
                    result = verifier_agent.run(state)  # type: ignore[attr-defined]
                    if not result:
                        raise RuntimeError("VerifierAgent rejected the UPS-IR structure.")
                    state.setdefault("artifacts", {})
                    state["artifacts"]["verified"] = "True"
                    return state

                return _node

            builder.add_node(name, _make_verifier_node(agent))
        else:
            builder.add_node(name, lambda s, agent=agent: agent.run(s))  # type: ignore[attr-defined]

    builder.set_entry_point(agent_names[0])
    for src, dst in zip(agent_names, agent_names[1:]):
        builder.add_edge(src, dst)
    builder.add_edge(agent_names[-1], END)

    graph = builder.compile()
    return graph.invoke(initial_state())


def main(argv: Iterable[str] | None = None) -> PipelineState:
    # Inline keys for local testing (user requested). Remove for production use.
    set_api_keys_inline()
    configure_logging()
    set_llm_cache(InMemoryCache())

    args = parse_args(argv or sys.argv[1:])

    if not args.md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {args.md_path}")

    # Decide which agents to import/build before importing heavy deps
    agent_sequence = parse_agent_sequence(args.agents)
    agents = create_agents(agent_sequence, args)

    if args.use_graph:
        logging.info("Running pipeline via LangGraph.")
        state = run_via_graph(agent_sequence, agents)
    else:
        logging.info("Running pipeline sequentially with agents: %s", ", ".join(agent_sequence))
        state = run_sequential(agent_sequence, agents)

    logging.info("Pipeline execution completed.")
    return state


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI convenience
        logging.exception("Pipeline execution failed: %s", exc)
        sys.exit(1)
