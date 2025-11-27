# UPS-IR Agent Pipeline

- TODO1: Divide the extractor agent
- TODO2: add search agent
- TODO3: Try LangGraph execution and branching.
- TODO4: Restructure the pipeline

A lightweight multi-agent pipeline that turns a paper in Markdown into a structured UPS-IR JSON and per-section artifacts.

## Quick Start
- Install deps: `pip install -r requirements.txt`
- Run full pipeline: `python main.py test2/part.md`
- Faster (skip image captioning): `python main.py test2/part.md --skip-annotation`
- Specify agents: `python main.py test2/part.md --agents reader,extractor,structurer`
- LangGraph mode: `python main.py test2/part.md --use-graph`

## What It Does
- Reader: load Markdown; optional image descriptions via `agents/picture.py` (needs `ARK_API_KEY`).
- Extractor: LLM extracts UPS-IR. Prompt enforces using entity ids in references (datasets/equations/experiments/relations).
- Structurer: normalize types, ids, metrics; writes `UPS-IR.json`.
- Verifier: schema + cross-reference check (accepts ids and some names/latex).
- Synthesizer: split UPS-IR into `UPS-IR_Output/<section>.json`.

## Useful Paths
- Input sample: `test2/part.md`
- Intermediates: `output/` (annotated md, extracted_info.json)
- Final: `UPS-IR.json`, `UPS-IR_Output/`
- Logs: `logs/agent_pipeline.log`

## Troubleshooting
- Missing md_path: run `python main.py test2/part.md`.
- JSON escape errors: extractor auto-fixes stray backslashes in LaTeX.
- Reference errors: ensure references use ids from their lists.
