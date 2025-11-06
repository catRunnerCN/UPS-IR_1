from __future__ import annotations

import json
import sys
import pathlib
from pathlib import Path

# 把项目根目录加入 sys.path，便于从 scripts/ 下导入 agents 包
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from agents.extractor import ExtractorAgent, ExtractorConfig
from langchain_core.runnables import RunnableLambda


def main():
    # 读取一小段 md 作为输入，避免处理过长文本
    md_path = Path('test2/part.md')
    text = md_path.read_text(encoding='utf-8')[:5000]

    # 用 RunnableLambda 模拟 LLM，返回符合预期结构的 JSON 字符串
    def fake_llm(_):
        payload = {
            "meta": {"title": "Demo", "authors": ["Tester"], "venue": "Arxiv", "year": 2024},
            "tasks": [{"id": "t1", "name": "Classification"}],
            "methods": [{"id": "m1", "name": "DemoNet", "uses_datasets": ["d1"], "uses_equations": ["e1"]}],
            "datasets": [{"id": "d1", "name": "CIFAR-10", "split": "test"}],
            "equations": [{"id": "e1", "latex": "a=b+c", "units": None}],
            "experiments": [{"method": "m1", "dataset": "d1", "metrics": [{"acc": 0.9}]}],
            "relations": [{"from": "m1", "to": "d1", "type": "uses"}],
        }
        return json.dumps(payload, ensure_ascii=False)

    llm_stub = RunnableLambda(fake_llm)

    agent = ExtractorAgent(
        config=ExtractorConfig(output_path=Path('output/extracted_info.json')),
        llm=llm_stub,
    )

    state = {"text": text}
    state = agent.run(state)

    print('OK ->', state['artifacts']['extracted_json'])


if __name__ == '__main__':
    main()
