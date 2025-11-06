"""Agent package for the UPS-IR pipeline.

为避免在导入包时加载所有子模块（及其第三方依赖），此 __init__ 不再执行
`from .xxx import ...`。请在需要的地方显式按子模块导入，例如：

    from agents.reader import ReaderAgent, ReaderConfig
    from agents.extractor import ExtractorAgent, ExtractorConfig

这样可以在只运行 Reader 时无需安装 LLM 相关依赖。
"""

__all__ = []
