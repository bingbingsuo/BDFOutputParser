"""
BDFOutputParser — 统一 BDF 量子化学输出解析库

两种输出:
  - Schema JSON: result.model_dump_json() — 标准化结构化数据
  - Markdown:    MarkdownReporter().render(result) — 人类可读报告
"""

from .parser import BDFOutputParser
from .reporters.markdown import MarkdownReporter
from .models import (
    BDFParseResult,
    ParseStatus,
    TaskType,
    Atom,
    EnergyData,
    GeometryData,
    FrequencyData,
    ExcitedState,
    TDDFTBlock,
    OptimizationData,
    SCFData,
)

__all__ = [
    "BDFOutputParser",
    "MarkdownReporter",
    "BDFParseResult",
    "ParseStatus",
    "TaskType",
    "Atom",
    "EnergyData",
    "GeometryData",
    "FrequencyData",
    "ExcitedState",
    "TDDFTBlock",
    "OptimizationData",
    "SCFData",
]
