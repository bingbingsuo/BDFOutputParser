# BDFOutputParser Development Guide

## Overview

统一 BDF 量子化学输出解析库，合并 BDFAssistant、BDFExecute、BDFEasyInput 三套解析器的最优逻辑。

## Architecture

```
BDF 输出文本 (.out / 原始字符串)
    ↓
BDFOutputParser.parse()
    ├── Pydantic v2 模型 (BDFParseResult)
    │   └── model_dump_json() → 标准化 JSON (数据存档 / agent 间传递)
    └── MarkdownReporter.render()
        └── 人类可读 Markdown 报告 (zh/en)
```

## Key Modules

| 模块 | 职责 |
|------|------|
| `models.py` | Pydantic v2 数据模型：Atom, EnergyData, GeometryData, FrequencyData, ExcitedState, TDDFTBlock, OptimizationData, SCFData, BDFParseResult |
| `patterns.py` | 编译正则常量，合并最优正则模式 |
| `parser.py` | `BDFOutputParser` 主类，自动检测 task_type |
| `reporters/markdown.py` | `MarkdownReporter` 中英文 Markdown 报告 |

## API

```python
from bdf_output_parser import BDFOutputParser, MarkdownReporter

parser = BDFOutputParser()
result = parser.parse(bdf_output_text)
# or: result = parser.parse_file("bdf.out")

# Schema JSON
json_str = result.model_dump_json(exclude_none=True, indent=2)

# Markdown 报告
md = MarkdownReporter(language="zh").render(result)
# or: MarkdownReporter(language="en").render(result)
```

## Extracted Fields

| Section | Key Fields |
|---------|-----------|
| energies | total_energy, scf_energy, electronic_energy, nuclear_repulsion, exchange, correlation, mp2_energy |
| geometry | atoms (element, x, y, z, units), charge, multiplicity, point_group, formula |
| frequencies | frequencies, ir_intensities, reduced_masses, force_constants, irreps, zpe, thermal, entropy, imaginary |
| thermochemistry | electronic_plus_zpe, electronic_plus_thermal, electronic_plus_enthalpy, electronic_plus_gibbs |
| tddft | blocks with isf/ialda, per-state energy_ev, wavelength_nm, oscillator_strength, dominant_transition |
| optimization | converged, n_steps, final_energy, max_force, rms_force |
| scf | converged, n_iterations, final_energy, diis_error |

## Testing

```bash
cd ~/bdf/BDFOutputParser
python -m pytest tests/ -v
```

80 tests covering all extraction sections.

## Dependencies

- Python ≥3.10, Pydantic v2
- No other dependencies — standalone package
