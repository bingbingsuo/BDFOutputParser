# BDFOutputParser

统一 BDF 量子化学输出解析库，供 BDFAssistant、BDFExecute 等共享使用。

**仓库**: [github.com/bingbingsuo/BDFOutputParser](https://github.com/bingbingsuo/BDFOutputParser)

合并 BDFAssistant、BDFExecute、BDFEasyInput 三套解析器的最优正则和提取逻辑。

## 安装

```bash
pip install -e ~/bdf/BDFOutputParser
```

依赖：Python ≥3.10, Pydantic v2。

## 快速开始

```python
from bdf_output_parser import BDFOutputParser, MarkdownReporter

parser = BDFOutputParser()

# 从字符串解析
result = parser.parse(bdf_output_text)

# 或从文件解析
result = parser.parse_file("bdf.out")

# 标准化 JSON（数据存档、agent 间传递）
json_str = result.model_dump_json(exclude_none=True, indent=2)

# Markdown 报告（人类可读）
md = MarkdownReporter(language="zh").render(result)
# or: MarkdownReporter(language="en").render(result)
```

## 提取内容

| 类别 | 字段 |
|------|------|
| 能量 | total_energy, scf_energy, electronic_energy, nuclear_repulsion, exchange, correlation, mp2_energy |
| 几何 | atoms (element, x, y, z, units), charge, multiplicity, point_group, formula |
| 频率 | frequencies, ir_intensities, reduced_masses, force_constants, irreps, zpe, thermal, entropy |
| 热化学 | electronic_plus_zpe, electronic_plus_thermal, electronic_plus_enthalpy, electronic_plus_gibbs |
| TDDFT | blocks (isf/ialda), states (energy_ev, wavelength_nm, oscillator_strength, dominant_transition) |
| 优化 | converged, n_steps, final_energy, max_force, rms_force |
| SCF | converged, n_iterations, final_energy, diis_error |

## JSON 输出示例

```json
{
  "status": "success",
  "task_type": "geometry_optimization",
  "energies": {
    "total_energy": -76.56789012,
    "electronic_energy": -76.54321000
  },
  "geometry": {
    "atoms": [
      {"element": "O", "x": 0.0, "y": 0.0, "z": 0.1168, "units": "angstrom"},
      {"element": "H", "x": 0.0, "y": 0.7615, "z": -0.4671, "units": "angstrom"},
      {"element": "H", "x": 0.0, "y": -0.7615, "z": -0.4671, "units": "angstrom"}
    ],
    "charge": 0.0,
    "multiplicity": 1,
    "formula": "H2O"
  },
  "optimization": {
    "converged": true,
    "n_steps": 3
  },
  "scf": {
    "converged": true,
    "n_iterations": 8
  }
}
```

## 测试

```bash
cd ~/bdf/BDFOutputParser
python -m pytest tests/ -v
```

80 个测试，覆盖所有提取类别。

## 架构

```
BDF 输出文本 (.out / 原始字符串)
    ↓
BDFOutputParser.parse()
    ├── Pydantic v2 模型 (BDFParseResult)
    │   └── model_dump_json() → 标准化 JSON
    └── MarkdownReporter.render()
        └── 人类可读 Markdown (zh/en)
```

## 项目来源

本库合并了以下项目的输出解析逻辑：

| 来源 | 文件 | 行数 |
|------|------|------|
| BDFAssistant | `result_analyzer.py` + `extractor.py` | ~1300 |
| BDFExecute | `extractor.py` | ~700 |
| BDFEasyInput (归档) | `output_parser.py` | ~2400 |

## 许可

Proprietary — 为 BDF 量子化学软件生态系统内部使用。
