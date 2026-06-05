"""
BDF Output Parser — 编译正则常量

合并 3 套解析器中最优的正则模式。
每个字段可能有多个候选正则（优先级从高到低）。
"""

import re

# 单精度浮点正则片段：匹配整数、小数、科学计数法
_FLOAT = r'[-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?'

# =============================================================================
# Energy — 优先级从 BDF 原生格式到通用格式
# =============================================================================

ENERGY_TOTAL: list[re.Pattern] = [
    re.compile(rf'E_tot\s*=\s*({_FLOAT})'),
    re.compile(rf'Total\s+energy\s*[:=]\s*({_FLOAT})', re.IGNORECASE),
    re.compile(rf'FINAL\s+ENERGY\s*[:=]\s*({_FLOAT})', re.IGNORECASE),
    re.compile(rf'SCF\s+energy\s*[:=]\s*({_FLOAT})', re.IGNORECASE),
    re.compile(rf'E\(SCF\)\s*=\s*({_FLOAT})', re.IGNORECASE),
]

ENERGY_ELECTRONIC = re.compile(rf'E_ele\s*=\s*({_FLOAT})', re.IGNORECASE)
ENERGY_NUCLEAR_REPULSION = re.compile(rf'E_nn\s*=\s*({_FLOAT})', re.IGNORECASE)
ENERGY_EXCHANGE = re.compile(rf'Exchange\s*=\s*({_FLOAT})', re.IGNORECASE)
ENERGY_CORRELATION = re.compile(rf'Correlation\s*=\s*({_FLOAT})', re.IGNORECASE)
ENERGY_MP2 = re.compile(rf'MP2\s+total\s+energy\s*=\s*({_FLOAT})', re.IGNORECASE)
ENERGY_KINETIC = re.compile(rf'Kinetic\s+energy\s*=\s*({_FLOAT})', re.IGNORECASE)
ENERGY_POTENTIAL = re.compile(rf'Potential\s+energy\s*=\s*({_FLOAT})', re.IGNORECASE)
ENERGY_SCF = re.compile(rf'SCF\s+energy\s*[:=]\s*({_FLOAT})', re.IGNORECASE)

# =============================================================================
# Convergence — BDF 正常终止 / SCF 收敛
# =============================================================================

CONVERGENCE_NORMAL = re.compile(
    r'Congratulations!\s+BDF\s+normal\s+termination', re.IGNORECASE
)

CONVERGENCE_BDF = re.compile(
    r'BDF\s+normal\s+termination', re.IGNORECASE
)

CONVERGENCE_FINAL_DELTA = re.compile(
    rf'Final\s+DeltaE\s*=\s*({_FLOAT}).*?'
    rf'Final\s+DeltaD\s*=\s*({_FLOAT})',
    re.IGNORECASE | re.DOTALL,
)

# =============================================================================
# Geometry — 4 种策略，按优先级排列
# =============================================================================

# 策略 0: Molecular Cartesian Coordinates (Angstrom) — 优化后的最终结构
# 匹配从标题到下一次 section 标记或文件结尾
GEOMETRY_ANGSTROM = re.compile(
    r'Molecular\s+Cartesian\s+Coordinates\s+\(X,Y,Z\)\s+in\s+Angstrom\s*:.*?(?=\n\n\S|\n\s+Force-RMS|\n\s+Redundant|\Z)',
    re.IGNORECASE | re.DOTALL,
)

GEOMETRY_ANGSTROM_HEADER = re.compile(
    r'Molecular\s+Cartesian\s+Coordinates\s+\(X,Y,Z\)\s+in\s+Angstrom\s*:',
    re.IGNORECASE,
)

# 策略 1: Cartcoord(Bohr) — BDF 标准坐标输出
GEOMETRY_BOHR = re.compile(
    r'Atom\s+Cartcoord\(Bohr\).*?(?=\n\n\S|\n\[|\n\|\||\nAtom\s+Cartcoord|\Z)',
    re.IGNORECASE | re.DOTALL,
)

# 策略 2: Optimized/Final geometry 关键词
GEOMETRY_FINAL = re.compile(
    r'(?:Optimized|Final|Converged).*?geometry.*?(?=\n\n|\n\[|\n\|\||$)',
    re.IGNORECASE | re.DOTALL,
)

# 策略 3: Geometry ... End geometry 块
GEOMETRY_INPUT_BLOCK = re.compile(
    r'Geometry\s*\n(.*?)End\s+geometry',
    re.IGNORECASE | re.DOTALL,
)

# 单精度浮点正则片段：匹配整数、小数、科学计数法
_FLOAT = r'[-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?'

# 坐标行匹配（元素 + 三个坐标）
COORD_LINE = re.compile(
    rf'^\s*([A-Z][a-z]?)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})',
    re.MULTILINE,
)

# BDF 编号格式: "    1       O        0.00000000          0.00000000          0.11730000"
COORD_LINE_NUMBERED = re.compile(
    rf'^\s*\d+\s+([A-Z][a-z]?)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})',
    re.MULTILINE,
)

# 坐标行带电荷列（Bohr 格式）
COORD_LINE_WITH_CHARGE = re.compile(
    rf'^\s*([A-Z][a-z]?)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})(?:\s+({_FLOAT}))?',
    re.MULTILINE,
)

GEOMETRY_CONVERGED = re.compile(
    r'Good\s+Job,\s+Geometry\s+Optimization\s+converged', re.IGNORECASE
)

GEOMETRY_NOT_CONVERGED = re.compile(
    r'Geometry\s+Optimization\s+not\s+converged', re.IGNORECASE
)

# 分子信息 — charge / multiplicity / point group
CHARGE = re.compile(r'Charge\s*=\s*([-+]?\d+)', re.IGNORECASE)
MULTIPLICITY = re.compile(r'Multiplicity\s*=\s*(\d+)', re.IGNORECASE)
POINT_GROUP = re.compile(r'Point\s+group\s*[:=]\s*(\S+)', re.IGNORECASE)

# =============================================================================
# Frequency — 逐行匹配
# =============================================================================

# 振动频率区段
VIB_SECTION = re.compile(
    r'Results\s+of\s+vibrations:.*?(?=Results\s+of\s+translations|$)',
    re.IGNORECASE | re.DOTALL,
)

# 平动/转动频率区段
TRANS_ROT_SECTION = re.compile(
    r'Results\s+of\s+translations\s+and\s+rotations:.*?(?=\n\s*\*\*\*|Thermal\s+Contributions|\n\s*\[|$)',
    re.IGNORECASE | re.DOTALL,
)

FREQUENCY_LINE = re.compile(
    r'^\s*Frequencies\s+(.*)$', re.MULTILINE
)

IR_INTENSITY_LINE = re.compile(
    r'^\s*IR intensities\s+(.*)$', re.MULTILINE
)

REDUCED_MASS_LINE = re.compile(
    r'^\s*Reduced masses\s+(.*)$', re.MULTILINE
)

FORCE_CONSTANT_LINE = re.compile(
    r'^\s*Force constants\s+(.*)$', re.MULTILINE
)

IRREP_LINE = re.compile(
    r'^\s*Irreps\s+(.*)$', re.MULTILINE
)

# 零点能 / 热力学
ZERO_POINT_ENERGY = re.compile(
    rf'Zero-point\s+Energy\s*=\s*({_FLOAT})', re.IGNORECASE
)
THERMAL_CORRECTION = re.compile(
    rf'Thermal\s+Correction\s*=\s*({_FLOAT})', re.IGNORECASE
)
ENTROPY = re.compile(
    rf'Entropy\s*=\s*({_FLOAT})', re.IGNORECASE
)
ENTHALPY = re.compile(
    rf'Enthalpy\s*=\s*({_FLOAT})', re.IGNORECASE
)
GIBBS_FREE_ENERGY = re.compile(
    rf'Gibbs\s+free\s+energy\s*=\s*({_FLOAT})', re.IGNORECASE
)

# =============================================================================
# TDDFT — 多 block 支持（按 Spin change 分割）
# =============================================================================

TDDFT_SPIN_CHANGE = re.compile(
    r'Spin change\s*:', re.IGNORECASE
)

TDDFT_HEADER = re.compile(
    r'No\.\s+Pair\s+ExSym', re.IGNORECASE
)

TDDFT_ISF = re.compile(r'isf\s*=?\s*([+-]?\d+)', re.IGNORECASE)
TDDFT_IALDA = re.compile(r'ialda\s*=?\s*([+-]?\d+)', re.IGNORECASE)
TDDFT_ITDA = re.compile(r'itda\s*=?\s*(\d+)', re.IGNORECASE)

# =============================================================================
# Optimization
# =============================================================================

OPT_KEYWORD = re.compile(
    r'Geometry\s+Optimization|BDFOPT', re.IGNORECASE
)

OPT_CONVERGED = re.compile(
    r'Good\s+Job,\s+Geometry\s+Optimization\s+converged', re.IGNORECASE
)

OPT_STEP = re.compile(
    r'Geometry\s+Optimization\s+step\s*:\s*(\d+)', re.IGNORECASE
)

OPT_ENERGY_CHANGE = re.compile(rf'Energy\s+change\s*=\s*({_FLOAT})', re.IGNORECASE)
OPT_MAX_FORCE = re.compile(rf'Max\s+force\s*=\s*({_FLOAT})', re.IGNORECASE)
OPT_RMS_FORCE = re.compile(rf'RMS\s+force\s*=\s*({_FLOAT})', re.IGNORECASE)
OPT_MAX_DISPLACEMENT = re.compile(rf'Max\s+displacement\s*=\s*({_FLOAT})', re.IGNORECASE)
OPT_RMS_DISPLACEMENT = re.compile(rf'RMS\s+displacement\s*=\s*({_FLOAT})', re.IGNORECASE)

# =============================================================================
# SCF
# =============================================================================

SCF_ITERATION = re.compile(
    r'diis/vshift\s+is\s+closed\s+at\s+iter\s*=\s*(\d+)', re.IGNORECASE
)

SCF_FINAL_RESULT = re.compile(
    r'Final\s+scf\s+result', re.IGNORECASE
)

SCF_DIIS_ERROR = re.compile(rf'DIIS\s+error\s*[:=]\s*({_FLOAT})', re.IGNORECASE)

# =============================================================================
# Thermochemistry
# =============================================================================

THERMO_E_ZPE = re.compile(rf'Electronic\s+\+\s+ZPE\s*=\s*({_FLOAT})', re.IGNORECASE)
THERMO_E_THERMAL = re.compile(rf'Electronic\s+\+\s+Thermal\s*=\s*({_FLOAT})', re.IGNORECASE)
THERMO_E_ENTHALPY = re.compile(rf'Electronic\s+\+\s+Enthalpy\s*=\s*({_FLOAT})', re.IGNORECASE)
THERMO_E_GIBBS = re.compile(rf'Electronic\s+\+\s+Gibbs\s*=\s*({_FLOAT})', re.IGNORECASE)

# Alternative TDDFT format: "No.     1    w=      9.8445 eV"
TDDFT_ALT_LINE = re.compile(
    r'No\.\s+(\d+)\s+w=\s+([-\d.]+)\s+eV\s+.*?f=\s+([-\d.]+)',
    re.IGNORECASE,
)

# =============================================================================
# Warnings / Errors
# =============================================================================

WARNING_PATTERNS: list[re.Pattern] = [
    re.compile(r'WARNING\s*:', re.IGNORECASE),
]

ERROR_PATTERNS: list[re.Pattern] = [
    re.compile(r'ERROR\s*:', re.IGNORECASE),
    re.compile(r'FATAL\s+error', re.IGNORECASE),
    re.compile(r'ABORT', re.IGNORECASE),
]

# =============================================================================
# Task type detection
# =============================================================================

TASK_TYPE_OPT = re.compile(r'Geometry\s+Optimization\s+step|BDFOPT', re.IGNORECASE)
TASK_TYPE_FREQ = re.compile(r'Results\s+of\s+vibrations', re.IGNORECASE)
TASK_TYPE_TDDFT = re.compile(r'TDDFT|TDDFT-SOC|Spin\s+change', re.IGNORECASE)

# =============================================================================
# SAO — Symmetry Adapted Orbital section
# =============================================================================

# Irrep header: "    Irrep A1         1    norb=   4"
SAO_IRREP_HEADER = re.compile(
    r'Irrep\s+(\S+)\s+(\d+)\s+norb\s*=\s*(\d+)'
)

# SAO label line: "  A1|1C1     1O2S0" or "  E1g|1C1     1U3D-1"
SAO_LABEL_LINE = re.compile(
    r'^\s*([A-Z]\w*)\|(\d+)C(\d+)\s+(.+)$'
)

# Individual AO label: "1O2P1" or "1O2S0"
SAO_AO_LABEL = re.compile(
    r'(\d+)([A-Z][a-z]?)(\d+)([SPDFGHI])(-?\d+)'
)

# Coefficient line: "            1.0000" or "            0.7071   0.7071"
SAO_COEFF_LINE = re.compile(
    r'^\s*([-+]?\d+\.\d+(?:\s+[-+]?\d+\.\d+)*)\s*$'
)

# Summary: "  Irrep :   A1        A2        B1        B2"
SAO_IRREP_NAMES = re.compile(
    r'Irrep\s*:\s*(.*)'
)

# Summary: "  Norb  :      4         0         2         1"
SAO_NORB_COUNTS = re.compile(
    r'Norb\s*:\s*(.*)'
)

# Point group: "  Point group name C(2V)"
SAO_POINT_GROUP = re.compile(
    r'Point\s+group\s+name\s+(\S+)', re.IGNORECASE
)

# Total basis: "  Total number of basis functions:       7       7"
SAO_NBASIS = re.compile(
    r'Total\s+number\s+of\s+basis\s+functions:\s+(\d+)', re.IGNORECASE
)
