"""
BDF Output Parser — Pydantic v2 数据模型

所有字段 Optional + None 默认，因为不同计算类型只产出部分字段。
原子字段名统一为 "element"。
所有能量单位 Hartree（BDF 原生）。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field


# =============================================================================
# Enums
# =============================================================================

class ParseStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    PARSE_ERROR = "parse_error"
    EMPTY = "empty"


class TaskType(str, Enum):
    SINGLE_POINT = "single_point"
    GEOMETRY_OPT = "geometry_optimization"
    FREQUENCY = "frequency"
    OPT_FREQ = "opt_freq"
    TDDFT = "tddft"
    UNKNOWN = "unknown"


# =============================================================================
# Sub-models
# =============================================================================

class Atom(BaseModel):
    """单个原子坐标"""
    element: str
    x: float
    y: float
    z: float
    units: Literal["bohr", "angstrom"] = "bohr"


class EnergyData(BaseModel):
    """BDF 输出中所有能量相关字段"""
    total_energy: Optional[float] = None
    scf_energy: Optional[float] = None
    electronic_energy: Optional[float] = None
    nuclear_repulsion: Optional[float] = None
    exchange: Optional[float] = None
    correlation: Optional[float] = None
    mp2_energy: Optional[float] = None
    kinetic_energy: Optional[float] = None
    potential_energy: Optional[float] = None


class GeometryData(BaseModel):
    """分子几何结构"""
    atoms: list[Atom] = []
    charge: Optional[float] = None
    multiplicity: Optional[int] = None
    point_group: Optional[str] = None

    @computed_field
    @property
    def natoms(self) -> int:
        return len(self.atoms)

    @computed_field
    @property
    def formula(self) -> str:
        counts: dict[str, int] = {}
        for a in self.atoms:
            counts[a.element] = counts.get(a.element, 0) + 1
        parts = []
        for elem in sorted(counts):
            n = counts[elem]
            parts.append(elem if n == 1 else f"{elem}{n}")
        return "".join(parts)


class FrequencyData(BaseModel):
    """频率计算结果"""
    frequencies: list[float] = []
    ir_intensities: list[float] = []
    reduced_masses: list[float] = []
    force_constants: list[float] = []
    irreps: list[str] = []
    translations_rotations: list[float] = []
    zero_point_energy: Optional[float] = None
    thermal_correction: Optional[float] = None
    entropy: Optional[float] = None
    enthalpy: Optional[float] = None
    gibbs_free_energy: Optional[float] = None

    @computed_field
    @property
    def imaginary_frequencies(self) -> list[float]:
        return [f for f in self.frequencies if f < 0]

    @computed_field
    @property
    def n_imaginary(self) -> int:
        return sum(1 for f in self.frequencies if f < 0)

    @computed_field
    @property
    def is_stable(self) -> bool:
        return self.n_imaginary == 0


class ThermochemistryData(BaseModel):
    """完整热化学数据"""
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    electronic_plus_zpe: Optional[float] = None
    electronic_plus_thermal: Optional[float] = None
    electronic_plus_enthalpy: Optional[float] = None
    electronic_plus_gibbs: Optional[float] = None


class ExcitedState(BaseModel):
    """单个 TDDFT 激发态"""
    index: int
    symmetry: Optional[str] = None
    energy_ev: float
    wavelength_nm: float
    oscillator_strength: float
    delta_s2: Optional[float] = None
    dominant_transition: Optional[str] = None


class TDDFTBlock(BaseModel):
    """一个 TDDFT 计算块（支持多 isf/ialda）"""
    isf: Optional[int] = None
    ialda: Optional[int] = None
    itda: Optional[int] = None
    tda: bool = False
    method: Optional[str] = None
    states: list[ExcitedState] = []


class OptimizationData(BaseModel):
    """几何优化结果"""
    converged: bool = False
    n_steps: int = 0
    final_energy: Optional[float] = None
    energy_change: Optional[float] = None
    max_force: Optional[float] = None
    rms_force: Optional[float] = None
    max_displacement: Optional[float] = None
    rms_displacement: Optional[float] = None


class SCFData(BaseModel):
    """SCF 收敛信息"""
    converged: bool = False
    n_iterations: int = 0
    final_energy: Optional[float] = None
    diis_error: Optional[float] = None


# =============================================================================
# Top-level result
# =============================================================================

class BDFParseResult(BaseModel):
    """BDF 输出解析完整结果 — 标准化 JSON schema"""
    status: ParseStatus = ParseStatus.EMPTY
    task_type: TaskType = TaskType.UNKNOWN

    energies: EnergyData = Field(default_factory=EnergyData)
    geometry: GeometryData = Field(default_factory=GeometryData)
    frequencies: FrequencyData = Field(default_factory=FrequencyData)
    thermochemistry: ThermochemistryData = Field(default_factory=ThermochemistryData)
    tddft_blocks: list[TDDFTBlock] = []
    optimization: OptimizationData = Field(default_factory=OptimizationData)
    scf: SCFData = Field(default_factory=SCFData)

    warnings: list[str] = []
    errors: list[str] = []
    source_file: Optional[str] = None

    @computed_field
    @property
    def is_success(self) -> bool:
        return self.status == ParseStatus.SUCCESS

    @computed_field
    @property
    def excited_states(self) -> list[ExcitedState]:
        """展平所有 TDDFT 块（向后兼容）"""
        return [s for blk in self.tddft_blocks for s in blk.states]
