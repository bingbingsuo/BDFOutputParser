"""
BDF Output Parser — Pydantic v2 数据模型

所有字段 Optional + None 默认，因为不同计算类型只产出部分字段。
原子字段名统一为 "element"。
所有能量单位 Hartree（BDF 原生）。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

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


class CVTransition(BaseModel):
    """TDDFT 激发态的单个组态矢量 (configuration vector) 跃迁"""
    from_orbital: str = ""          # e.g. "A1(   3 )"
    to_orbital: str = ""            # e.g. "A1(   4 )"
    coefficient: float = 0.0        # CI 系数
    percentage: float = 0.0         # 贡献百分比 (%)
    ipa_ev: float = 0.0             # 独立粒子近似能量 (eV)
    oai: Optional[float] = None     # 轨道对重叠积分


class ExcitedState(BaseModel):
    """单个 TDDFT 激发态"""
    index: int
    symmetry: Optional[str] = None
    energy_ev: float
    wavelength_nm: float
    oscillator_strength: float
    delta_s2: Optional[float] = None
    dominant_transition: Optional[str] = None
    total_energy_au: Optional[float] = None   # 激发态总能量 (a.u.)
    ova: Optional[float] = None               # 绝对重叠积分 (0=CT, 1=local)
    cv_transitions: list[CVTransition] = []    # 组态矢量跃迁列表
    label: str = ""                            # 全局排序标签: S1/S2/T1/T2


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
    homo_energy: Optional[float] = None      # HOMO 能量 (eV)
    lumo_energy: Optional[float] = None      # LUMO 能量 (eV)
    homo_lumo_gap: Optional[float] = None    # HOMO-LUMO gap (eV)


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


# =============================================================================
# Unified Result — .out + .out.tmp + .bdfh5 normalized interface
# =============================================================================

class UnifiedRunStatus(str, Enum):
    """BDF execution state normalized from HDF5/output/process evidence."""

    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    RUNNING = "running"
    CRASHED = "crashed"
    KILLED = "killed"
    UNKNOWN = "unknown"


class UnifiedParseStatus(str, Enum):
    """Parser coverage state for available BDF artifacts."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"


class UnifiedResultStatus(str, Enum):
    """Whether the result is usable for the requested calculation step."""

    USABLE = "usable"
    USABLE_WITH_WARNINGS = "usable_with_warnings"
    INCOMPLETE_RESTARTABLE = "incomplete_restartable"
    NOT_USABLE = "not_usable"
    UNKNOWN = "unknown"


class ConsistencyWarning(BaseModel):
    """Structured warning for conflicting HDF5/output facts."""

    field: str
    hdf5_value: Any = None
    output_value: Any = None
    tolerance: float | None = None
    severity: str = "warning"
    message: str = ""


class BDFUnifiedResult(BaseModel):
    """Versioned result interface consumed by BDFAssistant.

    This model is owned by BDFOutputParser.  It intentionally stores domain
    sections as dictionaries in the first slice so new BDF modules can be added
    without changing the top-level API.
    """

    schema_version: str = "BDFUnifiedResult-v0.1"
    parser_version: str = ""
    task_type: str | None = None
    run_status: UnifiedRunStatus = UnifiedRunStatus.UNKNOWN
    parse_status: UnifiedParseStatus = UnifiedParseStatus.UNAVAILABLE
    result_status: UnifiedResultStatus = UnifiedResultStatus.UNKNOWN
    success: bool = False

    execution: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    restart: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    field_sources: dict[str, str] = Field(default_factory=dict)
    consistency_warnings: list[ConsistencyWarning] = Field(default_factory=list)
    raw_refs: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# SAO — Symmetry Adapted Orbital (checksymm COMPASS output)
# =============================================================================

class AOLabel(BaseModel):
    """单个 AO 标签: "1O2P1" → atom=1, element=O, n=2, l=P, m=1"""
    atom_index: int
    element: str
    n: int                          # 主量子数
    l: str                          # 角量子数字母: S/P/D/F
    m: int                          # 磁量子数
    coeff: float                    # 在 SAO 中的系数


class SAOLine(BaseModel):
    """单个 symmetry-adapted orbital 的 AO 组成"""
    label: str                      # "A1|1C1"
    irrep: str                      # "A1"
    irrep_index: int                # 1-based
    component: int                  # 简并分量 (1 or 2 for degenerate irreps)
    aos: list[AOLabel] = []


class IrrepSAO(BaseModel):
    """一个不可约表示中所有 SAO 的信息"""
    irrep: str                      # "A1"
    irrep_index: int                # 1
    norb: int                       # 该 irrep 的轨道数
    saos: list[SAOLine] = []


class SAOParseResult(BaseModel):
    """checksymm COMPASS 输出中 SAO 区块的解析结果"""
    point_group: Optional[str] = None
    n_irreps: int = 0
    n_basis: int = 0
    irreps: list[IrrepSAO] = []


# =============================================================================
# Orbital Classification
# =============================================================================

class IrrepClassification(BaseModel):
    """单个不可约表示中的轨道五层级分类"""
    irrep: str                      # "A1"
    norb: int                       # 总轨道数（基函数数）
    frozen_core_labels: list[str] = []   # frozen core AO 标签
    outer_core_labels: list[str] = []    # outer core AO 标签
    inactive_labels: list[str] = []      # inactive（满占据 valence shell）
    active_labels: list[str] = []        # active（部分占据 valence shell）
    virtual_labels: list[str] = []       # virtual（未占据基函数）
    n_frozen_core_orbitals: int = 0     # SAO 数
    n_outer_core_orbitals: int = 0
    n_inactive_orbitals: int = 0
    n_active_orbitals: int = 0
    n_virtual_orbitals: int = 0
    # 每 tier 的原子壳层组成摘要: {"frozen_core": {"U": "5s,5p,5d", "F": "1s"}, ...}
    summary: dict[str, dict[str, str]] = {}


class OrbitalClassification(BaseModel):
    """分子轨道五层级分类"""
    molecule: str = ""
    point_group: str = ""
    n_electrons: int = 0
    n_frozen_core_electrons: int = 0
    n_outer_core_electrons: int = 0
    n_inactive_electrons: int = 0
    n_active_electrons: int = 0
    total_inactive_orbitals: int = 0
    total_active_orbitals: int = 0
    total_virtual_orbitals: int = 0
    n_basis: int = 0                    # 总基函数数
    per_irrep: list[IrrepClassification] = []
    # 全局 summary: 跨 irrep 合并的原子壳层组成
    summary: dict[str, dict[str, str]] = {}
    # 给用户的提示（如全满价层轨道被纳入活性空间，供用户抉择是否收窄）
    notes: list[str] = []


class CoreStateSummary(BaseModel):
    """BDFCoreState `.bdfh5` 的归一化只读摘要（接口契约 §5）。

    所有字段可空——BDFCoreState 仍在演进，缺失字段容忍。fail-closed：
    h5py 缺 / 文件缺 / schema 不符时 available=False。
    """
    available: bool = False
    reason: str = ""        # ok | h5py_unavailable | file_not_found | unsupported_schema | read_error
    path: str = ""          # .bdfh5 路径（审计）
    core_state_schema: str = ""  # /meta/core_state_schema（避免遮蔽 BaseModel.schema）
    status: str = ""        # /run/status: completed|failed|interrupted|running
    input_mode: str = ""    # /input/input_mode
    current_context_id: Optional[int] = None
    context_count: Optional[int] = None
    current_module: str = ""
    last_successful_module: str = ""
    failed_module: str = ""
    interrupted_module: str = ""
    restartable: Optional[bool] = None
    started_unix: Optional[int] = None
    completed_unix: Optional[int] = None
    elapsed_sec: Optional[float] = None
    workflow: dict = {}     # {workflow_id, status, module_plan:[...], modules:{...}}
    aliases: dict = {}      # {orbitals:{...}, geometries:{...}, chkfil:{...}}
    objects: dict = {}      # {orbitals:[...], geometries:[...], chkfil_snapshots:[...]}
    files: dict = {}        # {by_role:{chkfil:[...], geometry:[...], ...}}
    diagnostics: dict = {}  # {<module>:{last_failure:{...}}} 失败时
    provenance: dict = {}   # {build:{...}, runtime:{...}} 关键字段摘要
    restart: dict = {}      # Phase 6: /restart/{scratch,modules,assets} restart capability
