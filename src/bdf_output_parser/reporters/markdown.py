"""
BDF Output Parser — Markdown 报告生成器

将 BDFParseResult 渲染为人类可读的 Markdown 报告。
"""

from __future__ import annotations

from typing import Literal

from ..models import BDFParseResult, ParseStatus, TaskType


_TASK_TYPE_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "single_point": "单点能计算",
        "geometry_optimization": "几何优化",
        "frequency": "频率分析",
        "opt_freq": "优化+频率",
        "tddft": "TDDFT 激发态",
        "unknown": "未知",
    },
    "en": {
        "single_point": "Single Point",
        "geometry_optimization": "Geometry Optimization",
        "frequency": "Frequency",
        "opt_freq": "Opt+Freq",
        "tddft": "TDDFT Excited State",
        "unknown": "Unknown",
    },
}


class MarkdownReporter:
    """生成 BDF 计算结果的人类可读 Markdown 报告"""

    def __init__(self, language: Literal["zh", "en"] = "zh"):
        """
        Args:
            language: 报告语言。"zh" 为中文（默认），"en" 为英文。
        """
        self._lang = language
        self._task_labels = _TASK_TYPE_LABELS[language]

    def render(self, result: BDFParseResult) -> str:
        """渲染完整 Markdown 报告"""
        if result.status == ParseStatus.EMPTY:
            return self._render_empty()
        return self._render_full(result)

    # =========================================================================
    # Sections
    # =========================================================================

    def _render_empty(self) -> str:
        if self._lang == "zh":
            return "## 无计算结果\n\n未从 BDF 输出中提取到任何数据。"
        return "## No Results\n\nNo data extracted from BDF output."

    def _render_full(self, result: BDFParseResult) -> str:
        lines: list[str] = []
        lines.append(self._section_header(result))
        lines.append(self._section_summary(result))
        lines.append(self._section_energy(result))
        lines.append(self._section_scf(result))
        lines.append(self._section_geometry(result))
        lines.append(self._section_frequency(result))
        lines.append(self._section_thermochemistry(result))
        lines.append(self._section_tddft(result))
        lines.append(self._section_optimization(result))
        lines.append(self._section_warnings(result))
        return "\n\n".join(l for l in lines if l)

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    def _section_header(self, result: BDFParseResult) -> str:
        status_map = {
            ParseStatus.SUCCESS: ("success", "success"),
            ParseStatus.PARTIAL: ("warning", "partial"),
            ParseStatus.PARSE_ERROR: ("error", "error"),
            ParseStatus.EMPTY: ("error", "empty"),
        }
        _, status_text = status_map.get(result.status, ("", "unknown"))

        task_label = self._task_labels.get(result.task_type.value, result.task_type.value)

        if self._lang == "zh":
            return f"## BDF 计算结果\n\n**状态**: {status_text} | **任务类型**: {task_label}"
        return f"## BDF Calculation Result\n\n**Status**: {status_text} | **Task Type**: {task_label}"

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _section_summary(self, result: BDFParseResult) -> str:
        e = result.energies
        metrics: list[tuple[str, str]] = []

        if e.total_energy is not None:
            label = "Total Energy (Hartree)" if self._lang == "en" else "总能量 (Hartree)"
            metrics.append((label, f"{e.total_energy:.8f}"))

        if result.optimization.converged:
            label = "Optimization Converged" if self._lang == "en" else "优化收敛"
            metrics.append((label, "yes" if self._lang == "en" else "是"))

        if result.frequencies.is_stable and result.frequencies.frequencies:
            label = "Freq Stability" if self._lang == "en" else "频率稳定性"
            metrics.append((label, "stable" if self._lang == "en" else "稳定"))

        if result.scf.converged:
            label = "SCF Converged" if self._lang == "en" else "SCF 收敛"
            metrics.append((label, "yes" if self._lang == "en" else "是"))

        if result.excited_states:
            label = "Excited States" if self._lang == "en" else "激发态数量"
            metrics.append((label, str(len(result.excited_states))))

        if not metrics:
            return ""

        table_header = "| Metric | Value |\n|--------|-------|"
        rows = [f"| {k} | {v} |" for k, v in metrics]

        header = "Key Metrics" if self._lang == "en" else "关键指标"
        return f"### {header}\n\n{table_header}\n" + "\n".join(rows)

    # -------------------------------------------------------------------------
    # Energy
    # -------------------------------------------------------------------------

    def _section_energy(self, result: BDFParseResult) -> str:
        e = result.energies
        rows: list[tuple[str, str]] = []
        fields = [
            ("total_energy", "Total Energy", "总能量"),
            ("scf_energy", "SCF Energy", "SCF 能量"),
            ("electronic_energy", "Electronic Energy", "电子能量"),
            ("nuclear_repulsion", "Nuclear Repulsion", "核排斥能"),
            ("exchange", "Exchange", "交换能"),
            ("correlation", "Correlation", "关联能"),
            ("mp2_energy", "MP2 Energy", "MP2 能量"),
            ("kinetic_energy", "Kinetic Energy", "动能"),
            ("potential_energy", "Potential Energy", "势能"),
        ]
        for attr, en, zh in fields:
            v = getattr(e, attr)
            if v is not None:
                rows.append((en if self._lang == "en" else zh, f"{v:.8f} Hartree"))

        if not rows:
            return ""

        header = "Energy" if self._lang == "en" else "能量"
        table_header = "| Component | Value |\n|-----------|-------|"
        row_lines = [f"| {k} | {v} |" for k, v in rows]
        return f"### {header}\n\n{table_header}\n" + "\n".join(row_lines)

    # -------------------------------------------------------------------------
    # SCF
    # -------------------------------------------------------------------------

    def _section_scf(self, result: BDFParseResult) -> str:
        s = result.scf
        if s.n_iterations == 0 and s.final_energy is None:
            return ""

        if self._lang == "zh":
            lines = ["### SCF 收敛"]
        else:
            lines = ["### SCF Convergence"]

        conv = ("是" if s.converged else "否") if self._lang == "zh" else ("Yes" if s.converged else "No")
        if self._lang == "zh":
            lines.append(f"- 收敛: {conv}")
            if s.n_iterations:
                lines.append(f"- 迭代次数: {s.n_iterations}")
        else:
            lines.append(f"- Converged: {conv}")
            if s.n_iterations:
                lines.append(f"- Iterations: {s.n_iterations}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    def _section_geometry(self, result: BDFParseResult) -> str:
        geo = result.geometry
        if not geo.atoms:
            return ""

        if self._lang == "zh":
            lines = ["### 几何结构"]
            if geo.point_group:
                lines.append(f"- 点群: {geo.point_group}")
            lines.append(f"- 原子数: {geo.natoms}")
            lines.append(f"- 分子式: {geo.formula}")
            lines.append("")
            lines.append("| # | Element | X | Y | Z |")
            lines.append("|---|---------|---|---|---|")
        else:
            lines = ["### Geometry"]
            if geo.point_group:
                lines.append(f"- Point Group: {geo.point_group}")
            lines.append(f"- Atoms: {geo.natoms}")
            lines.append(f"- Formula: {geo.formula}")
            lines.append("")
            lines.append("| # | Element | X | Y | Z |")
            lines.append("|---|---------|---|---|---|")

        for i, a in enumerate(geo.atoms, 1):
            lines.append(f"| {i} | {a.element} | {a.x:.6f} | {a.y:.6f} | {a.z:.6f} |")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Frequency
    # -------------------------------------------------------------------------

    def _section_frequency(self, result: BDFParseResult) -> str:
        freq = result.frequencies
        if not freq.frequencies:
            return ""

        if self._lang == "zh":
            lines = ["### 频率分析"]
        else:
            lines = ["### Frequency Analysis"]

        if freq.zero_point_energy is not None:
            label = "Zero-Point Energy" if self._lang == "en" else "零点能"
            zpe_val = f"{freq.zero_point_energy:.6f} Hartree"
            lines.append(f"- {label}: {zpe_val}")

        if freq.imaginary_frequencies:
            imag = freq.imaginary_frequencies
            imag_str = ", ".join(f"{f:.1f}i" for f in imag)
            if self._lang == "zh":
                lines.append(f"- ⚠ 虚频: {imag_str}")
                lines.append("  > 检测到虚频，结构可能不是稳定的极小点")
            else:
                lines.append(f"- ⚠ Imaginary freqs: {imag_str}")
                lines.append("  > Structure may not be a stable minimum")
        elif freq.frequencies:
            if self._lang == "zh":
                lines.append("- ✓ 无虚频，是稳定的极小点")
            else:
                lines.append("- ✓ Stable minimum (no imaginary frequencies)")

        lines.append("")
        if self._lang == "zh":
            lines.append("| # | 频率 (cm⁻¹) | IR 强度 (km/mol) | Irrep |")
            lines.append("|---|-------------|-----------------|-------|")
        else:
            lines.append("| # | Freq (cm⁻¹) | IR Int (km/mol) | Irrep |")
            lines.append("|---|------------|----------------|-------|")

        for i, f in enumerate(freq.frequencies):
            ir = freq.ir_intensities[i] if i < len(freq.ir_intensities) else None
            ir_str = f"{ir:.4f}" if ir is not None else "-"
            irrep = freq.irreps[i] if i < len(freq.irreps) else "-"
            lines.append(f"| {i+1} | {f:.4f} | {ir_str} | {irrep} |")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Thermochemistry
    # -------------------------------------------------------------------------

    def _section_thermochemistry(self, result: BDFParseResult) -> str:
        t = result.thermochemistry
        has_any = any(
            v is not None
            for v in [t.electronic_plus_zpe, t.electronic_plus_thermal,
                      t.electronic_plus_enthalpy, t.electronic_plus_gibbs]
        )
        if not has_any:
            return ""

        if self._lang == "zh":
            lines = ["### 热化学数据"]
        else:
            lines = ["### Thermochemistry"]

        entries = [
            ("electronic_plus_zpe", "E + ZPE"),
            ("electronic_plus_thermal", "E + Thermal"),
            ("electronic_plus_enthalpy", "E + Enthalpy"),
            ("electronic_plus_gibbs", "E + Gibbs"),
        ]
        for attr, label in entries:
            v = getattr(t, attr)
            if v is not None:
                lines.append(f"- {label}: {v:.8f} Hartree")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # TDDFT
    # -------------------------------------------------------------------------

    def _section_tddft(self, result: BDFParseResult) -> str:
        states = result.excited_states
        if not states:
            return ""

        if self._lang == "zh":
            lines = ["### TDDFT 激发态"]
        else:
            lines = ["### TDDFT Excited States"]

        # Show per-block info if multiple blocks
        if result.tddft_blocks:
            for blk in result.tddft_blocks:
                if blk.isf is not None:
                    isf_label = f"isf={blk.isf}"
                    lines.append(f"\n**{isf_label}**")

        if self._lang == "zh":
            lines.append("")
            lines.append("| # | Sym | E (eV) | λ (nm) | f | Dom |")
            lines.append("|---|-----|--------|--------|-----|-----|")
        else:
            lines.append("")
            lines.append("| # | Sym | E (eV) | λ (nm) | f | Dom |")
            lines.append("|---|-----|--------|--------|-----|-----|")

        for s in states:
            sym = s.symmetry or "-"
            dom = s.dominant_transition or "-"
            lines.append(
                f"| {s.index} | {sym} | {s.energy_ev:.4f} | {s.wavelength_nm:.2f} | {s.oscillator_strength:.4f} | {dom} |"
            )

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    def _section_optimization(self, result: BDFParseResult) -> str:
        opt = result.optimization
        if opt.n_steps == 0 and not opt.converged:
            return ""

        if self._lang == "zh":
            lines = ["### 几何优化"]
        else:
            lines = ["### Geometry Optimization"]

        conv = ("已收敛" if opt.converged else "未收敛") if self._lang == "zh" else \
               ("Converged" if opt.converged else "Not Converged")
        lines.append(f"- {conv} ({opt.n_steps} steps)")

        if opt.final_energy is not None:
            label = "Final Energy" if self._lang == "en" else "最终能量"
            lines.append(f"- {label}: {opt.final_energy:.8f} Hartree")

        if opt.energy_change is not None:
            label = "Energy Change" if self._lang == "en" else "能量变化"
            lines.append(f"- {label}: {opt.energy_change:.6e}")

        if opt.max_force is not None:
            label = "Max Force" if self._lang == "en" else "最大受力"
            lines.append(f"- {label}: {opt.max_force:.6e}")

        if opt.rms_force is not None:
            label = "RMS Force" if self._lang == "en" else "RMS 受力"
            lines.append(f"- {label}: {opt.rms_force:.6e}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Warnings
    # -------------------------------------------------------------------------

    def _section_warnings(self, result: BDFParseResult) -> str:
        all_warnings = result.warnings + result.errors
        if not all_warnings:
            return ""

        if self._lang == "zh":
            lines = ["### 警告/错误"]
        else:
            lines = ["### Warnings / Errors"]

        for w in all_warnings:
            lines.append(f"- {w}")

        return "\n".join(lines)
