"""
BDF Output Parser — 主解析器

合并 BDFAssistant、BDFExecuteAgent、BDFEasyInput 三套解析器的最优逻辑。
自动检测 task_type，无需调用方传入。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .models import (
    AOLabel,
    Atom,
    BDFParseResult,
    EnergyData,
    ExcitedState,
    FrequencyData,
    GeometryData,
    IrrepSAO,
    OptimizationData,
    ParseStatus,
    SAOLine,
    SAOParseResult,
    SCFData,
    TaskType,
    TDDFTBlock,
    ThermochemistryData,
)
from . import patterns as P


class BDFOutputParser:
    """
    统一 BDF 输出解析器。

    Usage:
        parser = BDFOutputParser()
        result = parser.parse(bdf_output_text)
        # or
        result = parser.parse_file("bdf.out")

        # Schema JSON
        json_str = result.model_dump_json(exclude_none=True, indent=2)

        # Markdown
        from bdf_output_parser.reporters.markdown import MarkdownReporter
        md = MarkdownReporter().render(result)
    """

    def parse(self, content: str) -> BDFParseResult:
        """解析 BDF 输出文本，自动检测 task_type。"""
        if not content or not content.strip():
            return BDFParseResult(status=ParseStatus.EMPTY)

        task_type = self._detect_task_type(content)
        errors = self._extract_errors(content)
        warnings = self._extract_warnings(content)

        has_data = False

        energies = self._extract_energies(content, task_type)
        if energies.total_energy is not None:
            has_data = True

        geometry = self._extract_geometry(content)
        if geometry.atoms:
            has_data = True

        frequencies = self._extract_frequencies(content)
        if frequencies.frequencies:
            has_data = True

        thermochemistry = self._extract_thermochemistry(content)

        tddft_blocks = self._extract_tddft(content)
        if tddft_blocks:
            has_data = True

        optimization = self._extract_optimization(content)
        if optimization.n_steps > 0:
            has_data = True

        scf = self._extract_scf(content)

        if errors:
            status = ParseStatus.PARSE_ERROR
        elif has_data:
            status = ParseStatus.SUCCESS
        else:
            status = ParseStatus.PARTIAL

        return BDFParseResult(
            status=status,
            task_type=task_type,
            energies=energies,
            geometry=geometry,
            frequencies=frequencies,
            thermochemistry=thermochemistry,
            tddft_blocks=tddft_blocks,
            optimization=optimization,
            scf=scf,
            warnings=warnings,
            errors=errors,
        )

    def parse_file(self, path: str | Path) -> BDFParseResult:
        """从文件解析 BDF 输出。"""
        p = Path(path)
        if not p.exists():
            return BDFParseResult(status=ParseStatus.EMPTY, source_file=str(path))

        content = p.read_text(encoding="utf-8", errors="ignore")
        result = self.parse(content)
        result.source_file = str(path)
        return result

    # =========================================================================
    # Task type detection
    # =========================================================================

    def _detect_task_type(self, content: str) -> TaskType:
        has_opt = bool(P.TASK_TYPE_OPT.search(content))
        has_freq = bool(P.TASK_TYPE_FREQ.search(content))
        has_tddft = bool(P.TASK_TYPE_TDDFT.search(content))

        if has_opt and has_freq:
            return TaskType.OPT_FREQ
        if has_tddft:
            return TaskType.TDDFT
        if has_opt:
            return TaskType.GEOMETRY_OPT
        if has_freq:
            return TaskType.FREQUENCY
        return TaskType.SINGLE_POINT

    # =========================================================================
    # Energy
    # =========================================================================

    def _extract_energies(self, content: str, task_type: TaskType) -> EnergyData:
        # 优化任务取最后出现的能量（最终几何），否则取第一个
        if task_type == TaskType.GEOMETRY_OPT:
            total = self._find_last_match(content, P.ENERGY_TOTAL)
        else:
            total = self._find_first_match(content, P.ENERGY_TOTAL)

        electronic = self._match_float(content, P.ENERGY_ELECTRONIC)
        nuclear = self._match_float(content, P.ENERGY_NUCLEAR_REPULSION)
        exchange = self._match_float(content, P.ENERGY_EXCHANGE)
        correlation = self._match_float(content, P.ENERGY_CORRELATION)
        mp2 = self._match_float(content, P.ENERGY_MP2)
        kinetic = self._match_float(content, P.ENERGY_KINETIC)
        potential = self._match_float(content, P.ENERGY_POTENTIAL)
        scf = self._match_float(content, P.ENERGY_SCF)

        return EnergyData(
            total_energy=total,
            scf_energy=scf,
            electronic_energy=electronic,
            nuclear_repulsion=nuclear,
            exchange=exchange,
            correlation=correlation,
            mp2_energy=mp2,
            kinetic_energy=kinetic,
            potential_energy=potential,
        )

    # =========================================================================
    # Geometry
    # =========================================================================

    def _extract_geometry(self, content: str) -> GeometryData:
        atoms = self._extract_atoms(content)
        charge = self._match_float(content, P.CHARGE)
        mult_m = P.MULTIPLICITY.search(content)
        multiplicity = int(mult_m.group(1)) if mult_m else None
        pg = self._match_str(content, P.POINT_GROUP)

        return GeometryData(
            atoms=atoms,
            charge=charge,
            multiplicity=multiplicity,
            point_group=pg,
        )

    def _extract_atoms(self, content: str) -> list[Atom]:
        # 策略 0: 取最后一个 Angstrom 坐标块（优化最终结构）
        angstrom_matches = list(P.GEOMETRY_ANGSTROM_HEADER.finditer(content))
        if angstrom_matches:
            # 从最后一个 header 之后提取，直到遇到下一个非坐标 section
            start = angstrom_matches[-1].end()
            tail = content[start:start + 100000]  # 足够容纳大分子坐标
            atoms = self._parse_coord_lines(tail, "angstrom")
            if atoms:
                return atoms

        # 策略 1: Bohr (Cartcoord) — BDF 输出格式: elem, x, y, z[, charge]
        bohr_matches = list(P.GEOMETRY_BOHR.finditer(content))
        if bohr_matches:
            section = bohr_matches[-1].group(0)
            atoms = self._parse_coord_lines(section, "bohr", charge_last=True)
            if atoms:
                return atoms

        # 策略 2: Final/Optimized geometry
        final_m = P.GEOMETRY_FINAL.search(content)
        if final_m:
            atoms = self._parse_coord_lines(final_m.group(0), "bohr")
            if atoms:
                return atoms

        # 策略 3: Geometry ... End geometry
        input_m = P.GEOMETRY_INPUT_BLOCK.search(content)
        if input_m:
            atoms = self._parse_coord_lines(input_m.group(1), "bohr")
            if atoms:
                return atoms

        return []

    def _parse_coord_lines(self, section: str, units: str,
                           charge_last: bool = False) -> list[Atom]:
        """从文本区段中解析原子坐标行。

        charge_last: BDF 输出 Bohr 格式 (elem, x, y, z, charge)
                     未设置时按 BDFEasyInput 格式 (elem, charge, x, y, z)
        """
        atoms = []
        skip_words = {
            "molecular", "cartesian", "coordinates", "angstrom", "bohr",
            "atom", "cartcoord", "charge", "basis", "optimized", "final",
            "converged", "geometry", "end", "no.", "frequencies", "results",
            "reduced", "masses", "force", "constants", "ir", "intensities",
            "irreps", "normal",
        }

        def _is_skip(elem: str) -> bool:
            if not elem or elem[0].isdigit():
                return True
            return elem.lower() in skip_words

        # 带电荷列的格式 (elem + 4 values)
        for m in P.COORD_LINE_WITH_CHARGE.finditer(section):
            elem = m.group(1).strip()
            if _is_skip(elem):
                continue
            try:
                if m.group(5) is not None:
                    if charge_last:
                        # BDF 输出: elem, x, y, z, charge → 取 2,3,4
                        atoms.append(Atom(
                            element=elem,
                            x=float(m.group(2)),
                            y=float(m.group(3)),
                            z=float(m.group(4)),
                            units=units,
                        ))
                    else:
                        # BDFEasyInput 格式: elem, charge, x, y, z → 取 3,4,5
                        atoms.append(Atom(
                            element=elem,
                            x=float(m.group(3)),
                            y=float(m.group(4)),
                            z=float(m.group(5)),
                            units=units,
                        ))
                else:
                    # 标准格式: elem, x, y, z
                    atoms.append(Atom(
                        element=elem,
                        x=float(m.group(2)),
                        y=float(m.group(3)),
                        z=float(m.group(4)),
                        units=units,
                    ))
            except ValueError:
                continue

        if atoms:
            return atoms

        # 回退: BDF 编号格式 "1  O  0.0  0.0  0.0"
        for m in P.COORD_LINE_NUMBERED.finditer(section):
            elem = m.group(1).strip()
            if _is_skip(elem):
                continue
            try:
                atoms.append(Atom(
                    element=elem,
                    x=float(m.group(2)),
                    y=float(m.group(3)),
                    z=float(m.group(4)),
                    units=units,
                ))
            except ValueError:
                continue

        if atoms:
            return atoms

        # 最后回退: "O  0.0  0.0  0.0"
        for m in P.COORD_LINE.finditer(section):
            elem = m.group(1).strip()
            if _is_skip(elem):
                continue
            try:
                atoms.append(Atom(
                    element=elem,
                    x=float(m.group(2)),
                    y=float(m.group(3)),
                    z=float(m.group(4)),
                    units=units,
                ))
            except ValueError:
                continue

        return atoms

    # =========================================================================
    # Frequency
    # =========================================================================

    def _extract_frequencies(self, content: str) -> FrequencyData:
        frequencies: list[float] = []
        ir_intensities: list[float] = []
        reduced_masses: list[float] = []
        force_constants: list[float] = []
        irreps: list[str] = []

        # 优先从 vibrations 区段提取
        vib_m = P.VIB_SECTION.search(content)
        section = vib_m.group(0) if vib_m else content

        frequencies = self._extract_number_lines(section, P.FREQUENCY_LINE)
        ir_intensities = self._extract_number_lines(section, P.IR_INTENSITY_LINE)
        reduced_masses = self._extract_number_lines(section, P.REDUCED_MASS_LINE)
        force_constants = self._extract_number_lines(section, P.FORCE_CONSTANT_LINE)
        irreps = self._extract_string_lines(section, P.IRREP_LINE)

        # 平动/转动频率
        tr_m = P.TRANS_ROT_SECTION.search(content)
        translations_rotations = []
        if tr_m:
            translations_rotations = self._extract_number_lines(tr_m.group(0), P.FREQUENCY_LINE)

        # 热力学
        zpe = self._match_float(content, P.ZERO_POINT_ENERGY)
        thermal = self._match_float(content, P.THERMAL_CORRECTION)
        entropy = self._match_float(content, P.ENTROPY)
        enthalpy = self._match_float(content, P.ENTHALPY)
        gibbs = self._match_float(content, P.GIBBS_FREE_ENERGY)

        return FrequencyData(
            frequencies=frequencies,
            ir_intensities=ir_intensities,
            reduced_masses=reduced_masses,
            force_constants=force_constants,
            irreps=irreps,
            translations_rotations=translations_rotations,
            zero_point_energy=zpe,
            thermal_correction=thermal,
            entropy=entropy,
            enthalpy=enthalpy,
            gibbs_free_energy=gibbs,
        )

    # =========================================================================
    # Thermochemistry
    # =========================================================================

    def _extract_thermochemistry(self, content: str) -> ThermochemistryData:
        ezpe = self._match_float(content, P.THERMO_E_ZPE)
        ethermal = self._match_float(content, P.THERMO_E_THERMAL)
        eenthalpy = self._match_float(content, P.THERMO_E_ENTHALPY)
        egibbs = self._match_float(content, P.THERMO_E_GIBBS)

        if all(v is None for v in (ezpe, ethermal, eenthalpy, egibbs)):
            return ThermochemistryData()

        return ThermochemistryData(
            electronic_plus_zpe=ezpe,
            electronic_plus_thermal=ethermal,
            electronic_plus_enthalpy=eenthalpy,
            electronic_plus_gibbs=egibbs,
        )

    # =========================================================================
    # TDDFT
    # =========================================================================

    def _extract_tddft(self, content: str) -> list[TDDFTBlock]:
        spin_matches = list(P.TDDFT_SPIN_CHANGE.finditer(content))
        if not spin_matches:
            # fallback: 检查是否有 TDDFT header
            if P.TDDFT_HEADER.search(content):
                states = self._parse_excited_states_block(content)
                if states:
                    return [TDDFTBlock(states=states)]
            return []

        blocks: list[TDDFTBlock] = []
        for idx, match in enumerate(spin_matches):
            start = match.start()
            end = spin_matches[idx + 1].start() if idx + 1 < len(spin_matches) else len(content)
            block = content[start:end]

            # 提取 isf（在当前块或之前回溯）
            isf = self._find_int_before(content, start, P.TDDFT_ISF)
            ialda = self._find_int_before(content, start, P.TDDFT_IALDA)
            itda = self._find_int_before(content, start, P.TDDFT_ITDA)

            states = self._parse_excited_states_block(block)
            blocks.append(TDDFTBlock(
                isf=isf,
                ialda=ialda,
                itda=itda,
                states=states,
            ))

        return blocks

    def _parse_excited_states_block(self, block: str) -> list[ExcitedState]:
        """解析单个 TDDFT 块中的激发态表"""
        states: list[ExcitedState] = []
        header_m = P.TDDFT_HEADER.search(block)
        if not header_m:
            return states

        lines = block[header_m.end():].splitlines()
        started = False
        for line in lines:
            if not line.strip() or line.strip().startswith('***'):
                if started:
                    break
                continue
            started = True
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                idx = int(parts[0])
                # BDF header: No. Pair ExSym ExEnergies Wavelengths f dS2 Dominant
                # parts: [idx, sym1, n1, sym2, energy_ev, "eV", wavelength_nm, "nm", f, dS2, ...]
                symmetry = parts[3]  # ExSym label (e.g. "A1")
                energy_ev = float(parts[4])
                wavelength_nm = float(parts[6])
                osc = float(parts[8])
                d_s2 = float(parts[9]) if len(parts) > 9 else None
                dominant = " ".join(parts[10:]).strip() if len(parts) > 10 else None
                states.append(ExcitedState(
                    index=idx,
                    symmetry=symmetry,
                    energy_ev=energy_ev,
                    wavelength_nm=wavelength_nm,
                    oscillator_strength=osc,
                    delta_s2=d_s2,
                    dominant_transition=dominant or None,
                ))
            except (ValueError, IndexError):
                continue

        return states

    # =========================================================================
    # Optimization
    # =========================================================================

    def _extract_optimization(self, content: str) -> OptimizationData:
        if not P.OPT_KEYWORD.search(content):
            return OptimizationData()

        converged = bool(P.OPT_CONVERGED.search(content))
        step_nums = [int(m.group(1)) for m in P.OPT_STEP.finditer(content)]
        n_steps = max(step_nums) if step_nums else 0

        energy_change = self._match_float(content, P.OPT_ENERGY_CHANGE)
        max_force = self._match_float(content, P.OPT_MAX_FORCE)
        rms_force = self._match_float(content, P.OPT_RMS_FORCE)
        max_disp = self._match_float(content, P.OPT_MAX_DISPLACEMENT)
        rms_disp = self._match_float(content, P.OPT_RMS_DISPLACEMENT)

        # 优化任务的 final_energy 取最后一个 E_tot
        final_energy = self._find_last_match(content, P.ENERGY_TOTAL)

        return OptimizationData(
            converged=converged,
            n_steps=n_steps,
            final_energy=final_energy,
            energy_change=energy_change,
            max_force=max_force,
            rms_force=rms_force,
            max_displacement=max_disp,
            rms_displacement=rms_disp,
        )

    # =========================================================================
    # SCF
    # =========================================================================

    def _extract_scf(self, content: str) -> SCFData:
        converged = bool(P.CONVERGENCE_NORMAL.search(content)) or \
                    bool(P.CONVERGENCE_BDF.search(content))

        # 检查 Final DeltaE/DeltaD
        delta_m = P.CONVERGENCE_FINAL_DELTA.search(content)
        if delta_m and not converged:
            try:
                de = abs(float(delta_m.group(1)))
                dd = abs(float(delta_m.group(2)))
                if de < 1e-6 and dd < 1e-4:
                    converged = True
            except ValueError:
                pass

        # SCF 迭代次数：取最后一个 diis closed
        scf_iters = list(P.SCF_ITERATION.finditer(content))
        n_iterations = int(scf_iters[-1].group(1)) if scf_iters else 0

        final_energy = self._match_float(content, P.ENERGY_SCF)
        diis_error = self._match_float(content, P.SCF_DIIS_ERROR)

        return SCFData(
            converged=converged,
            n_iterations=n_iterations,
            final_energy=final_energy,
            diis_error=diis_error,
        )

    # =========================================================================
    # Warnings / Errors
    # =========================================================================

    def _extract_warnings(self, content: str) -> list[str]:
        warnings = []
        for pat in P.WARNING_PATTERNS:
            for m in pat.finditer(content):
                line = m.group(0).strip()
                if line and line not in warnings:
                    warnings.append(line)
        return warnings

    def _extract_errors(self, content: str) -> list[str]:
        errors = []
        for pat in P.ERROR_PATTERNS:
            for m in pat.finditer(content):
                line = m.group(0).strip()
                if line and line not in errors:
                    errors.append(line)
        return errors

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _match_float(content: str, pattern: re.Pattern) -> Optional[float]:
        m = pattern.search(content)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                pass
        return None

    @staticmethod
    def _match_str(content: str, pattern: re.Pattern) -> Optional[str]:
        m = pattern.search(content)
        return m.group(1).strip() if m else None

    @staticmethod
    def _find_first_match(content: str, patterns: list[re.Pattern]) -> Optional[float]:
        for pat in patterns:
            m = pat.search(content)
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    @staticmethod
    def _find_last_match(content: str, patterns: list[re.Pattern]) -> Optional[float]:
        last: Optional[float] = None
        for pat in patterns:
            for m in pat.finditer(content):
                try:
                    last = float(m.group(1))
                except (ValueError, IndexError):
                    continue
        return last

    @staticmethod
    def _extract_number_lines(section: str, pattern: re.Pattern) -> list[float]:
        """从多行匹配中提取所有浮点数"""
        values: list[float] = []
        for m in pattern.finditer(section):
            line = m.group(1).strip()
            if "(cm" in line:
                continue
            for v in line.split():
                try:
                    values.append(float(v))
                except ValueError:
                    continue
        return values

    @staticmethod
    def _extract_string_lines(section: str, pattern: re.Pattern) -> list[str]:
        """从多行匹配中提取所有字符串"""
        values: list[str] = []
        for m in pattern.finditer(section):
            line = m.group(1).strip()
            values.extend(line.split())
        return values

    @staticmethod
    def _find_int_before(content: str, pos: int, pattern: re.Pattern) -> Optional[int]:
        """在 pos 之前回溯查找整数"""
        scope = content[max(0, pos - 100000):pos]
        matches = list(pattern.finditer(scope))
        if matches:
            try:
                return int(matches[-1].group(1))
            except (ValueError, IndexError):
                pass
        return None

    # =========================================================================
    # SAO — Symmetry Adapted Orbital 解析 (checksymm / COMPASS+godetail)
    # =========================================================================

    def parse_sao(self, content: str) -> SAOParseResult:
        """解析 checksymm 输出的 SAO 区块。

        从 COMPASS+godetail 输出中提取：
        - 点群名称
        - 每个不可约表示的轨道数和 SAO 组成
        - 每个 SAO 的 AO 组合与系数
        """
        pg = None
        pg_m = P.SAO_POINT_GROUP.search(content)
        if pg_m:
            pg = pg_m.group(1)

        n_basis = 0
        basis_m = P.SAO_NBASIS.search(content)
        if basis_m:
            n_basis = int(basis_m.group(1))

        # 找到 SAO 区段的起止位置
        sao_start = content.find("Symmetry adapted orbital")
        if sao_start == -1:
            return SAOParseResult(point_group=pg, n_basis=n_basis)

        # 找到 Irrep summary 行作为结尾
        irrep_summary_m = P.SAO_IRREP_NAMES.search(content, sao_start)
        sao_end = irrep_summary_m.start() if irrep_summary_m else len(content)

        section = content[sao_start:sao_end]

        # 解析每个 irrep 的 SAO
        irreps: list[IrrepSAO] = []
        irrep_headers = list(P.SAO_IRREP_HEADER.finditer(section))

        for idx, (hdr, next_hdr) in enumerate(zip(irrep_headers, irrep_headers[1:] + [None])):
            irrep_name = hdr.group(1)
            irrep_index = int(hdr.group(2))
            norb = int(hdr.group(3))

            hdr_start = hdr.end()
            hdr_end = next_hdr.start() if next_hdr else len(section)
            irrep_block = section[hdr_start:hdr_end]

            saos = self._parse_sao_lines(irrep_block, irrep_name, irrep_index)

            irreps.append(IrrepSAO(
                irrep=irrep_name,
                irrep_index=irrep_index,
                norb=norb,
                saos=saos,
            ))

        return SAOParseResult(
            point_group=pg,
            n_irreps=len(irreps),
            n_basis=n_basis,
            irreps=irreps,
        )

    def _parse_sao_lines(self, block: str, irrep: str, irrep_index: int) -> list[SAOLine]:
        """解析一个 irrep 区块内的 SAO 行"""
        saos: list[SAOLine] = []
        lines = block.splitlines()

        i = 0
        while i < len(lines):
            sao_label_m = P.SAO_LABEL_LINE.search(lines[i])
            if not sao_label_m:
                i += 1
                continue

            sao_label = sao_label_m.group(0).strip().split()[0]  # "A1|1C1"
            sao_idx = int(sao_label_m.group(2))
            comp = int(sao_label_m.group(3))

            # 解析 AO 标签（在当前行和后续行都可能出现）
            ao_raw = sao_label_m.group(4).strip()
            ao_labels = self._parse_ao_labels(ao_raw)

            # 下一行是系数
            i += 1
            coeffs = []
            if i < len(lines):
                coeff_m = P.SAO_COEFF_LINE.search(lines[i])
                if coeff_m:
                    coeffs = [float(v) for v in coeff_m.group(1).split()]

            # 将系数绑定到各 AO
            for j, aol in enumerate(ao_labels):
                if j < len(coeffs):
                    aol.coeff = coeffs[j]

            saos.append(SAOLine(
                label=sao_label,
                irrep=irrep,
                irrep_index=irrep_index,
                component=comp,
                aos=ao_labels,
            ))

            i += 1

        return saos

    @staticmethod
    def _parse_ao_labels(ao_text: str) -> list[AOLabel]:
        """解析一行中所有 AO 标签: '1O1S0    1O2S0'"""
        labels: list[AOLabel] = []
        for m in P.SAO_AO_LABEL.finditer(ao_text):
            try:
                labels.append(AOLabel(
                    atom_index=int(m.group(1)),
                    element=m.group(2),
                    n=int(m.group(3)),
                    l=m.group(4),
                    m=int(m.group(5)),
                    coeff=0.0,  # 后续由系数行更新
                ))
            except (ValueError, IndexError):
                continue
        return labels
