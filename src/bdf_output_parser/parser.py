"""
BDF Output Parser — 主解析器

合并 BDFAssistant、BDFExecuteAgent、BDFEasyInput 三套解析器的最优逻辑。
自动检测 task_type，无需调用方传入。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .models import (
    AOLabel,
    Atom,
    BDFParseResult,
    BDFUnifiedResult,
    ConsistencyWarning,
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
    UnifiedParseStatus,
    UnifiedResultStatus,
    UnifiedRunStatus,
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

    def read(
        self,
        *,
        out_path: str | Path | None = None,
        out_tmp_path: str | Path | None = None,
        hdf5_path: str | Path | None = None,
        artifact_paths: dict[str, str] | None = None,
        artifact_manifest: dict[str, Any] | None = None,
    ) -> BDFUnifiedResult:
        """Read BDF output artifacts into one parser-owned result model.

        This first slice deliberately wraps existing readers:
        ``parse(text)`` for ``.out`` and ``BDFCoreStateInspector`` for
        ``.bdfh5``.  Missing optional files are recorded in ``raw_refs`` rather
        than raised as exceptions.
        """

        output_text = self._read_optional_text(out_path)
        out_tmp_text = self._read_optional_text(out_tmp_path)
        parse_result = self.parse(output_text) if output_text is not None else None
        core_summary = self._read_core_state(hdf5_path)

        parse_status = self._unified_parse_status(parse_result, out_path)
        output_run_status = self._run_status_from_output(parse_result, output_text)
        hdf5_run_status = self._run_status_from_core_state(core_summary)

        diagnostics = self._build_unified_diagnostics(
            parse_result,
            core_summary,
            output_text,
            out_tmp_text,
        )
        run_status = (
            hdf5_run_status
            or self._run_status_from_diagnostics(diagnostics)
            or output_run_status
            or UnifiedRunStatus.UNKNOWN
        )
        restart = self._build_unified_restart(
            core_summary,
            artifact_paths,
            artifact_manifest,
        )
        consistency_warnings = self._build_consistency_warnings(
            hdf5_run_status=hdf5_run_status,
            output_run_status=output_run_status,
            parse_result=parse_result,
        )
        result_status = self._unified_result_status(
            run_status,
            parse_status,
            diagnostics,
            restart,
        )

        field_sources = self._build_field_sources(
            parse_result,
            core_summary,
            diagnostics,
            restart,
        )
        raw_refs = self._build_raw_refs(out_path, out_tmp_path, hdf5_path)
        results = self._build_results(parse_result)

        return BDFUnifiedResult(
            task_type=str(parse_result.task_type.value) if parse_result else None,
            run_status=run_status,
            parse_status=parse_status,
            result_status=result_status,
            success=result_status
            in {UnifiedResultStatus.USABLE, UnifiedResultStatus.USABLE_WITH_WARNINGS},
            execution=self._build_execution(core_summary),
            results=results,
            diagnostics=diagnostics,
            restart=restart,
            artifacts={
                "paths": artifact_paths or {},
                "manifest": artifact_manifest or {},
            },
            quality=self._build_quality(
                parse_result,
                diagnostics,
                consistency_warnings,
            ),
            field_sources=field_sources,
            consistency_warnings=consistency_warnings,
            raw_refs=raw_refs,
        )

    @staticmethod
    def _read_optional_text(path: str | Path | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        return p.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_core_state(path: str | Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        from .core_state import BDFCoreStateInspector

        summary = BDFCoreStateInspector().read(str(p))
        return summary.model_dump(mode="json")

    @staticmethod
    def _unified_parse_status(
        parse_result: BDFParseResult | None,
        out_path: str | Path | None,
    ) -> UnifiedParseStatus:
        if parse_result is None:
            return UnifiedParseStatus.UNAVAILABLE
        if parse_result.status == ParseStatus.SUCCESS:
            return UnifiedParseStatus.COMPLETE
        if parse_result.status == ParseStatus.PARTIAL:
            return UnifiedParseStatus.PARTIAL
        if parse_result.status == ParseStatus.PARSE_ERROR:
            return UnifiedParseStatus.FAILED
        return UnifiedParseStatus.EMPTY

    @staticmethod
    def _run_status_from_core_state(
        core_summary: dict[str, Any] | None,
    ) -> UnifiedRunStatus | None:
        if not core_summary or not core_summary.get("available"):
            return None
        raw = str(core_summary.get("status") or "").lower()
        if raw in {item.value for item in UnifiedRunStatus}:
            return UnifiedRunStatus(raw)
        return None

    @staticmethod
    def _run_status_from_output(
        parse_result: BDFParseResult | None,
        output_text: str | None,
    ) -> UnifiedRunStatus | None:
        if output_text and P.CONVERGENCE_BDF.search(output_text):
            return UnifiedRunStatus.COMPLETED
        if parse_result is None:
            return None
        if parse_result.status == ParseStatus.SUCCESS:
            return UnifiedRunStatus.COMPLETED
        if parse_result.status == ParseStatus.PARSE_ERROR:
            return UnifiedRunStatus.FAILED
        return None

    @staticmethod
    def _run_status_from_diagnostics(
        diagnostics: dict[str, Any],
    ) -> UnifiedRunStatus | None:
        if diagnostics.get("primary_failure"):
            return UnifiedRunStatus.FAILED
        return None

    @staticmethod
    def _unified_result_status(
        run_status: UnifiedRunStatus,
        parse_status: UnifiedParseStatus,
        diagnostics: dict[str, Any],
        restart: dict[str, Any],
    ) -> UnifiedResultStatus:
        has_warnings = bool(diagnostics.get("warnings"))
        has_restart = bool(
            restart.get("assets")
            or restart.get("scratch")
            or restart.get("modules")
        )
        if run_status == UnifiedRunStatus.COMPLETED:
            if parse_status in {
                UnifiedParseStatus.COMPLETE,
                UnifiedParseStatus.PARTIAL,
            }:
                if has_warnings:
                    return UnifiedResultStatus.USABLE_WITH_WARNINGS
                return UnifiedResultStatus.USABLE
            return UnifiedResultStatus.UNKNOWN
        if run_status in {UnifiedRunStatus.FAILED, UnifiedRunStatus.INTERRUPTED}:
            if has_restart:
                return UnifiedResultStatus.INCOMPLETE_RESTARTABLE
            return UnifiedResultStatus.NOT_USABLE
        return UnifiedResultStatus.UNKNOWN

    @staticmethod
    def _build_results(parse_result: BDFParseResult | None) -> dict[str, Any]:
        if parse_result is None:
            return {}
        return {
            "energies": parse_result.energies.model_dump(mode="json"),
            "scf": parse_result.scf.model_dump(mode="json"),
            "geometry": parse_result.geometry.model_dump(mode="json"),
            "optimization": parse_result.optimization.model_dump(mode="json"),
            "frequency": parse_result.frequencies.model_dump(mode="json"),
            "thermochemistry": parse_result.thermochemistry.model_dump(mode="json"),
            "tddft": {
                "blocks": [
                    block.model_dump(mode="json")
                    for block in parse_result.tddft_blocks
                ],
                "excited_states": [
                    state.model_dump(mode="json")
                    for state in parse_result.excited_states
                ],
            },
        }

    @staticmethod
    def _build_execution(core_summary: dict[str, Any] | None) -> dict[str, Any]:
        if not core_summary:
            return {}
        return {
            "core_state_summary": core_summary,
            "run": {
                "status": core_summary.get("status"),
                "current_module": core_summary.get("current_module"),
                "last_successful_module": core_summary.get("last_successful_module"),
                "failed_module": core_summary.get("failed_module"),
                "interrupted_module": core_summary.get("interrupted_module"),
                "restartable": core_summary.get("restartable"),
                "elapsed_sec": core_summary.get("elapsed_sec"),
            },
            "workflow": core_summary.get("workflow") or {},
            "provenance": core_summary.get("provenance") or {},
        }

    def _build_unified_diagnostics(
        self,
        parse_result: BDFParseResult | None,
        core_summary: dict[str, Any] | None,
        output_text: str | None,
        out_tmp_text: str | None,
    ) -> dict[str, Any]:
        diagnostics = {
            "status": None,
            "primary_failure": None,
            "warnings": [],
            "secondary_diagnostics": [],
            "evidence": [],
            "recoverable": None,
            "suggested_actions": [],
        }

        if core_summary:
            diagnostics["status"] = core_summary.get("status")
            self._add_core_state_diagnostics(diagnostics, core_summary)

        if parse_result:
            for error in parse_result.errors:
                diagnostics["secondary_diagnostics"].append(
                    {"source": "output", "message": error, "severity": "error"}
                )
            for warning in parse_result.warnings:
                diagnostics["warnings"].append(
                    {"source": "output", "message": warning, "severity": "warning"}
                )

        for source, text in (
            ("output", output_text),
            ("output_tmp", out_tmp_text),
        ):
            if not text:
                continue
            blocks = self._extract_structured_blocks(text, "BDF_ERROR")
            for body in blocks:
                record = self._structured_block_to_record(
                    body,
                    source=source,
                    default_severity="fatal",
                )
                if diagnostics["primary_failure"] is None:
                    diagnostics["primary_failure"] = record
                else:
                    diagnostics["secondary_diagnostics"].append(record)
            for body in self._extract_structured_blocks(text, "BDF_WARNING"):
                diagnostics["warnings"].append(
                    self._structured_block_to_record(
                        body,
                        source=source,
                        default_severity="warning",
                    )
                )

        primary = diagnostics.get("primary_failure")
        if isinstance(primary, dict):
            diagnostics["recoverable"] = primary.get("recoverable")
            suggestion = primary.get("suggestion")
            if suggestion:
                diagnostics["suggested_actions"].append(suggestion)
        return diagnostics

    def _add_core_state_diagnostics(
        self,
        diagnostics: dict[str, Any],
        core_summary: dict[str, Any],
    ) -> None:
        records = core_summary.get("diagnostics") or {}
        if not isinstance(records, dict):
            return
        for module, module_data in records.items():
            if not isinstance(module_data, dict):
                continue
            failure = self._latest_diagnostic_record(module_data, "failure")
            if failure:
                normalized = dict(failure)
                normalized.setdefault("module", module)
                normalized.setdefault("source", "hdf5")
                diagnostics["primary_failure"] = normalized
                diagnostics["evidence"].append(f"core_state:/diagnostics/{module}/last_failure")
            warning = self._latest_diagnostic_record(module_data, "warning")
            if warning:
                normalized_warning = dict(warning)
                normalized_warning.setdefault("module", module)
                normalized_warning.setdefault("source", "hdf5")
                diagnostics["warnings"].append(normalized_warning)
                diagnostics["evidence"].append(f"core_state:/diagnostics/{module}/last_warning")

    @staticmethod
    def _latest_diagnostic_record(
        module_data: dict[str, Any],
        kind: str,
    ) -> dict[str, Any] | None:
        last = module_data.get(f"last_{kind}")
        if isinstance(last, dict) and last:
            return last
        records = module_data.get(f"{kind}s")
        if not isinstance(records, dict) or not records:
            return None
        latest_key = sorted(str(key) for key in records)[-1]
        latest = records.get(latest_key)
        return latest if isinstance(latest, dict) else None

    @staticmethod
    def _extract_structured_blocks(text: str, tag: str) -> list[str]:
        pattern = re.compile(rf"\[{tag}\](.*?)\[/{tag}\]", re.IGNORECASE | re.DOTALL)
        return [match.group(1) for match in pattern.finditer(text)]

    @staticmethod
    def _structured_block_to_record(
        body: str,
        *,
        source: str,
        default_severity: str,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "source": source,
            "message": body.strip(),
            "severity": default_severity,
        }
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            normalized_key = key.strip().lower()
            if not normalized_key:
                continue
            record[normalized_key] = value.strip()
        if record.get("primary") is not None:
            record["primary"] = str(record["primary"]).lower() in {"yes", "true", "1"}
        record["severity"] = record.get("severity") or default_severity
        return record

    @staticmethod
    def _build_unified_restart(
        core_summary: dict[str, Any] | None,
        artifact_paths: dict[str, str] | None,
        artifact_manifest: dict[str, Any] | None,
    ) -> dict[str, Any]:
        restart = {}
        if core_summary and isinstance(core_summary.get("restart"), dict):
            restart.update(core_summary["restart"])
        if artifact_paths:
            restart.setdefault("artifact_paths", artifact_paths)
        if artifact_manifest:
            restart.setdefault("artifact_manifest", artifact_manifest)
        return restart

    @staticmethod
    def _build_field_sources(
        parse_result: BDFParseResult | None,
        core_summary: dict[str, Any] | None,
        diagnostics: dict[str, Any],
        restart: dict[str, Any],
    ) -> dict[str, str]:
        sources = {}
        if core_summary and core_summary.get("available"):
            sources["run_status"] = "hdf5"
            if core_summary.get("diagnostics"):
                sources["diagnostics.primary_failure"] = "hdf5"
        elif parse_result:
            sources["run_status"] = "output"
        primary = diagnostics.get("primary_failure") if isinstance(diagnostics, dict) else None
        if isinstance(primary, dict) and primary.get("source"):
            sources.setdefault("diagnostics.primary_failure", str(primary["source"]))
        if parse_result:
            if parse_result.energies.total_energy is not None:
                sources["results.energies.total_energy"] = "output"
            if parse_result.energies.scf_energy is not None:
                sources["results.energies.scf_energy"] = "output"
            if parse_result.scf:
                sources["results.scf.converged"] = "output"
            if parse_result.geometry.atoms:
                sources["results.geometry.final"] = "output"
            if parse_result.optimization.n_steps > 0:
                sources["results.optimization.converged"] = "output"
            if parse_result.frequencies.frequencies:
                sources["results.frequency.frequencies"] = "output"
            if parse_result.tddft_blocks:
                sources["results.tddft.states"] = "output"
        if restart:
            if core_summary and core_summary.get("restart"):
                sources["restart.assets"] = "hdf5"
            else:
                sources["restart.assets"] = "assistant_manifest"
        return sources

    @staticmethod
    def _build_raw_refs(
        out_path: str | Path | None,
        out_tmp_path: str | Path | None,
        hdf5_path: str | Path | None,
    ) -> dict[str, Any]:
        def ref(path: str | Path | None) -> dict[str, Any]:
            if path is None:
                return {"path": None, "exists": False}
            p = Path(path)
            return {
                "path": str(p),
                "exists": p.exists(),
                "size": p.stat().st_size if p.exists() else None,
            }

        return {
            "out": ref(out_path),
            "out_tmp": ref(out_tmp_path),
            "hdf5": ref(hdf5_path),
        }

    @staticmethod
    def _build_consistency_warnings(
        *,
        hdf5_run_status: UnifiedRunStatus | None,
        output_run_status: UnifiedRunStatus | None,
        parse_result: BDFParseResult | None,
    ) -> list[ConsistencyWarning]:
        warnings = []
        if (
            hdf5_run_status
            and output_run_status
            and hdf5_run_status != output_run_status
        ):
            warnings.append(
                ConsistencyWarning(
                    field="run_status",
                    hdf5_value=hdf5_run_status.value,
                    output_value=output_run_status.value,
                    severity="warning",
                    message="HDF5 run status and output-derived status disagree.",
                )
            )
        if parse_result and parse_result.scf.final_energy is not None:
            total = parse_result.energies.total_energy
            scf = parse_result.scf.final_energy
            if total is not None and abs(total - scf) > 1e-8:
                warnings.append(
                    ConsistencyWarning(
                        field="results.energies.scf_energy",
                        hdf5_value=None,
                        output_value=scf,
                        tolerance=1e-8,
                        severity="info",
                        message="Output total energy and SCF final energy differ.",
                    )
                )
        return warnings

    @staticmethod
    def _build_quality(
        parse_result: BDFParseResult | None,
        diagnostics: dict[str, Any],
        consistency_warnings: list[ConsistencyWarning],
    ) -> dict[str, Any]:
        quality = {
            "warnings_count": len(diagnostics.get("warnings") or []),
            "consistency_warnings_count": len(consistency_warnings),
        }
        if parse_result:
            quality["n_imaginary_frequencies"] = parse_result.frequencies.n_imaginary
            quality["frequency_stable"] = parse_result.frequencies.is_stable
        return quality

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
        scf = self._extract_scf_energy(content, task_type)

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
        # 详细格式优先: "No. 1 w= ... a.u. f= ... Ova= ..."
        detail_states = self._parse_tddft_detail_format(content)
        if detail_states:
            self._sort_and_label(detail_states)
            return [TDDFTBlock(states=detail_states)]

        # 备选格式: "No.  1   w=  9.8445 eV ... f=  0.0906"
        alt_states = self._parse_tddft_alt_format(content)
        if alt_states:
            self._sort_and_label(alt_states)
            return [TDDFTBlock(states=alt_states)]

        spin_matches = list(P.TDDFT_SPIN_CHANGE.finditer(content))
        if not spin_matches:
            # fallback: 检查是否有 TDDFT header
            if P.TDDFT_HEADER.search(content):
                states = self._parse_excited_states_block(content)
                if states:
                    self._sort_and_label(states)
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

        # 跨 block 全局排序
        self._sort_tddft_blocks(blocks)
        return blocks

    @staticmethod
    def _sort_and_label(states: list[ExcitedState]) -> None:
        """按能量升序排列激发态，分配 S1/S2/T1/T2 标签。

        如果状态没有 isf 信息（单块单重态），默认全部标 S。
        """
        states.sort(key=lambda s: s.energy_ev)
        for i, s in enumerate(states):
            # 无 isf 的默认标 S（单重态）
            s.index = i + 1
            s.label = f"S{i + 1}"

    @staticmethod
    def _sort_tddft_blocks(blocks: list[TDDFTBlock]) -> None:
        """跨 TDDFT block 全局排序：按 isf 分组，每组内按能量排序并标 S1/S2/T1/T2。"""
        # 收集所有 (isf, state) 对
        pairs: list[tuple[Optional[int], ExcitedState]] = []
        for block in blocks:
            for s in block.states:
                pairs.append((block.isf, s))

        # 按 isf 分组，每组内按能量升序
        singlets = [(i, s) for i, s in pairs if i in (0, None)]
        triplets = [(i, s) for i, s in pairs if i == 1]

        singlets.sort(key=lambda x: x[1].energy_ev)
        triplets.sort(key=lambda x: x[1].energy_ev)

        # 分配全局标签
        for idx, (_, s) in enumerate(singlets):
            s.label = f"S{idx + 1}"
        for idx, (_, s) in enumerate(triplets):
            s.label = f"T{idx + 1}"

        # 更新 block 内排序
        for block in blocks:
            block.states.sort(key=lambda s: s.energy_ev)

    def _parse_excited_states_block(self, block: str) -> list[ExcitedState]:
        """解析单个 TDDFT 块中的激发态表"""
        states: list[ExcitedState] = []
        header_m = P.TDDFT_HEADER.search(block)
        if not header_m:
            # 尝试备选格式: "No.  1   w=  9.8445 eV  ...  f=  0.0906"
            states = self._parse_tddft_alt_format(block)
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

    @staticmethod
    def _parse_tddft_detail_format(content: str) -> list[ExcitedState]:
        """详细 TDDFT 格式: 'No. 1 w= ... a.u. f= ... Ova= ...' + CV lines."""
        from .models import CVTransition
        lines = content.splitlines()
        states: list[ExcitedState] = []
        current_state: Optional[ExcitedState] = None

        for line in lines:
            # Match detailed per-state header
            dm = P.TDDFT_DETAIL_LINE.search(line)
            if dm:
                try:
                    idx = int(dm.group(1))
                    energy_ev = float(dm.group(2))
                    total_au = float(dm.group(3))
                    osc = float(dm.group(4))
                    wl = 1239.84193 / energy_ev if energy_ev > 0 else 0.0
                    state = ExcitedState(
                        index=idx,
                        energy_ev=energy_ev,
                        wavelength_nm=round(wl, 2),
                        oscillator_strength=osc,
                        total_energy_au=total_au,
                        ova=float(dm.group(6)) if dm.group(6) else None,
                        delta_s2=float(dm.group(5)) if dm.group(5) else None,
                    )
                    states.append(state)
                    current_state = state
                except (ValueError, IndexError):
                    continue
                continue

            # Match CV transition line (associated with current state)
            cv_m = P.TDDFT_CV_LINE.search(line)
            if cv_m and current_state:
                try:
                    cv = CVTransition(
                        from_orbital=f"{cv_m.group(1)}({int(cv_m.group(2))})",
                        to_orbital=f"{cv_m.group(3)}({int(cv_m.group(4))})",
                        coefficient=float(cv_m.group(5)),
                        percentage=float(cv_m.group(6)),
                        ipa_ev=float(cv_m.group(7)),
                        oai=float(cv_m.group(8)) if cv_m.group(8) else None,
                    )
                    current_state.cv_transitions.append(cv)
                    # Set dominant transition from first (largest) CV
                    if not current_state.dominant_transition and cv.percentage > 10:
                        current_state.dominant_transition = (
                            f"{cv.from_orbital}->{cv.to_orbital} ({cv.percentage:.0f}%)"
                        )
                except (ValueError, IndexError):
                    continue

        return states

    @staticmethod
    def _parse_tddft_alt_format(block: str) -> list[ExcitedState]:
        """备选 TDDFT 格式: 'No.  1   w=  9.8445 eV  ...  f=  0.0906'"""
        states: list[ExcitedState] = []
        for m in P.TDDFT_ALT_LINE.finditer(block):
            try:
                idx = int(m.group(1))
                energy_ev = float(m.group(2))
                osc = float(m.group(3))
                wavelength_nm = 1239.84193 / energy_ev if energy_ev > 0 else 0.0
                states.append(ExcitedState(
                    index=idx,
                    energy_ev=energy_ev,
                    wavelength_nm=round(wavelength_nm, 2),
                    oscillator_strength=osc,
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

    def _extract_scf_energy(self, content: str, task_type: TaskType) -> Optional[float]:
        """Extract final SCF energy from BDF output.

        BDF simple HF/DFT energy jobs commonly print the converged SCF energy
        as ``E_tot = ...`` instead of a literal ``SCF energy = ...`` line.
        Keep explicit SCF-energy labels first, then fall back to ``E_tot``.
        """
        explicit = self._match_float(content, P.ENERGY_SCF)
        if explicit is not None:
            return explicit
        if task_type == TaskType.GEOMETRY_OPT:
            return self._find_last_match(content, [P.ENERGY_SCF_ETOT])
        return self._match_float(content, P.ENERGY_SCF_ETOT)

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

        task_type = self._detect_task_type(content)
        final_energy = self._extract_scf_energy(content, task_type)
        diis_error = self._match_float(content, P.SCF_DIIS_ERROR)

        homo = self._match_float(content, P.HOMO_ENERGY)
        lumo = self._match_float(content, P.LUMO_ENERGY)
        gap = self._match_float(content, P.HOMO_LUMO_GAP)

        return SCFData(
            converged=converged,
            n_iterations=n_iterations,
            final_energy=final_energy,
            diis_error=diis_error,
            homo_energy=homo,
            lumo_energy=lumo,
            homo_lumo_gap=gap,
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
