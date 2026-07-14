"""Unified .out + .bdfh5 result interface tests."""

from __future__ import annotations

import pytest

from bdf_output_parser import BDFOutputParser
from bdf_output_parser.models import (
    UnifiedParseStatus,
    UnifiedResultStatus,
    UnifiedRunStatus,
)


def _write_core_state(
    path,
    *,
    status: str = "completed",
    failed_module: str = "",
    with_diagnostics: bool = False,
    with_restart: bool = False,
    with_scf_results: bool = False,
    scf_energy: float = -76.12345678,
    scf_converged: bool = True,
    scf_iterations: int = 12,
    with_context_scf: bool = False,
    with_opt_results: bool = False,
    opt_converged: bool = True,
    with_legacy_optgeom: bool = False,
    with_tddft_results: bool = False,
    tddft_energy_shift: float = 0.0,
    tddft_warning: bool = False,
) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("meta/core_state_schema", data="BDFCoreState-v1.0")
        h5.create_dataset("run/status", data=status)
        h5.create_dataset("run/last_successful_module", data="scf")
        if failed_module:
            h5.create_dataset("run/failed_module", data=failed_module)
        if with_diagnostics:
            failure = h5.create_group("diagnostics/scf/failures/000001")
            failure.attrs["schema_version"] = "BDFDiagnostic-v1.0"
            failure.attrs["diagnostic_id"] = "000001"
            failure.attrs["kind"] = "failure"
            failure.attrs["module"] = "scf"
            failure.attrs["phase"] = "scf_convergence"
            failure.attrs["code"] = "SCF_NOT_CONVERGED"
            failure.attrs["category"] = "SCF_CONVERGENCE"
            failure.attrs["severity"] = "fatal"
            failure.attrs["recoverable"] = "yes"
            failure.attrs["message"] = "SCF did not converge"
            failure.attrs["suggestion"] = "restart with readmo"
            failure.attrs["primary"] = "yes"
            last_failure = h5.create_group("diagnostics/scf/last_failure")
            for key, value in failure.attrs.items():
                last_failure.attrs[key] = value
            warning = h5.create_group("diagnostics/tddft/warnings/000001")
            warning.attrs["kind"] = "warning"
            warning.attrs["module"] = "tddft"
            warning.attrs["code"] = "TDDFT_UNSTABLE_REFERENCE_EXCITATION"
            warning.attrs["category"] = "TDDFT_RESULT_QUALITY"
            warning.attrs["severity"] = "warning"
            warning.attrs["imaginary_complex_count"] = 1
            warning.attrs["negative_excitation_count"] = 0
            warning.attrs["itda"] = 0
            last_warning = h5.create_group("diagnostics/tddft/last_warning")
            for key, value in warning.attrs.items():
                last_warning.attrs[key] = value
        if tddft_warning and not with_diagnostics:
            warning = h5.create_group("diagnostics/tddft/warnings/000001")
            warning.attrs["kind"] = "warning"
            warning.attrs["module"] = "tddft"
            warning.attrs["code"] = "TDDFT_UNSTABLE_REFERENCE_EXCITATION"
            warning.attrs["category"] = "TDDFT_RESULT_QUALITY"
            warning.attrs["severity"] = "warning"
            warning.attrs["imaginary_complex_count"] = 1
            warning.attrs["negative_excitation_count"] = 0
            warning.attrs["itda"] = 0
            last_warning = h5.create_group("diagnostics/tddft/last_warning")
            for key, value in warning.attrs.items():
                last_warning.attrs[key] = value
        if with_restart:
            scratch = h5.create_group("restart/scratch")
            scratch.attrs["preserved"] = True
            scratch.attrs["tmpdir"] = "/tmp/bdf-scratch"
            asset = h5.create_group("restart/assets/scforb")
            asset.attrs["kind"] = "scforb"
            asset.attrs["exists"] = True
        if with_scf_results:
            energy = h5.require_group("results/energy")
            energy.create_dataset("total_energy_hartree", data=scf_energy)
            scf = h5.require_group("results/scf")
            scf.create_dataset("scf_energy_hartree", data=scf_energy)
            scf.create_dataset("converged", data=1 if scf_converged else 0)
            scf.create_dataset("n_iterations", data=scf_iterations)
            scf.create_dataset("source_context_id", data=1)
            scf.create_dataset("source_module_ordinal", data=3)
            scf.attrs["producer_module"] = "scf"
            scf.attrs["result_role"] = "latest_scf"
            scf.attrs["source"] = "context:1"
            scf.attrs["schema_version"] = "BDFScfResult-v1.0"
        if with_context_scf:
            context = h5.require_group("contexts/000001/results")
            energy = context.require_group("energy")
            energy.create_dataset("total_energy_hartree", data=scf_energy)
            scf = context.require_group("scf")
            scf.create_dataset("scf_energy_hartree", data=scf_energy)
            scf.create_dataset("converged", data=1 if scf_converged else 0)
            scf.create_dataset("n_iterations", data=scf_iterations)
            scf.attrs["producer_module"] = "scf"
            scf.attrs["result_role"] = "context_scf"
            scf.attrs["context_id"] = 1
            scf.attrs["module_ordinal"] = 3
            scf.attrs["schema_version"] = "BDFScfResult-v1.0"
        if with_opt_results:
            current = h5.require_group("results/geometry/current")
            current.create_dataset("natom", data=3)
            current.create_dataset("labels", data=_encoded_labels(["O", "H", "H"]))
            current.create_dataset(
                "coordinates_bohr",
                data=[0.0, 0.0, 0.1, 0.0, 1.4, -0.2, 0.0, -1.4, -0.2],
            )
            current.create_dataset("step_index", data=2)
            current.attrs["source"] = "bdfopt_bdf"
            current.attrs["optimizer"] = "bdf_native"
            current.attrs["schema_version"] = "BDFOptResult-v1.0"
            if opt_converged:
                final = h5.require_group("results/geometry/final")
                final.create_dataset("natom", data=3)
                final.create_dataset("labels", data=_encoded_labels(["O", "H", "H"]))
                final.create_dataset(
                    "coordinates_bohr",
                    data=[0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, -1.5, 0.0],
                )
                final.create_dataset("step_index", data=3)
                final.attrs["source"] = "bdfopt_bdf"
                final.attrs["optimizer"] = "bdf_native"
                final.attrs["schema_version"] = "BDFOptResult-v1.0"
            opt = h5.require_group("results/optimization")
            opt.create_dataset("converged", data=1 if opt_converged else 0)
            opt.create_dataset("n_steps", data=3)
            opt.create_dataset("max_steps", data=50)
            opt.create_dataset("info_code", data=0 if opt_converged else 1)
            opt.attrs["info_label"] = "geo_err_noerr" if opt_converged else "geo_err_failedtoconverge"
            opt.attrs["optimizer"] = "bdf_native"
            opt.attrs["source"] = "bdfopt_bdf"
            opt.attrs["schema_version"] = "BDFOptResult-v1.0"
        if with_legacy_optgeom:
            legacy = h5.require_group("legacy/optgeom/current")
            legacy.create_dataset("natom", data=3)
            legacy.create_dataset("labels", data=_encoded_labels(["O", "H", "H"]))
            legacy.create_dataset(
                "coordinates",
                data=[0.0, 0.0, 0.2, 0.0, 1.3, -0.1, 0.0, -1.3, -0.1],
            )
            legacy.create_dataset("igeom", data=4)
            legacy.attrs["unit"] = "bohr"
            legacy.attrs["producer"] = "rw_molegeom_in_punch"
        if with_tddft_results:
            tddft = h5.require_group("results/tddft")
            tddft.create_dataset("itda", data=0)
            tddft.create_dataset("tda", data=0)
            tddft.create_dataset("isf", data=0)
            tddft.create_dataset("n_roots_requested", data=3)
            tddft.create_dataset("n_roots_found", data=3)
            tddft.create_dataset("imaginary_complex_count", data=1 if tddft_warning else 0)
            tddft.create_dataset("negative_excitation_count", data=0)
            tddft.attrs["schema_version"] = "BDFTddftResult-v1.0"
            tddft.attrs["producer_module"] = "tddft"
            tddft.attrs["source"] = "tddft_prt_exc"
            states = h5.require_group("results/tddft/states")
            states.create_dataset("root_index", data=[1, 2, 3])
            states.create_dataset(
                "energy_ev",
                data=[7.4012 + tddft_energy_shift, 9.8445 + tddft_energy_shift, 10.1123 + tddft_energy_shift],
            )
            states.create_dataset("wavelength_nm", data=[167.52, 125.94, 122.61])
            states.create_dataset("oscillator_strength", data=[0.0000, 0.0906, 0.0111])
            states.create_dataset("quality_flag", data=[1 if tddft_warning else 0, 0, 0])
            states.create_dataset("delta_s2", data=[0.0, 0.0, 0.0])
            states.create_dataset("dominant_percent", data=[0.0 if tddft_warning else 62.0, 58.0, 40.0])
            states.create_dataset("ipa_ev", data=[8.0, 10.0, 11.0])
            states.create_dataset("ova", data=[0.70, 0.80, 0.65])
            states.attrs["unit_energy"] = "eV"
            states.attrs["unit_wavelength"] = "nm"


def _encoded_labels(symbols: list[str]) -> list[int]:
    return [int.from_bytes(symbol.ljust(8).encode("ascii"), "little") for symbol in symbols]


def test_read_output_only_energy(tmp_path):
    out = tmp_path / "h2o.out"
    out.write_text(
        """
 E_tot = -76.12345678
 diis/vshift is closed at iter = 12
 Final scf result
 Congratulations! BDF normal termination
""",
        encoding="utf-8",
    )

    result = BDFOutputParser().read(out_path=out)

    assert result.run_status == UnifiedRunStatus.COMPLETED
    assert result.parse_status == UnifiedParseStatus.COMPLETE
    assert result.result_status == UnifiedResultStatus.USABLE
    assert result.success is True
    assert result.results["energies"]["scf_energy"] == pytest.approx(-76.12345678)
    assert result.field_sources["results.energies.scf_energy"] == "output"
    assert result.raw_refs["out"]["exists"] is True


def test_read_hdf5_only_failure_with_diagnostics_and_restart(tmp_path):
    hdf5 = tmp_path / "fail.bdfh5"
    _write_core_state(
        hdf5,
        status="failed",
        failed_module="scf",
        with_diagnostics=True,
        with_restart=True,
    )

    result = BDFOutputParser().read(hdf5_path=hdf5)

    assert result.run_status == UnifiedRunStatus.FAILED
    assert result.parse_status == UnifiedParseStatus.UNAVAILABLE
    assert result.result_status == UnifiedResultStatus.INCOMPLETE_RESTARTABLE
    assert result.success is False
    assert result.diagnostics["primary_failure"]["code"] == "SCF_NOT_CONVERGED"
    assert (
        result.diagnostics["warnings"][0]["code"]
        == "TDDFT_UNSTABLE_REFERENCE_EXCITATION"
    )
    assert result.restart["scratch"]["preserved"] is True
    assert result.field_sources["run_status"] == "hdf5"
    assert result.field_sources["diagnostics.primary_failure"] == "hdf5"


def test_read_hdf5_only_scf_scalar_results(tmp_path):
    hdf5 = tmp_path / "energy.bdfh5"
    _write_core_state(hdf5, with_scf_results=True, with_context_scf=True)

    result = BDFOutputParser().read(hdf5_path=hdf5)

    assert result.results["energies"]["total_energy"] == pytest.approx(-76.12345678)
    assert result.results["energies"]["scf_energy"] == pytest.approx(-76.12345678)
    assert result.results["scf"]["final_energy"] == pytest.approx(-76.12345678)
    assert result.results["scf"]["converged"] is True
    assert result.results["scf"]["n_iterations"] == 12
    assert result.results["scf"]["latest"]["scf"]["source_context_id"] == 1
    assert result.results["scf"]["contexts"]["000001"]["scf"]["n_iterations"] == 12
    assert result.field_sources["results.energies.scf_energy"] == "hdf5"
    assert result.field_sources["results.scf.contexts"] == "hdf5"


def test_read_hdf5_native_opt_geometry_and_summary(tmp_path):
    hdf5 = tmp_path / "opt.bdfh5"
    _write_core_state(hdf5, with_opt_results=True)

    result = BDFOutputParser().read(hdf5_path=hdf5)

    geometry = result.results["geometry"]
    assert geometry["current"]["step_index"] == 2
    assert geometry["current"]["atoms"][0]["element"] == "O"
    assert geometry["final"]["step_index"] == 3
    assert geometry["atoms"] == geometry["final"]["atoms"]
    assert result.results["optimization"]["converged"] is True
    assert result.results["optimization"]["n_steps"] == 3
    assert result.results["optimization"]["max_steps"] == 50
    assert result.results["optimization"]["info_label"] == "geo_err_noerr"
    assert result.field_sources["results.geometry.current"] == "bdfopt_bdf"
    assert result.field_sources["results.geometry.final"] == "hdf5"
    assert result.field_sources["results.optimization.n_steps"] == "hdf5"


def test_read_hdf5_legacy_optgeom_current_as_parser_fallback(tmp_path):
    hdf5 = tmp_path / "legacy-opt.bdfh5"
    _write_core_state(hdf5, with_legacy_optgeom=True)

    result = BDFOutputParser().read(hdf5_path=hdf5)

    current = result.results["geometry"]["current"]
    assert current["source"] == "hdf5_legacy"
    assert current["step_index"] == 4
    assert current["atoms"][1]["element"] == "H"
    assert result.results["geometry"]["atoms"] == []
    assert result.field_sources["results.geometry.current"] == "hdf5_legacy"


def test_read_hdf5_only_tddft_roots(tmp_path):
    hdf5 = tmp_path / "tddft.bdfh5"
    _write_core_state(hdf5, with_tddft_results=True)

    result = BDFOutputParser().read(hdf5_path=hdf5)

    tddft = result.results["tddft"]
    assert tddft["n_roots_found"] == 3
    assert tddft["itda"] == 0
    assert tddft["tda"] is False
    assert tddft["excited_states"][1]["energy_ev"] == pytest.approx(9.8445)
    assert tddft["excited_states"][1]["oscillator_strength"] == pytest.approx(0.0906)
    assert tddft["excited_states"][0]["quality"] == "normal"
    assert result.field_sources["results.tddft.states"] == "hdf5"
    assert result.field_sources["results.tddft.n_roots_found"] == "hdf5"


def test_read_hdf5_tddft_warning_flows_to_quality_and_diagnostics(tmp_path):
    hdf5 = tmp_path / "tddft-warning.bdfh5"
    _write_core_state(hdf5, with_tddft_results=True, tddft_warning=True)

    result = BDFOutputParser().read(hdf5_path=hdf5)

    warning = result.diagnostics["warnings"][0]
    assert warning["code"] == "TDDFT_UNSTABLE_REFERENCE_EXCITATION"
    assert warning["imaginary_complex_count"] == 1
    assert result.results["tddft"]["excited_states"][0]["quality"] == "imaginary_or_complex"
    assert result.quality["tddft_imaginary_complex_count"] == 1
    assert result.quality["tddft_unstable_reference"] is True
    assert result.field_sources["diagnostics.warnings"] == "hdf5"


def test_read_prefers_hdf5_tddft_roots_over_output_and_warns(tmp_path):
    out = tmp_path / "tddft.out"
    out.write_text(
        """
 Spin change :

  No. Pair   ExSym   ExEnergies     Wavelengths      f     D<S^2>          Dominant Excitations             IPA   Ova     En-E1
   1   A1    1  A1    7.4012 eV        167.52 nm   0.0000   0.0000   X A1(1)->A1(2)
   2   A1    2  A1    9.8445 eV        125.94 nm   0.0906   0.0000   X A1(1)->A1(3)
   3   A1    3  A1   10.1123 eV        122.61 nm   0.0111   0.0000   X A1(1)->A1(4)
 Congratulations! BDF normal termination
""",
        encoding="utf-8",
    )
    hdf5 = tmp_path / "tddft.bdfh5"
    _write_core_state(hdf5, with_tddft_results=True, tddft_energy_shift=0.01)

    result = BDFOutputParser().read(out_path=out, hdf5_path=hdf5)

    assert result.results["tddft"]["excited_states"][0]["energy_ev"] == pytest.approx(7.4112)
    assert result.field_sources["results.tddft.states"] == "hdf5"
    warnings = [w for w in result.consistency_warnings if w.field == "results.tddft.states[1].energy_ev"]
    assert warnings
    assert warnings[0].hdf5_value == pytest.approx(7.4112)
    assert warnings[0].output_value == pytest.approx(7.4012)


def test_read_prefers_hdf5_scf_energy_over_output_and_warns(tmp_path):
    out = tmp_path / "job.out"
    out.write_text(
        """
 E_tot = -76.0
 Congratulations! BDF normal termination
""",
        encoding="utf-8",
    )
    hdf5 = tmp_path / "job.bdfh5"
    _write_core_state(hdf5, with_scf_results=True, scf_energy=-77.0)

    result = BDFOutputParser().read(out_path=out, hdf5_path=hdf5)

    assert result.results["energies"]["scf_energy"] == pytest.approx(-77.0)
    assert result.field_sources["results.energies.scf_energy"] == "hdf5"
    warnings = [w for w in result.consistency_warnings if w.field == "results.energies.scf_energy"]
    assert warnings
    assert warnings[-1].hdf5_value == pytest.approx(-77.0)
    assert warnings[-1].output_value == pytest.approx(-76.0)


def test_read_reports_status_conflict_between_output_and_hdf5(tmp_path):
    out = tmp_path / "job.out"
    out.write_text(
        """
 E_tot = -76.0
 Congratulations! BDF normal termination
""",
        encoding="utf-8",
    )
    hdf5 = tmp_path / "job.bdfh5"
    _write_core_state(hdf5, status="failed", failed_module="scf")

    result = BDFOutputParser().read(out_path=out, hdf5_path=hdf5)

    assert result.run_status == UnifiedRunStatus.FAILED
    assert result.consistency_warnings
    warning = result.consistency_warnings[0]
    assert warning.field == "run_status"
    assert warning.hdf5_value == "failed"
    assert warning.output_value == "completed"


def test_read_extracts_structured_diagnostic_from_output(tmp_path):
    out = tmp_path / "job.out"
    out.write_text(
        """
[BDF_ERROR]
code: SCF_NOT_CONVERGED
category: SCF_CONVERGENCE
message: SCF did not converge
[/BDF_ERROR]
""",
        encoding="utf-8",
    )

    result = BDFOutputParser().read(out_path=out)

    assert result.run_status == UnifiedRunStatus.FAILED
    assert result.diagnostics["primary_failure"]["source"] == "output"
    assert result.diagnostics["primary_failure"]["code"] == "SCF_NOT_CONVERGED"
    assert result.diagnostics["primary_failure"]["category"] == "SCF_CONVERGENCE"
    assert result.field_sources["diagnostics.primary_failure"] == "output"


def test_read_records_missing_optional_refs(tmp_path):
    result = BDFOutputParser().read(
        out_path=tmp_path / "missing.out",
        out_tmp_path=tmp_path / "missing.out.tmp",
        hdf5_path=tmp_path / "missing.bdfh5",
    )

    assert result.run_status == UnifiedRunStatus.UNKNOWN
    assert result.parse_status == UnifiedParseStatus.UNAVAILABLE
    assert result.raw_refs["out"]["exists"] is False
    assert result.raw_refs["out_tmp"]["exists"] is False
    assert result.raw_refs["hdf5"]["exists"] is False
