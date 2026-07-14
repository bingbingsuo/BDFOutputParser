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
    assert "SCF_NOT_CONVERGED" in result.diagnostics["primary_failure"]["message"]


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
