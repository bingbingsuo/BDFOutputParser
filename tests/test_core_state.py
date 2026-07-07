"""BDFCoreStateInspector 只读摘要器测试。

覆盖：真实 test003.bdfh5、合成 completed/failed/interrupted/orbitals-alias、
missing file、unsupported schema、missing h5py（fail-closed）。
"""

import sys
from pathlib import Path

import pytest

# 合成 .bdfh5 与真实读取都需要 h5py；环境无 h5py 则整个文件跳过。
h5py = pytest.importorskip("h5py")

from bdf_output_parser import BDFCoreStateInspector, CoreStateSummary  # noqa: E402

REAL_TEST003 = Path("/Users/bsuo/tests/bdf/debug/test003.bdfh5")


def _make_bdfh5(path, *, schema="BDFCoreState-v1.0", status="completed",
                failed_module="", interrupted_module="",
                last_successful_module="scf",
                with_orbital_alias=False, with_diagnostics=False,
                with_restart=False):
    """写一个最小但结构合法的合成 .bdfh5。"""
    with h5py.File(path, "w") as f:
        f.create_dataset("meta/core_state_schema", data=schema)
        f.create_dataset("input/input_mode", data="easyinput")
        f.create_dataset("run/status", data=status)
        f.create_dataset("run/last_successful_module", data=last_successful_module)
        f.create_dataset("run/restartable", data=0)
        f.create_dataset("run/elapsed_sec", data=1.5)
        if failed_module:
            f.create_dataset("run/failed_module", data=failed_module)
        if interrupted_module:
            f.create_dataset("run/interrupted_module", data=interrupted_module)
        # workflow
        f.create_dataset("workflows/000001/workflow_id", data="workflow:000001")
        f.create_dataset("workflows/000001/status", data=status)
        f.create_dataset(
            "workflows/000001/module_plan_json",
            data='[{"module":"compass","step":1},{"module":"scf","step":2}]',
        )
        if with_orbital_alias:
            obj = f.create_group("objects/orbitals/000001")
            obj.attrs["object_id"] = "000001"
            obj.attrs["object_kind"] = "orbital"
            obj.attrs["status"] = "complete"
            obj.attrs["path"] = "test.scforb"
            obj.attrs["storage_policy"] = "reference_only"
            al = f.create_group("aliases/orbitals/latest_scf")
            al.attrs["target"] = "/objects/orbitals/000001"
            al.attrs["target_kind"] = "orbital"
            al.attrs["updated_by"] = "scf"
        if with_diagnostics:
            lf = f.create_group("diagnostics/scf/last_failure")
            lf.attrs["reason"] = "SCF did not converge"
            lf.attrs["failure_class"] = "controlled_error"
            lf.attrs["restartable"] = 1
        if with_restart:
            scratch = f.create_group("restart/scratch")
            scratch.attrs["tmpdir"] = "/tmp/bdf-scratch"
            scratch.attrs["preserved"] = True
            scratch.attrs["preserve_reason"] = "module_failure"
            scratch.attrs["lock_files_removed"] = True
            mod = f.create_group("restart/modules/000002")
            mod.attrs["module"] = "scf"
            mod.attrs["ordinal"] = 2
            mod.attrs["status"] = "failed"
            mod.attrs["restartable_from_here"] = "planned"
            mod.attrs["support_status"] = "planned_first"
            asset = f.create_group("restart/assets/scforb")
            asset.attrs["kind"] = "scforb"
            asset.attrs["producer_module"] = "scf"
            asset.attrs["exists"] = True
            asset.attrs["persistent"] = False
            asset.attrs["required_for"] = "scf_readmo_restart"


# -----------------------------------------------------------------------------
# 真实数据（test003.bdfh5 — bdfopt, BDFCoreState-v1.0）
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_TEST003.exists(), reason="test003.bdfh5 不在本机")
class TestRealTest003:
    def test_completed_status(self):
        s = BDFCoreStateInspector().read(str(REAL_TEST003))
        assert s.available is True
        assert s.reason == "ok"
        assert s.core_state_schema == "BDFCoreState-v1.0"
        assert s.status == "completed"
        assert s.last_successful_module == "bdfopt"
        assert s.input_mode == "fortran"
        assert s.restartable is False
        assert s.elapsed_sec == 5.0

    def test_geometry_alias_and_objects(self):
        s = BDFCoreStateInspector().read(str(REAL_TEST003))
        cur = s.aliases.get("geometries", {}).get("current", {})
        assert cur.get("target") == "/objects/geometries/000002"
        assert cur.get("target_kind") == "geometry"
        assert len(s.objects.get("geometries", [])) == 2

    def test_workflow_module_plan(self):
        s = BDFCoreStateInspector().read(str(REAL_TEST003))
        modules = [m.get("module") for m in s.workflow.get("module_plan", [])]
        assert "compass" in modules and "bdfopt" in modules
        # 两个已完成的 module 节点
        assert set(s.workflow.get("modules", {}).keys()) == {"000000", "000001"}

    def test_files_by_role_records_missing_output(self):
        s = BDFCoreStateInspector().read(str(REAL_TEST003))
        roles = s.files.get("by_role", {})
        assert "chkfil" in roles and "output" in roles
        # test003 的 output 文件 state=missing（如实记录）
        assert any(e.get("state") == "missing" for e in roles.get("output", []))

    def test_provenance_git(self):
        s = BDFCoreStateInspector().read(str(REAL_TEST003))
        assert s.provenance["build"]["git_short_hash"] == "5c82e7578"

    def test_json_serializable(self):
        s = BDFCoreStateInspector().read(str(REAL_TEST003))
        import json
        d = json.loads(s.model_dump_json())
        assert d["available"] is True
        assert d["status"] == "completed"


# -----------------------------------------------------------------------------
# 合成场景
# -----------------------------------------------------------------------------
class TestSynthetic:
    def test_completed_run(self, tmp_path):
        p = tmp_path / "ok.bdfh5"
        _make_bdfh5(p, status="completed", last_successful_module="scf")
        s = BDFCoreStateInspector().read(str(p))
        assert s.available is True
        assert s.status == "completed"
        assert s.last_successful_module == "scf"

    def test_restart_contract_summary(self, tmp_path):
        p = tmp_path / "restart.bdfh5"
        _make_bdfh5(
            p,
            status="failed",
            failed_module="scf",
            last_successful_module="xuanyuan",
            with_restart=True,
        )
        s = BDFCoreStateInspector().read(str(p))
        assert s.available is True
        assert s.restart["scratch"]["preserved"] is True
        assert s.restart["scratch"]["preserve_reason"] == "module_failure"
        assert s.restart["modules"]["000002"]["support_status"] == "planned_first"
        assert s.restart["assets"]["scforb"]["required_for"] == "scf_readmo_restart"
        assert s.input_mode == "easyinput"
        assert s.workflow["status"] == "failed"
        assert len(s.workflow["module_plan"]) == 2

    def test_failed_run_with_diagnostics(self, tmp_path):
        p = tmp_path / "fail.bdfh5"
        _make_bdfh5(p, status="failed", failed_module="scf", with_diagnostics=True)
        s = BDFCoreStateInspector().read(str(p))
        assert s.status == "failed"
        assert s.failed_module == "scf"
        diag = s.diagnostics.get("scf", {}).get("last_failure", {})
        assert diag.get("reason") == "SCF did not converge"
        assert diag.get("failure_class") == "controlled_error"

    def test_interrupted_run(self, tmp_path):
        p = tmp_path / "int.bdfh5"
        _make_bdfh5(p, status="interrupted", interrupted_module="scf")
        s = BDFCoreStateInspector().read(str(p))
        assert s.status == "interrupted"
        assert s.interrupted_module == "scf"

    def test_orbital_alias(self, tmp_path):
        """本地无 SCF 新格式样本，用合成 .bdfh5 验证 orbitals alias 解析。"""
        p = tmp_path / "scf.bdfh5"
        _make_bdfh5(p, with_orbital_alias=True)
        s = BDFCoreStateInspector().read(str(p))
        assert len(s.objects.get("orbitals", [])) == 1
        al = s.aliases.get("orbitals", {}).get("latest_scf", {})
        assert al.get("target") == "/objects/orbitals/000001"
        assert al.get("target_kind") == "orbital"


# -----------------------------------------------------------------------------
# fail-closed
# -----------------------------------------------------------------------------
class TestFailClosed:
    def test_missing_file(self, tmp_path):
        s = BDFCoreStateInspector().read(str(tmp_path / "nope.bdfh5"))
        assert s.available is False
        assert s.reason == "file_not_found"

    def test_unsupported_schema(self, tmp_path):
        p = tmp_path / "old.bdfh5"
        _make_bdfh5(p, schema="BDFContextHDF5-v1.0")
        s = BDFCoreStateInspector().read(str(p))
        assert s.available is False
        assert s.reason == "unsupported_schema"
        assert s.core_state_schema == "BDFContextHDF5-v1.0"

    def test_missing_h5py(self, tmp_path, monkeypatch):
        p = tmp_path / "x.bdfh5"
        p.write_bytes(b"")
        # 模拟 h5py 未安装：sys.modules['h5py'] = None 会让 import h5py 抛 ImportError
        monkeypatch.setitem(sys.modules, "h5py", None)
        s = BDFCoreStateInspector().read(str(p))
        assert s.available is False
        assert s.reason == "h5py_unavailable"

    def test_corrupt_file_read_error(self, tmp_path):
        p = tmp_path / "corrupt.bdfh5"
        p.write_bytes(b"not an hdf5 file")
        s = BDFCoreStateInspector().read(str(p))
        assert s.available is False
        assert s.reason == "read_error"

    def test_tolerates_missing_optional_fields(self, tmp_path):
        """只有 meta+run/status 的极简 .bdfh5 也能读，其余字段空。"""
        p = tmp_path / "minimal.bdfh5"
        with h5py.File(p, "w") as f:
            f.create_dataset("meta/core_state_schema", data="BDFCoreState-v1.0")
            f.create_dataset("run/status", data="running")
        s = BDFCoreStateInspector().read(str(p))
        assert s.available is True
        assert s.status == "running"
        assert s.workflow == {}
        assert s.aliases == {}
