"""
BDF Output Parser — BDFCoreState 只读 Inspector

读取 BDF 产出的 `BDFTASK.bdfh5` (HDF5, BDFCoreState-v1.0)，归一化为
``CoreStateSummary``。fail-closed：h5py 缺 / 文件缺 / schema 不符时返回
``available=False``，**不抛异常**，不影响调用方对 BDF 结果的判断。

只读、不写、不保持 HDF5 文件句柄跨 BDF 进程。

对照接口契约：``bdf-pkg-full/specs/BDFAssistant-BDFCoreState-Interface-v1.0.md`` §5/§6。
``_scalar`` helper 移植自 ``bdf-pkg-full/tools/check_bdfh5_core_state.py``。
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import CoreStateSummary

CORE_STATE_SCHEMA = "BDFCoreState-v1.0"


def _scalar(h5, path: str, default=None):
    """读 HDF5 scalar，处理 bytes / shape (1,) / numpy 标量。移植自 check 工具。"""
    if path not in h5:
        return default
    value = h5[path][()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "shape") and getattr(value, "shape", None) == (1,):
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
    return value


def _attrs_to_dict(group) -> dict:
    """把 HDF5 group 的 attrs 转成 JSON-safe dict（处理 bytes / numpy 标量 / array）。"""
    out = {}
    for k, v in group.attrs.items():
        if isinstance(v, bytes):
            out[k] = v.decode("utf-8", errors="replace")
        elif hasattr(v, "item"):
            try:
                out[k] = v.item()
                if isinstance(out[k], bytes):
                    out[k] = out[k].decode("utf-8", errors="replace")
            except Exception:
                out[k] = v
        elif hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _is_dataset(obj) -> bool:
    return hasattr(obj, "shape") and hasattr(obj, "dtype")


class BDFCoreStateInspector:
    """只读 ``.bdfh5`` 摘要器。不写 HDF5，不保持文件句柄。"""

    def read(self, path: str) -> CoreStateSummary:
        """读 ``.bdfh5`` → ``CoreStateSummary``。fail-closed，不抛异常。"""
        # 1. h5py 可用性
        try:
            import h5py
        except ImportError:
            return CoreStateSummary(
                available=False, reason="h5py_unavailable", path=str(path)
            )

        p = Path(path)
        if not p.is_file():
            return CoreStateSummary(
                available=False, reason="file_not_found", path=str(path)
            )

        try:
            with h5py.File(p, "r") as h5:
                return self._read_open(h5, str(path))
        except Exception as e:
            return CoreStateSummary(
                available=False, reason="read_error", path=str(path),
                provenance={"error": str(e)},
            )

    # ------------------------------------------------------------------
    # 内部：h5 已打开、已确认可读
    def _read_open(self, h5, path: str) -> CoreStateSummary:
        schema = str(_scalar(h5, "/meta/core_state_schema", "") or "")
        if schema != CORE_STATE_SCHEMA:
            return CoreStateSummary(
                available=False, reason="unsupported_schema", path=path, core_state_schema=schema,
            )

        s = CoreStateSummary(available=True, reason="ok", path=path, core_state_schema=schema)

        # /run (§6.2)
        s.status = str(_scalar(h5, "/run/status", "") or "")
        s.current_context_id = _scalar(h5, "/run/current_context_id", None)
        s.context_count = _scalar(h5, "/run/context_count", None)
        s.current_module = str(_scalar(h5, "/run/current_module", "") or "")
        s.last_successful_module = str(_scalar(h5, "/run/last_successful_module", "") or "")
        s.failed_module = str(_scalar(h5, "/run/failed_module", "") or "")
        s.interrupted_module = str(_scalar(h5, "/run/interrupted_module", "") or "")
        _restartable = _scalar(h5, "/run/restartable", None)
        s.restartable = bool(_restartable) if _restartable is not None else None
        s.started_unix = _scalar(h5, "/run/started_unix", None)
        s.completed_unix = _scalar(h5, "/run/completed_unix", None)
        s.elapsed_sec = _scalar(h5, "/run/elapsed_sec", None)

        # /input (§6.1)
        s.input_mode = str(_scalar(h5, "/input/input_mode", "") or "")

        # /workflows/000001 (§6.3)
        s.workflow = self._read_workflow(h5)

        # /aliases (§6.7)
        s.aliases = self._read_aliases(h5)

        # /objects (§6.6)
        s.objects = self._read_objects(h5)

        # /files/by_role (§6.5)
        s.files = self._read_files(h5)

        # /diagnostics (§6.9)
        s.diagnostics = self._read_diagnostics(h5)

        # /provenance (§6.10) — 关键字段摘要
        s.provenance = self._read_provenance(h5)

        return s

    def _read_workflow(self, h5) -> dict:
        wf = {}
        base = "/workflows/000001"
        if base not in h5:
            return wf
        wf["workflow_id"] = str(_scalar(h5, base + "/workflow_id", "") or "")
        wf["status"] = str(_scalar(h5, base + "/status", "") or "")
        plan_text = _scalar(h5, base + "/module_plan_json", "")
        try:
            wf["module_plan"] = json.loads(plan_text) if plan_text else []
        except Exception:
            wf["module_plan"] = []
        mods = {}
        mp = base + "/modules"
        if mp in h5:
            for name in h5[mp].keys():
                mods[name] = _attrs_to_dict(h5[f"{mp}/{name}"])
        wf["modules"] = mods
        return wf

    def _read_aliases(self, h5) -> dict:
        out = {}
        if "/aliases" not in h5:
            return out
        for category in ("orbitals", "geometries", "chkfil"):
            base = f"/aliases/{category}"
            if base not in h5:
                continue
            cat = {}
            for name in h5[base].keys():
                leaf = h5[f"{base}/{name}"]
                # 只读叶 alias（带 attrs 的 group），跳过子目录如 history
                if not _is_dataset(leaf):
                    cat[name] = _attrs_to_dict(leaf)
            out[category] = cat
        return out

    def _read_objects(self, h5) -> dict:
        out = {}
        if "/objects" not in h5:
            return out
        for kind in ("orbitals", "geometries", "chkfil_snapshots", "wavefunctions", "tddft"):
            base = f"/objects/{kind}"
            if base not in h5:
                continue
            items = []
            for name in h5[base].keys():
                leaf = h5[f"{base}/{name}"]
                if not _is_dataset(leaf):
                    items.append(_attrs_to_dict(leaf))
            if items:
                out[kind] = items
        return out

    def _read_files(self, h5) -> dict:
        out = {}
        base = "/files/by_role"
        if base not in h5:
            return out
        roles = {}
        for role in h5[base].keys():
            rpath = f"{base}/{role}"
            entries = []
            if rpath in h5:
                for fname in h5[rpath].keys():
                    leaf = h5[f"{rpath}/{fname}"]
                    if not _is_dataset(leaf):
                        entries.append(_attrs_to_dict(leaf))
            if entries:
                roles[role] = entries
        out["by_role"] = roles
        return out

    def _read_diagnostics(self, h5) -> dict:
        out = {}
        if "/diagnostics" not in h5:
            return out
        for module in h5["/diagnostics"].keys():
            lf = f"/diagnostics/{module}/last_failure"
            if lf in h5:
                out[module] = {"last_failure": _attrs_to_dict(h5[lf])}
        return out

    def _read_provenance(self, h5) -> dict:
        """只拉 agent 关心的关键 scalar，不递归整个嵌套。"""
        pulls = {
            "build": [
                "/provenance/build/version_text",
                "/provenance/build/source/git_commit",
                "/provenance/build/source/git_short_hash",
            ],
            "runtime": [
                "/provenance/runtime/python_version",
                "/provenance/runtime/hostname",
                "/provenance/runtime/workdir",
                "/provenance/runtime/h5py_version",
            ],
        }
        out = {}
        for section, paths in pulls.items():
            d = {}
            for p in paths:
                v = _scalar(h5, p, None)
                if v is not None and v != "":
                    d[p.split("/")[-1]] = v
            if d:
                out[section] = d
        return out
