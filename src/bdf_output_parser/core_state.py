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


def _dataset_list(h5, path: str) -> list | None:
    if path not in h5:
        return None
    value = h5[path][()]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    return value


def _decode_bdf_label(value) -> str:
    """Decode BDF fixed-width character labels stored as int64 cdafile data."""

    try:
        raw = int(value).to_bytes(8, byteorder="little", signed=False)
        text = raw.decode("ascii", errors="ignore").strip()
        if text:
            return text.capitalize()
    except (OverflowError, ValueError, TypeError):
        pass
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore").strip().capitalize()
    return str(value).strip().capitalize()


def _list_get(values: list, idx: int):
    return values[idx] if idx < len(values) else None


def _optional_float(target: dict, key: str, values: list, idx: int) -> None:
    value = _list_get(values, idx)
    if value is not None:
        target[key] = float(value)


def _optional_int(target: dict, key: str, values: list, idx: int) -> None:
    value = _list_get(values, idx)
    if value is not None:
        target[key] = int(value)


def _tddft_quality_label(flag: int) -> str:
    if flag == 1:
        return "imaginary_or_complex"
    if flag == 2:
        return "negative_excitation"
    if flag == 3:
        return "imaginary_or_complex_and_negative"
    return "normal"


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

        # /restart (Phase 6) — scratch preservation + restart capability
        s.restart = self._read_restart(h5)

        # /results and /contexts/<id>/results — first-slice scientific facts
        s.results = self._read_results(h5)

        return s

    def _read_results(self, h5) -> dict:
        """Read first-slice scientific results from HDF5.

        The first supported authority is SCF scalar data.  Top-level
        /results/scf is a latest cache; /contexts/<id>/results/scf preserves
        nested calculation history, such as repeated SCF calls from bdfopt.
        """
        out = {}

        latest = self._read_result_root(h5, "/results")
        if latest:
            out.update(latest)

        contexts = {}
        if "/contexts" in h5:
            for context_id in h5["/contexts"].keys():
                context_result = self._read_result_root(
                    h5,
                    f"/contexts/{context_id}/results",
                )
                if context_result:
                    contexts[context_id] = context_result
        if contexts:
            out["contexts"] = contexts

        return out

    def _read_result_root(self, h5, base: str) -> dict:
        result = {}

        energy = {}
        total_energy = _scalar(h5, f"{base}/energy/total_energy_hartree", None)
        if total_energy is not None:
            energy["total_energy_hartree"] = total_energy
        if energy:
            result["energy"] = energy

        scf = {}
        scf_energy = _scalar(h5, f"{base}/scf/scf_energy_hartree", None)
        if scf_energy is not None:
            scf["scf_energy_hartree"] = scf_energy
        converged = _scalar(h5, f"{base}/scf/converged", None)
        if converged is not None:
            scf["converged"] = bool(converged)
        n_iterations = _scalar(h5, f"{base}/scf/n_iterations", None)
        if n_iterations is not None:
            scf["n_iterations"] = int(n_iterations)
        source_context_id = _scalar(h5, f"{base}/scf/source_context_id", None)
        if source_context_id is not None:
            scf["source_context_id"] = int(source_context_id)
        source_module_ordinal = _scalar(h5, f"{base}/scf/source_module_ordinal", None)
        if source_module_ordinal is not None:
            scf["source_module_ordinal"] = int(source_module_ordinal)
        scf_path = f"{base}/scf"
        if scf_path in h5 and not _is_dataset(h5[scf_path]):
            attrs = _attrs_to_dict(h5[scf_path])
            if attrs:
                scf["attrs"] = attrs
        if scf:
            result["scf"] = scf

        geometry = self._read_geometry_results(h5, base)
        if geometry:
            result["geometry"] = geometry

        optimization = self._read_optimization_result(h5, base)
        if optimization:
            result["optimization"] = optimization

        tddft = self._read_tddft_result(h5, base)
        if tddft:
            result["tddft"] = tddft

        return result

    def _read_geometry_results(self, h5, base: str) -> dict:
        geometry = {}
        current = self._read_geometry_group(
            h5,
            f"{base}/geometry/current",
            coordinates_name="coordinates_bohr",
            source_default="hdf5",
        )
        final = self._read_geometry_group(
            h5,
            f"{base}/geometry/final",
            coordinates_name="coordinates_bohr",
            source_default="hdf5",
        )
        if current:
            geometry["current"] = current
        if final:
            geometry["final"] = final

        if base == "/results" and not current:
            legacy_current = self._read_geometry_group(
                h5,
                "/legacy/optgeom/current",
                coordinates_name="coordinates",
                source_default="hdf5_legacy",
            )
            if legacy_current:
                legacy_current["source"] = "hdf5_legacy"
                geometry["current"] = legacy_current
        return geometry

    def _read_geometry_group(
        self,
        h5,
        path: str,
        *,
        coordinates_name: str,
        source_default: str,
    ) -> dict:
        if path not in h5 or _is_dataset(h5[path]):
            return {}

        natom = _scalar(h5, f"{path}/natom", None)
        labels = _dataset_list(h5, f"{path}/labels")
        coordinates = _dataset_list(h5, f"{path}/{coordinates_name}")
        if coordinates is None and coordinates_name != "coordinates":
            coordinates = _dataset_list(h5, f"{path}/coordinates")
        if natom is None or labels is None or coordinates is None:
            return {}

        natom = int(natom)
        if natom <= 0 or len(labels) < natom or len(coordinates) < 3 * natom:
            return {}

        symbols = [_decode_bdf_label(labels[idx]) for idx in range(natom)]
        coordinate_rows = []
        atoms = []
        for idx, symbol in enumerate(symbols):
            xyz = [
                float(coordinates[3 * idx]),
                float(coordinates[3 * idx + 1]),
                float(coordinates[3 * idx + 2]),
            ]
            coordinate_rows.append(xyz)
            atoms.append(
                {
                    "element": symbol,
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
                    "units": "bohr",
                }
            )

        attrs = _attrs_to_dict(h5[path])
        result = {
            "natom": natom,
            "labels": symbols,
            "coordinates_bohr": coordinate_rows,
            "atoms": atoms,
            "source": str(attrs.get("source") or source_default),
            "optimizer": attrs.get("optimizer"),
            "attrs": attrs,
        }
        step_index = _scalar(h5, f"{path}/step_index", None)
        if step_index is None:
            step_index = _scalar(h5, f"{path}/igeom", None)
        if step_index is not None:
            result["step_index"] = int(step_index)
        return result

    def _read_optimization_result(self, h5, base: str) -> dict:
        opt_path = f"{base}/optimization"
        if opt_path not in h5 or _is_dataset(h5[opt_path]):
            return {}

        attrs = _attrs_to_dict(h5[opt_path])
        result = {}
        converged = _scalar(h5, f"{opt_path}/converged", None)
        if converged is not None:
            result["converged"] = bool(converged)
        n_steps = _scalar(h5, f"{opt_path}/n_steps", None)
        if n_steps is not None:
            result["n_steps"] = int(n_steps)
        max_steps = _scalar(h5, f"{opt_path}/max_steps", None)
        if max_steps is not None:
            result["max_steps"] = int(max_steps)
        info_code = _scalar(h5, f"{opt_path}/info_code", None)
        if info_code is not None:
            result["info_code"] = int(info_code)
        if attrs.get("info_label") is not None:
            result["info_label"] = attrs.get("info_label")
        if attrs.get("optimizer") is not None:
            result["optimizer"] = attrs.get("optimizer")
        if attrs.get("source") is not None:
            result["source"] = attrs.get("source")
        if attrs:
            result["attrs"] = attrs
        return result

    def _read_tddft_result(self, h5, base: str) -> dict:
        tddft_path = f"{base}/tddft"
        states_path = f"{tddft_path}/states"
        if tddft_path not in h5 or _is_dataset(h5[tddft_path]):
            return {}

        result: dict = {}
        attrs = _attrs_to_dict(h5[tddft_path])
        for key in (
            "itda",
            "tda",
            "isf",
            "n_roots_requested",
            "n_roots_found",
            "imaginary_complex_count",
            "negative_excitation_count",
        ):
            value = _scalar(h5, f"{tddft_path}/{key}", None)
            if value is None:
                continue
            if key == "tda":
                result[key] = bool(value)
            else:
                result[key] = int(value)
        if attrs:
            result["attrs"] = attrs

        root_index = _dataset_list(h5, f"{states_path}/root_index")
        energy_ev = _dataset_list(h5, f"{states_path}/energy_ev")
        wavelength_nm = _dataset_list(h5, f"{states_path}/wavelength_nm")
        oscillator = _dataset_list(h5, f"{states_path}/oscillator_strength")
        if not root_index or energy_ev is None or wavelength_nm is None or oscillator is None:
            return result

        n_states = min(len(root_index), len(energy_ev), len(wavelength_nm), len(oscillator))
        quality_flag = _dataset_list(h5, f"{states_path}/quality_flag") or []
        delta_s2 = _dataset_list(h5, f"{states_path}/delta_s2") or []
        dominant_percent = _dataset_list(h5, f"{states_path}/dominant_percent") or []
        ipa_ev = _dataset_list(h5, f"{states_path}/ipa_ev") or []
        ova = _dataset_list(h5, f"{states_path}/ova") or []
        irrep_index = _dataset_list(h5, f"{states_path}/irrep_index") or []
        state_index_in_irrep = _dataset_list(h5, f"{states_path}/state_index_in_irrep") or []
        states_attrs = _attrs_to_dict(h5[states_path]) if states_path in h5 else {}

        states = []
        for idx in range(n_states):
            state = {
                "index": int(root_index[idx]),
                "energy_ev": float(energy_ev[idx]),
                "wavelength_nm": float(wavelength_nm[idx]),
                "oscillator_strength": float(oscillator[idx]),
            }
            qflag = _list_get(quality_flag, idx)
            if qflag is not None:
                state["quality_flag"] = int(qflag)
                state["quality"] = _tddft_quality_label(int(qflag))
            _optional_float(state, "delta_s2", delta_s2, idx)
            _optional_float(state, "dominant_percent", dominant_percent, idx)
            _optional_float(state, "ipa_ev", ipa_ev, idx)
            _optional_float(state, "ova", ova, idx)
            _optional_int(state, "irrep_index", irrep_index, idx)
            _optional_int(state, "state_index_in_irrep", state_index_in_irrep, idx)
            states.append(state)

        result["states"] = states
        if states_attrs:
            result["states_attrs"] = states_attrs
        return result

    def _read_restart(self, h5) -> dict:
        """Read /restart/{scratch,modules,assets} (Phase 6 restart contract).

        Conservative: returns {} when the group is absent so callers can treat
        a missing restart capability as "no restart info" rather than failure.
        """
        restart = {}
        if "/restart" not in h5:
            return restart

        scratch_base = "/restart/scratch"
        if scratch_base in h5:
            restart["scratch"] = _attrs_to_dict(h5[scratch_base])

        modules = {}
        modules_base = "/restart/modules"
        if modules_base in h5:
            for name in h5[modules_base]:
                modules[name] = _attrs_to_dict(h5[modules_base + "/" + name])
        if modules:
            restart["modules"] = modules

        assets = {}
        assets_base = "/restart/assets"
        if assets_base in h5:
            for name in h5[assets_base]:
                assets[name] = _attrs_to_dict(h5[assets_base + "/" + name])
        if assets:
            restart["assets"] = assets

        return restart

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
            module_data = {}
            lf = f"/diagnostics/{module}/last_failure"
            if lf in h5:
                module_data["last_failure"] = _attrs_to_dict(h5[lf])
            lw = f"/diagnostics/{module}/last_warning"
            if lw in h5:
                module_data["last_warning"] = _attrs_to_dict(h5[lw])
            failures = self._read_diagnostic_records(h5, f"/diagnostics/{module}/failures")
            if failures:
                module_data["failures"] = failures
            warnings = self._read_diagnostic_records(h5, f"/diagnostics/{module}/warnings")
            if warnings:
                module_data["warnings"] = warnings
            if module_data:
                out[module] = module_data
        return out

    def _read_diagnostic_records(self, h5, base: str) -> dict:
        records = {}
        if base not in h5:
            return records
        for name in h5[base].keys():
            leaf = h5[f"{base}/{name}"]
            if not _is_dataset(leaf):
                records[name] = _attrs_to_dict(leaf)
        return records

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
