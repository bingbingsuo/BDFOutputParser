"""
BDF Output Parser — Orbital Classifier

根据原子电子组态 + SAO 解析结果，自动区分芯层(core)和价层(active)轨道。
为 MCSCF/MRCI 活性空间选择提供每不可约表示的轨道数。
"""

from __future__ import annotations

from typing import Optional

from .elements import ELEMENTS
from .models import (
    AOLabel,
    IrrepClassification,
    IrrepSAO,
    OrbitalClassification,
    SAOParseResult,
)

# 惰性气体序列（按原子序数递增）
_NOBLE_GASES = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]
_L_LETTERS = "SPDFGHI"


class OrbitalClassifier:
    """根据 SAO 数据 + 原子组态区分 core/active 轨道"""

    def classify(
        self,
        sao: SAOParseResult,
        atoms: list[str],
        overrides: dict[str, list[str]] | None = None,
    ) -> OrbitalClassification:
        """
        三层级分类: frozen_core / outer_core / valence，每不可约表示。

        frozen_core: 惰性气体芯中 n < max_core_n 的轨道（永不激发）
        outer_core:  惰性气体芯中 n = max_core_n 的轨道（MRCI 可激发）
        valence:     惰性气体芯之外的轨道（MCSCF/MRCI 活性空间）

        overrides 允许用户调整默认分类:
          {"frozen_core": ["U:6p", "F:2s"],  # 移入 frozen_core
           "outer_core":  ["U:5d"],           # 移入 outer_core
           "valence":     ["U:5d", "F:2p"]}   # 移入 valence

        格式: "Element:nl" — "U:5f", "F:2p"
        多元素同覆盖: 对每个 atom_index 匹配 element symbol 执行覆盖。
        """
        overrides = overrides or {}
        atom_frozen: dict[int, set[tuple[int, str]]] = {}
        atom_outer:  dict[int, set[tuple[int, str]]] = {}
        atom_valence: dict[int, set[tuple[int, str]]] = {}

        for idx, sym in enumerate(atoms, start=1):
            fz, oc, vl = self._get_shell_tiers(sym)

            # 应用用户覆盖
            fz, oc, vl = self._apply_overrides(sym, fz, oc, vl, overrides)

            atom_frozen[idx] = fz
            atom_outer[idx] = oc
            atom_valence[idx] = vl

        # 汇总电子数
        n_fz = n_oc = n_vl = 0
        for sym in atoms:
            fz, oc, vl = self._get_shell_tiers(sym)
            fz, oc, vl = self._apply_overrides(sym, fz, oc, vl, overrides)
            config = ELEMENTS[sym].eleconfig_dict
            for (n, letter), occ in config.items():
                key = (n, letter.upper())
                if key in fz:
                    n_fz += occ
                elif key in oc:
                    n_oc += occ
                else:
                    n_vl += occ

        # 对每个 irrep 分类
        per_irrep: list[IrrepClassification] = []
        for irrep_data in sao.irreps:
            fz_set: set[tuple] = set()
            oc_set: set[tuple] = set()
            vl_set: set[tuple] = set()
            fz_labels, oc_labels, vl_labels = [], [], []

            for sao_line in irrep_data.saos:
                for ao in sao_line.aos:
                    label = f"{ao.atom_index}{ao.element}{ao.n}{ao.l}{ao.m}"
                    key = (ao.n, ao.l)
                    if key in atom_frozen.get(ao.atom_index, set()):
                        fz_labels.append(label)
                        fz_set.add(label)
                    elif key in atom_outer.get(ao.atom_index, set()):
                        oc_labels.append(label)
                        oc_set.add(label)
                    else:
                        vl_labels.append(label)
                        vl_set.add(label)

            per_irrep.append(IrrepClassification(
                irrep=irrep_data.irrep,
                norb=irrep_data.norb,
                frozen_core_labels=fz_labels,
                outer_core_labels=oc_labels,
                valence_labels=vl_labels,
                n_frozen_core=len(fz_labels),
                n_outer_core=len(oc_labels),
                n_valence=len(vl_labels),
                # 去重轨道数（每 (atom,n,l,m) 唯一）
                n_frozen_core_orbitals=len(fz_set),
                n_outer_core_orbitals=len(oc_set),
                n_valence_orbitals=len(vl_set),
            ))

        # 全局去重: 同标签的 AO 在对称性等价的 irreps 中可能重复出现
        all_vl_labels = set()
        for ir in per_irrep:
            all_vl_labels.update(ir.valence_labels)
        total_vl_orbs = len(all_vl_labels)

        return OrbitalClassification(
            molecule="".join(atoms),
            point_group=sao.point_group or "",
            n_electrons=n_fz + n_oc + n_vl,
            n_frozen_core_electrons=n_fz,
            n_outer_core_electrons=n_oc,
            n_valence_electrons=n_vl,
            total_valence_orbitals=total_vl_orbs,
            per_irrep=per_irrep,
        )

    @staticmethod
    def _get_shell_tiers(
        symbol: str,
    ) -> tuple[set[tuple[int, str]], set[tuple[int, str]], set[tuple[int, str]]]:
        """返回 (frozen_core, outer_core, valence) 三层级壳层集合。

        frozen_core: noble gas core 中 n < max_core_n（深芯，永不激发）
        outer_core:  noble gas core 中 n = max_core_n（MRCI 可激发）
        valence:     noble gas core 之外（MCSCF/MRCI 活性）

        U: [Rn] 5f3 6d 7s2
           Rn max_n=6 → frozen={n≤5}, outer={n=6}, valence={5f,6d,7s}
        O: [He] 2s2 2p4
           He max_n=1 → frozen={1s}, outer={}, valence={2s,2p}
        H: 1s1 → frozen={}, outer={}, valence={1s}
        """
        ele = ELEMENTS[symbol]
        config = ele.eleconfig
        max_core_n = 0
        core_config: dict[tuple[int, str], int] = {}

        if config.startswith("["):
            bracket_end = config.index("]")
            noble_gas = config[1:bracket_end]
            if noble_gas in ELEMENTS:
                core_config = dict(ELEMENTS[noble_gas].eleconfig_dict)
                if core_config:
                    max_core_n = max(n for (n, _) in core_config)

        frozen_core = {(n, letter.upper()) for (n, letter) in core_config
                       if n < max_core_n}
        outer_core = {(n, letter.upper()) for (n, letter) in core_config
                      if n == max_core_n}

        all_config = ele.eleconfig_dict
        valence = {(n, letter.upper()) for (n, letter) in all_config} - frozen_core - outer_core

        return frozen_core, outer_core, valence

    @staticmethod
    def _apply_overrides(
        symbol: str,
        fz: set[tuple[int, str]],
        oc: set[tuple[int, str]],
        vl: set[tuple[int, str]],
        overrides: dict[str, list[str]],
    ) -> tuple[set, set, set]:
        """应用用户覆盖规则。返回修改后的 (frozen, outer, valence) 集合。

        覆盖格式: "Element:nl" — "U:5f", "F:2p"
        仅覆盖与 symbol 匹配的元素。
        """
        fz, oc, vl = set(fz), set(oc), set(vl)
        all_shells = fz | oc | vl

        for tier_key, spec_list in overrides.items():
            if tier_key not in ("frozen_core", "outer_core", "valence"):
                continue
            for spec in spec_list:
                shell = _parse_override_spec(spec)
                if shell is None:
                    continue
                elem, n, l = shell
                if elem.upper() != symbol.upper():
                    continue
                key = (n, l.upper())
                if key not in all_shells:
                    continue
                # 从所有集合中移除，再添加到目标集合
                fz.discard(key)
                oc.discard(key)
                vl.discard(key)
                if tier_key == "frozen_core":
                    fz.add(key)
                elif tier_key == "outer_core":
                    oc.add(key)
                else:
                    vl.add(key)

        return fz, oc, vl


def _parse_override_spec(spec: str) -> tuple[str, int, str] | None:
    """解析 'U:5f' → ('U', 5, 'F')。"""
    import re
    m = re.match(r'^([A-Z][a-z]?):(\d+)([spdfghi])$', spec, re.IGNORECASE)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def _detect_effective_group(sao: SAOParseResult) -> str:
    """从 SAO irreps 反推有效阿贝尔点群（不依赖输出中印的 Point group name 文本）。

    当用户指定 `group D2h` 后，BDF 以 D2h 分解 SAO，但输出仍打印全对称性。
    此函数检查 irrep 名称判断实际使用的点群。
    """
    if not sao.irreps:
        return (sao.point_group or "").upper()

    irrep_names = {ir.irrep.upper() for ir in sao.irreps if ir.irrep}
    # 去除空 irrep (norb=0) 和可能的空白

    # D2h irreps: Ag, B1g, B2g, B3g, Au, B1u, B2u, B3u
    d2h_set = {"AG", "B1G", "B2G", "B3G", "AU", "B1U", "B2U", "B3U"}
    if irrep_names.issubset(d2h_set):
        return "D2H"

    # D2 irreps: A, B1, B2, B3
    d2_set = {"A", "B1", "B2", "B3"}
    if irrep_names.issubset(d2_set):
        return "D2"

    # C2v irreps: A1, A2, B1, B2
    c2v_set = {"A1", "A2", "B1", "B2"}
    if irrep_names.issubset(c2v_set):
        return "C2V"

    # C2h irreps: Ag, Bg, Au, Bu
    c2h_set = {"AG", "BG", "AU", "BU"}
    if irrep_names.issubset(c2h_set):
        return "C2H"

    # Cs irreps: A', A"
    cs_set = {"A'", "A''", "A\'", "A\'\'"}
    if irrep_names.issubset(cs_set):
        return "CS"

    # C2 irreps: A, B
    c2_set = {"A", "B"}
    if irrep_names.issubset(c2_set):
        return "C2"

    # C1: just A
    if irrep_names == {"A"} or irrep_names == {"AG"}:
        return "C1"

    # Fallback: use printed point group
    return (sao.point_group or "").upper()


def classify_with_symmetry(
    sao: SAOParseResult,
    atoms: list[str],
) -> dict:
    """
    完整的对称性感知分类流程。

    返回 dict:
      abelian: bool — 有效点群是否为阿贝尔群
      point_group: str — 原始打印的点群
      effective_group: str — 从 irrep 反推的有效点群
      subgroup: str | None — 若非阿贝尔群，推荐的最大阿贝尔子群
      message: str — 给用户的建议
      classification: OrbitalClassification | None — 若是阿贝尔群，直接分类
    """
    from .symmetry import is_abelian, recommend_subgroup

    classifier = OrbitalClassifier()
    pg_print = (sao.point_group or "").upper()
    pg_effective = _detect_effective_group(sao)

    if not pg_effective:
        return {
            "abelian": True,
            "point_group": pg_print,
            "effective_group": "",
            "subgroup": None,
            "message": "未检测到分子对称性。",
            "classification": classifier.classify(sao, atoms),
        }

    if is_abelian(pg_effective):
        return {
            "abelian": True,
            "point_group": pg_print,
            "effective_group": pg_effective,
            "subgroup": None,
            "message": f"有效点群 {pg_effective} 为阿贝尔群，MCSCF/MRCI 可直接使用。",
            "classification": classifier.classify(sao, atoms),
        }

    subgroup = recommend_subgroup(pg_effective)
    if subgroup:
        msg = (
            f"点群 {pg_effective} 为非阿贝尔群，BDF 多组态方法不支持。"
            f"建议使用最大阿贝尔子群 {subgroup}。"
            f"请重新执行 checksymm 并设置 group {subgroup}。"
        )
    else:
        msg = f"点群 {pg_effective} 为非阿贝尔群，且未找到推荐的阿贝尔子群。建议手动指定 D2h 或更低对称性。"

    return {
        "abelian": False,
        "point_group": pg_print,
        "effective_group": pg_effective,
        "subgroup": subgroup,
        "message": msg,
        "classification": None,
    }
