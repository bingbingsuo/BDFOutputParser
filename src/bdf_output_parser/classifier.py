"""
BDF Output Parser — Orbital Classifier

五层级自动分类 + 用户覆盖 → MCSCF 输入参数。

五层级:
  frozen_core: [Rn] n<max_n, 深芯 → close (与 outer_core 合并, 计算量不大)
  outer_core:  [Rn] n=max_n, 外芯 → close
  inactive:    valence 满占据 → close (或 active, 用户决定)
  active:      valence 部分占据 → active + actel
  virtual:     未占据基函数 → 活性空间中空轨道自动处理

MCSCF 关键词:
  close:  每 irrep 的 fz+oc (+ optional inactive) 轨道数
  active: 每 irrep 的 active 轨道数
  actel:  active 电子总数
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
        五层级分类: frozen_core / outer_core / inactive / active / virtual。

        frozen_core: [Rn] n < max_core_n（深层芯，永不激发）
        outer_core:  [Rn] n = max_core_n（外层芯，MRCI 可关联）
        inactive:    valence shell 满占据（2×(2l+1) e⁻）
        active:      valence shell 部分占据（MCSCF 活性空间）
        virtual:     所有未占据基函数

        overrides 格式:
          {"frozen_core": ["F:2s", "F:2p"], "inactive": ["U:7s"]}
        """
        overrides = overrides or {}
        atom_tiers: dict[int, dict[str, set[tuple[int, str]]]] = {}

        for idx, sym in enumerate(atoms, start=1):
            fz, oc, vl = self._get_shell_tiers(sym)

            # 周期表价轨道全部归入 active（不按占据数拆分）
            active = vl
            inactive: set[tuple[int, str]] = set()

            fz, oc, inactive, active = self._apply_overrides_v2(
                sym, fz, oc, inactive, active, overrides,
            )

            atom_tiers[idx] = {
                "frozen_core": fz, "outer_core": oc,
                "inactive": inactive, "active": active,
            }

        # 汇总电子数
        e_fz = e_oc = e_inact = e_act = 0
        for sym in atoms:
            fz, oc, vl = self._get_shell_tiers(sym)
            active = vl
            inactive = set()
            fz, oc, inactive, active = self._apply_overrides_v2(
                sym, fz, oc, inactive, active, overrides,
            )
            config = ELEMENTS[sym].eleconfig_dict
            for (n, letter), occ in config.items():
                key = (n, letter.upper())
                if key in fz:
                    e_fz += occ
                elif key in oc:
                    e_oc += occ
                elif key in inactive:
                    e_inact += occ
                elif key in active:
                    e_act += occ

        # 对每个 irrep，按 SAO（分子轨道）分类
        # 每个 SAO 取其主导 AO 所在的 tier，优先级: active > inactive > outer > frozen > virtual
        _TIER_PRIORITY = {"active": 0, "inactive": 1, "outer_core": 2, "frozen_core": 3, "virtual": 4}
        tier_keys = ("frozen_core", "outer_core", "inactive", "active", "virtual")
        per_irrep: list[IrrepClassification] = []
        global_shell_sets: dict[str, dict[str, set[str]]] = {k: {} for k in tier_keys}

        for irrep_data in sao.irreps:
            sao_tiers: dict[str, list[int]] = {k: [] for k in tier_keys}
            all_ao_labels: dict[str, set[str]] = {k: set() for k in tier_keys}
            sao_labels: dict[str, list[str]] = {k: [] for k in tier_keys}
            irrep_shell_sets: dict[str, dict[str, set[str]]] = {k: {} for k in tier_keys}

            for idx, sao_line in enumerate(irrep_data.saos):
                # 找出此 SAO 中主导 AO 所在的 tier
                best_tier = "virtual"
                best_prio = _TIER_PRIORITY["virtual"]
                ao_info: list[tuple[str, str]] = []
                for ao in sao_line.aos:
                    label = f"{ao.atom_index}{ao.element}{ao.n}{ao.l}{ao.m}"
                    tiers = atom_tiers.get(ao.atom_index, {})
                    key = (ao.n, ao.l)
                    tier = self._lookup_tier(key, tiers)
                    ao_info.append((label, tier))
                    all_ao_labels[tier].add(label)
                    prio = _TIER_PRIORITY.get(tier, 4)
                    if prio < best_prio:
                        best_prio = prio
                        best_tier = tier

                # 将 SAO 计为主导 tier 的分子轨道
                sao_tiers[best_tier].append(idx)
                for label, tier in ao_info:
                    sao_labels[tier].append(label)

            irrep_summary = _build_shell_summary(sao_labels, atoms)
            global_shell_sets = _merge_shell_sets(global_shell_sets, irrep_shell_sets, sao_labels, atoms)

            per_irrep.append(IrrepClassification(
                irrep=irrep_data.irrep, norb=irrep_data.norb,
                frozen_core_labels=sao_labels["frozen_core"],
                outer_core_labels=sao_labels["outer_core"],
                inactive_labels=sao_labels["inactive"],
                active_labels=sao_labels["active"],
                virtual_labels=sao_labels["virtual"],
                n_frozen_core_orbitals=len(sao_tiers["frozen_core"]),
                n_outer_core_orbitals=len(sao_tiers["outer_core"]),
                n_inactive_orbitals=len(sao_tiers["inactive"]),
                n_active_orbitals=len(sao_tiers["active"]),
                n_virtual_orbitals=len(sao_tiers["virtual"]),
                summary=irrep_summary,
            ))

        # 全局去重 shell summary
        global_summary: dict[str, dict[str, str]] = {}
        for tier in tier_keys:
            global_summary[tier] = {}
            for elem in sorted(global_shell_sets.get(tier, {})):
                shells = sorted(global_shell_sets[tier][elem], key=_shell_sort_key)
                global_summary[tier][elem] = _format_shell_list(shells)

        # 全局汇总（跨 irrep 去重 SAO 计数）
        total_inact = sum(ir.n_inactive_orbitals for ir in per_irrep)
        total_act = sum(ir.n_active_orbitals for ir in per_irrep)
        total_virt = sum(ir.n_virtual_orbitals for ir in per_irrep)

        return OrbitalClassification(
            summary=global_summary,
            molecule="".join(atoms),
            point_group=sao.point_group or "",
            n_electrons=e_fz + e_oc + e_inact + e_act,
            n_frozen_core_electrons=e_fz,
            n_outer_core_electrons=e_oc,
            n_inactive_electrons=e_inact,
            n_active_electrons=e_act,
            total_inactive_orbitals=total_inact,
            total_active_orbitals=total_act,
            total_virtual_orbitals=total_virt,
            n_basis=sao.n_basis,
            per_irrep=per_irrep,
        )

    @staticmethod
    def _split_valence(
        symbol: str, valence: set[tuple[int, str]]
    ) -> tuple[set[tuple[int, str]], set[tuple[int, str]]]:
        """将 valence shells 拆分为 inactive（满占据）和 active（部分占据）。"""
        ele = ELEMENTS[symbol]
        config = ele.eleconfig_dict
        inactive, active = set(), set()
        for (n, l_letter) in valence:
            n_letter = l_letter.upper()
            occ = config.get((n, n_letter.lower()), 0)
            l_val = "SPDFGHI".index(n_letter)
            capacity = 2 * (2 * l_val + 1)
            if occ >= capacity:
                inactive.add((n, n_letter))
            else:
                active.add((n, n_letter))
        return inactive, active

    @staticmethod
    def _lookup_tier(
        key: tuple[int, str], tiers: dict[str, set[tuple[int, str]]]
    ) -> str:
        for tier in ("frozen_core", "outer_core", "inactive", "active"):
            if key in tiers.get(tier, set()):
                return tier
        return "virtual"

    @staticmethod
    def _apply_overrides_v2(
        symbol: str,
        fz: set, oc: set, inactive: set, active: set,
        overrides: dict[str, list[str]],
    ) -> tuple[set, set, set, set]:
        """应用用户覆盖到五层级。"""
        fz, oc, inactive, active = set(fz), set(oc), set(inactive), set(active)
        all_shells = fz | oc | inactive | active
        valid_tiers = {"frozen_core", "outer_core", "inactive", "active"}

        for tier_key, spec_list in overrides.items():
            if tier_key not in valid_tiers:
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
                for s in (fz, oc, inactive, active):
                    s.discard(key)
                {"frozen_core": fz, "outer_core": oc,
                 "inactive": inactive, "active": active}[tier_key].add(key)

        return fz, oc, inactive, active

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


def _shell_sort_key(shell: str) -> tuple[int, str]:
    """排序键: '5f' → (5, 'f')。"""
    n = int(''.join(c for c in shell if c.isdigit()) or '0')
    l = ''.join(c for c in shell if c.isalpha()).lower()
    return (n, l)


def _format_shell_list(shells: list[str]) -> str:
    """合并相同 n 的壳层: ['5d','5f','6s','6p'] → '5df, 6sp'。"""
    if not shells:
        return ""
    grouped: dict[int, list[str]] = {}
    for s in shells:
        n = int(''.join(c for c in s if c.isdigit()) or '0')
        l = ''.join(c for c in s if c.isalpha()).lower()
        grouped.setdefault(n, []).append(l)
    parts = []
    for n in sorted(grouped):
        parts.append(f"{n}{''.join(sorted(grouped[n]))}")
    return ", ".join(parts)


def _build_shell_summary(
    sao_labels: dict[str, list[str]], atoms: list[str]
) -> dict[str, dict[str, str]]:
    """从 SAO 标签中提取每 tier 的原子壳层组成。

    标签格式: "1U5F-2" → atom_index=1, element=U, n=5, l=F, m=-2
    """
    import re as _r
    tier_keys = ("frozen_core", "outer_core", "inactive", "active", "virtual")
    result: dict[str, dict[str, str]] = {}
    for tier in tier_keys:
        shells: dict[str, set[str]] = {}  # "U" → {"5f", "6d", ...}
        for label in sao_labels.get(tier, []):
            m = _r.match(r'(\d+)([A-Z][a-z]?)(\d+)([A-Z])(-?\d+)', label)
            if m:
                element = m.group(2)
                shell = f"{m.group(3)}{m.group(4).lower()}"
                shells.setdefault(element, set()).add(shell)
        result[tier] = {
            elem: _format_shell_list(sorted(list(s), key=_shell_sort_key))
            for elem, s in sorted(shells.items(),
                                   key=lambda x: (len(atoms) > 1 and atoms.index(x[0]) if x[0] in atoms else 99, x[0]))
        }
    return result


def _merge_shell_sets(
    merged: dict[str, dict[str, set[str]]],
    irrep_sets: dict[str, dict[str, set[str]]],
    sao_labels: dict[str, list[str]],
    atoms: list[str],
) -> dict[str, dict[str, set[str]]]:
    """跨 irrep 合并壳层集合。"""
    import re as _r
    tier_keys = ("frozen_core", "outer_core", "inactive", "active", "virtual")
    for tier in tier_keys:
        for label in sao_labels.get(tier, []):
            m = _r.match(r'(\d+)([A-Z][a-z]?)(\d+)([A-Z])(-?\d+)', label)
            if m:
                element = m.group(2)
                shell = f"{m.group(3)}{m.group(4).lower()}"
                merged.setdefault(tier, {}).setdefault(element, set()).add(shell)
    return merged


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
