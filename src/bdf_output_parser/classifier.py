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
    ) -> OrbitalClassification:
        """
        分类每个不可约表示中的芯层和活性轨道。

        Args:
            sao: checksymm 输出的 SAO 解析结果
            atoms: 分子中各原子的元素符号列表 ["O", "H", "H"]

        Returns:
            OrbitalClassification，含每 irrep 的 core/active 轨道数
        """
        # Step 1: 构建每个原子的 core/valence (n, l) 集合
        atom_core: dict[int, set[tuple[int, str]]] = {}    # atom_idx → {(n, L), ...}
        atom_valence: dict[int, set[tuple[int, str]]] = {}  # atom_idx → {(n, L), ...}

        for idx, sym in enumerate(atoms, start=1):
            core, valence = self._get_core_valence_shells(sym)
            atom_core[idx] = core
            atom_valence[idx] = valence

        # Step 2: 汇总电子数
        n_core_e = 0
        n_active_e = 0
        for sym in atoms:
            ele = ELEMENTS[sym]
            core_shells, _ = self._get_core_valence_shells(sym)
            config = ele.eleconfig_dict
            for (n, letter), occ in config.items():
                l_letter = letter.upper()
                if (n, l_letter) in core_shells:
                    n_core_e += occ
                else:
                    n_active_e += occ

        # Step 3: 对每个 irrep 分类 SAO
        per_irrep: list[IrrepClassification] = []
        for irrep_data in sao.irreps:
            core_labels: list[str] = []
            active_labels: list[str] = []

            for sao_line in irrep_data.saos:
                for ao in sao_line.aos:
                    label = f"{ao.atom_index}{ao.element}{ao.n}{ao.l}{ao.m}"
                    core_shells = atom_core.get(ao.atom_index, set())
                    if (ao.n, ao.l) in core_shells:
                        core_labels.append(label)
                    else:
                        active_labels.append(label)

            per_irrep.append(IrrepClassification(
                irrep=irrep_data.irrep,
                norb=irrep_data.norb,
                core_ao_labels=core_labels,
                active_ao_labels=active_labels,
                n_core=len(core_labels),
                n_active=len(active_labels),
            ))

        return OrbitalClassification(
            molecule="".join(atoms),
            point_group=sao.point_group or "",
            n_electrons=n_core_e + n_active_e,
            n_core_electrons=n_core_e,
            n_active_electrons=n_active_e,
            per_irrep=per_irrep,
        )

    @staticmethod
    def _get_core_valence_shells(
        symbol: str,
    ) -> tuple[set[tuple[int, str]], set[tuple[int, str]]]:
        """返回原子的 (core_shells, valence_shells)。

        core_shells: 惰性气体芯对应的 (n, L_letter) 集合
        valence_shells: 芯层之外所有占据的 (n, L_letter) 集合

        Example:
            O: [He] 2s2 2p4 → core={(1,'S')}, valence={(2,'S'), (2,'P')}
        """
        ele = ELEMENTS[symbol]
        config = ele.eleconfig  # e.g. "[He] 2s2 2p4"

        # 找到惰性气体芯
        core_config: dict[tuple[int, str], int] = {}
        if config.startswith("["):
            bracket_end = config.index("]")
            noble_gas = config[1:bracket_end]
            if noble_gas in ELEMENTS:
                core_config = dict(ELEMENTS[noble_gas].eleconfig_dict)

        # 芯层 (n, L_letter) 集合
        core_shells = {(n, letter.upper()) for (n, letter) in core_config}

        # 所有占据的 (n, L_letter) 集合
        all_config = ele.eleconfig_dict
        all_shells = {(n, letter.upper()) for (n, letter) in all_config}

        # 价层 = 所有占据 - 芯层
        valence_shells = all_shells - core_shells

        return core_shells, valence_shells


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
