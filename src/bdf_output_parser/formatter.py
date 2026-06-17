"""SAO 轨道分类格式化输出。"""

from __future__ import annotations

import re
from typing import Any


def format_sao_classification(cls: Any, sao: Any) -> str:
    """格式化 OrbitalClassification 和 SAO 数据为可读文本。

    格式规则：
    - 单原子轨道: 7 字符宽左对齐，每行 8 个
    - 成键/反键组合(多中心): 14 字符宽左对齐，每行 4 个

    Args:
        cls: OrbitalClassification
        sao: SAOParseResult
    """
    # 计算每个 SAO 的 tier 标签
    _TIER_PRIO = {"active": 0, "inactive": 1, "outer_core": 2, "frozen_core": 3, "virtual": 4}

    lines = []
    tier_names = [
        ("frozen_core", "frozen_core"),
        ("outer_core", "outer_core"),
        ("active", "active"),
    ]

    for tier_key, tier_attr in tier_names:
        n_e = getattr(cls, f"n_{tier_attr}_electrons", 0)
        if n_e == 0 and tier_key == "inactive":
            continue
        lines.append(f"\n### {tier_key} ({n_e} e⁻)")

        for ir_cls, ir_sao in zip(cls.per_irrep, sao.irreps):
            n_orb = getattr(ir_cls, f"n_{tier_attr}_orbitals", 0)
            if n_orb == 0:
                continue

            sao_labels = _get_tier_sao_labels(ir_sao, ir_cls, tier_key, _TIER_PRIO)

            lines.append(f"\nIrrep: {ir_cls.irrep}  ({n_orb} SAOs)")

            single = [l for l in sao_labels if "+" not in l]
            multi = [l for l in sao_labels if "+" in l]

            if single:
                lines.append(_format_group(single, width=7, columns=8))
            if multi:
                lines.append(_format_group(multi, width=14, columns=4))

    return "\n".join(lines)


def _get_tier_sao_labels(
    ir_sao: Any, ir_cls: Any, tier_key: str, tier_prio: dict[str, int],
) -> list[str]:
    """返回某个 irrep 中属于指定 tier 的所有 SAO 的简短标签。"""
    result = []
    for sline in ir_sao.saos:
        # 判断这个 SAO 的 tier
        best_tier = "virtual"
        best_p = 4
        for ao in sline.aos:
            labels = {
                "frozen_core": set(l for l in ir_cls.frozen_core_labels
                                   if l.startswith(f"{ao.atom_index}{ao.element}{ao.n}{ao.l}")),
                "outer_core": set(l for l in ir_cls.outer_core_labels
                                  if l.startswith(f"{ao.atom_index}{ao.element}{ao.n}{ao.l}")),
                "active": set(l for l in ir_cls.active_labels
                              if l.startswith(f"{ao.atom_index}{ao.element}{ao.n}{ao.l}")),
                "inactive": set(l for l in ir_cls.inactive_labels
                                if l.startswith(f"{ao.atom_index}{ao.element}{ao.n}{ao.l}")),
            }
            for t in ["active", "inactive", "outer_core", "frozen_core"]:
                if labels[t]:
                    p = tier_prio.get(t, 4)
                    if p < best_p:
                        best_p = p
                        best_tier = t
                    break

        if best_tier == tier_key:
            label = _sao_label(sline)
            if label:
                result.append(label)

    return result


def _sao_label(sline: Any) -> str:
    """构建 SAO 的简短标签。

    单中心: "U5f0"
    双中心同元素: "F2p1+F2p1"
    双中心异元素: "U5f0+F2p1"
    """
    # Group AOs by element
    parts: list[str] = []
    has_same_elem_multi = False
    elements = sorted(set(ao.element for ao in sline.aos))
    if len(sline.aos) > 1 and len(elements) == 1:
        has_same_elem_multi = True

    if has_same_elem_multi:
        # 同元素多中心：F2s0+F2s0（不区原子序号，对称等价）
        parts = []
        for ao in sline.aos:
            parts.append(f"{ao.element}{ao.n}{ao.l.lower()}{ao.m}")
        return "+".join(parts)
    else:
        # 单中心或其他情况
        ao = sline.aos[0]
        return f"{ao.element}{ao.n}{ao.l.lower()}{ao.m}"


def _format_group(labels: list[str], width: int, columns: int) -> str:
    rows = []
    for i in range(0, len(labels), columns):
        chunk = labels[i : i + columns]
        rows.append("  " + "".join(f"{l:<{width}}" for l in chunk).rstrip())
    return "\n".join(rows)
