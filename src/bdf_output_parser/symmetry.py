"""
BDF Output Parser — 分子对称性映射表

BDF 多组态方法（MCSCF/MRCI）仅支持阿贝尔群（D2h 及其子群）。
当 checksymm 检测到非阿贝尔点群时，需要推荐最大阿贝尔子群计算 SAO。

参考: Altmann & Herzig, Point-Group Theory Tables
"""

from __future__ import annotations

# =============================================================================
# 阿贝尔群列表（MCSCF/MRCI 支持）
# =============================================================================

ABELIAN_GROUPS: set[str] = {
    "C1", "C2", "CS", "CI",
    "C2V", "C2H", "D2", "D2H",
}

# =============================================================================
# 非阿贝尔群 → 最大阿贝尔子群
# =============================================================================

SUBGROUP_MAP: dict[str, str] = {
    # 线性分子
    "D(INF)": "D2H",    # D∞h
    "C(INF)": "C2V",    # C∞v
    "D(LIN)": "D2H",    # D∞h (BDF 输出格式)
    "C(LIN)": "C2V",    # C∞v (BDF 输出格式)

    # 立方群
    "T":  "D2",
    "TH": "D2H",
    "TD": "D2D",
    "O":  "D2H",
    "OH": "D2H",

    # 二十面体群
    "I":  "D2",
    "IH": "D2H",

    # Cnv (n≥3)
    "C3V": "CS",
    "C4V": "C2V",
    "C5V": "CS",
    "C6V": "C2V",

    # Cnh (n≥3)
    "C3H": "C2H",
    "C4H": "C2H",
    "C5H": "C2H",
    "C6H": "C2H",

    # Dn (n≥3)
    "D3": "C2",
    "D4": "D2",
    "D5": "C2",
    "D6": "D2",

    # Dnd (n≥3)
    "D3D": "C2H",
    "D4D": "D2D",
    "D5D": "C2H",
    "D6D": "D2D",

    # Dnh (n≥3)
    "D3H": "C2V",
    "D4H": "D2H",
    "D5H": "C2V",
    "D6H": "D2H",

    # Sn (n>2)
    "S4": "C2",
    "S6": "C2",
    "S8": "C2",
}

# =============================================================================
# irrep 相关性表
# 非阿贝尔 irrep → 最大阿贝尔子群 irreps
# =============================================================================

# D∞h → D2h
IRREP_CORRELATION_DINFH_TO_D2H: dict[str, list[str]] = {
    # g (gerade) parity
    "A1G": ["AG"],              # Σg+
    "A2G": ["B1G"],             # Σg-
    "E1G": ["B2G", "B3G"],      # Πg  → B2g ⊕ B3g
    "E2G": ["AG", "B1G"],       # Δg  → Ag ⊕ B1g
    "E3G": ["B2G", "B3G"],      # Φg  → B2g ⊕ B3g
    "E4G": ["AG", "B1G"],       # Γg  → Ag ⊕ B1g
    "E5G": ["B2G", "B3G"],
    "E6G": ["AG", "B1G"],
    "E7G": ["B2G", "B3G"],
    "E8G": ["AG", "B1G"],
    "E9G": ["B2G", "B3G"],
    # u (ungerade) parity
    "A1U": ["AU"],              # Σu+
    "A2U": ["B1U"],             # Σu-
    "E1U": ["B2U", "B3U"],      # Πu  → B2u ⊕ B3u
    "E2U": ["AU", "B1U"],       # Δu  → Au ⊕ B1u
    "E3U": ["B2U", "B3U"],      # Φu  → B2u ⊕ B3u
    "E4U": ["AU", "B1U"],
    "E5U": ["B2U", "B3U"],
    "E6U": ["AU", "B1U"],
    "E7U": ["B2U", "B3U"],
    "E8U": ["AU", "B1U"],
    "E9U": ["B2U", "B3U"],
}

# Oh → D2h
IRREP_CORRELATION_OH_TO_D2H: dict[str, list[str]] = {
    "A1G": ["AG"],
    "A2G": ["B1G"],
    "EG":  ["AG", "B1G"],
    "T1G": ["B1G", "B2G", "B3G"],
    "T2G": ["AG", "B2G", "B3G"],
    "A1U": ["AU"],
    "A2U": ["B1U"],
    "EU":  ["AU", "B1U"],
    "T1U": ["B1U", "B2U", "B3U"],
    "T2U": ["AU", "B2U", "B3U"],
}

# 按主要群查找相关性表
_IRREP_CORRELATION_TABLES: dict[str, dict[str, list[str]]] = {
    "D(INF)": IRREP_CORRELATION_DINFH_TO_D2H,
    "D(LIN)": IRREP_CORRELATION_DINFH_TO_D2H,
    "C(INF)": IRREP_CORRELATION_DINFH_TO_D2H,  # C∞v ⊂ D∞h
    "C(LIN)": IRREP_CORRELATION_DINFH_TO_D2H,
    "OH": IRREP_CORRELATION_OH_TO_D2H,
    "O": IRREP_CORRELATION_OH_TO_D2H,
}


def is_abelian(point_group: str) -> bool:
    """检查点群是否为阿贝尔群（MCSCF/MRCI 支持）。"""
    return point_group.upper() in ABELIAN_GROUPS


def recommend_subgroup(point_group: str) -> str | None:
    """若非阿贝尔群，返回推荐的最大阿贝尔子群名称。"""
    key = point_group.upper()
    return SUBGROUP_MAP.get(key)


def get_irrep_correlation(point_group: str) -> dict[str, list[str]]:
    """返回非阿贝尔 irrep → 子群 irrep 的相关性表。"""
    key = point_group.upper()
    return _IRREP_CORRELATION_TABLES.get(key, {})
