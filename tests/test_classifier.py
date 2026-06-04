"""OrbitalClassifier 测试"""

import pytest
from bdf_output_parser import OrbitalClassifier, SAOParseResult, IrrepSAO, SAOLine, AOLabel


SAO_WATER_C2V = SAOParseResult(
    point_group="C(2V)",
    n_irreps=4,
    n_basis=7,
    irreps=[
        IrrepSAO(irrep="A1", irrep_index=1, norb=4, saos=[
            SAOLine(label="A1|1C1", irrep="A1", irrep_index=1, component=1,
                    aos=[AOLabel(atom_index=1, element="O", n=1, l="S", m=0, coeff=1.0)]),
            SAOLine(label="A1|2C1", irrep="A1", irrep_index=1, component=1,
                    aos=[AOLabel(atom_index=1, element="O", n=2, l="S", m=0, coeff=1.0)]),
            SAOLine(label="A1|3C1", irrep="A1", irrep_index=1, component=1,
                    aos=[AOLabel(atom_index=1, element="O", n=2, l="P", m=0, coeff=1.0)]),
            SAOLine(label="A1|4C1", irrep="A1", irrep_index=1, component=1,
                    aos=[AOLabel(atom_index=2, element="H", n=1, l="S", m=0, coeff=0.7071),
                         AOLabel(atom_index=3, element="H", n=1, l="S", m=0, coeff=0.7071)]),
        ]),
        IrrepSAO(irrep="A2", irrep_index=2, norb=0, saos=[]),
        IrrepSAO(irrep="B1", irrep_index=3, norb=2, saos=[
            SAOLine(label="B1|1C1", irrep="B1", irrep_index=3, component=1,
                    aos=[AOLabel(atom_index=1, element="O", n=2, l="P", m=1, coeff=1.0)]),
            SAOLine(label="B1|2C1", irrep="B1", irrep_index=3, component=1,
                    aos=[AOLabel(atom_index=2, element="H", n=1, l="S", m=0, coeff=0.7071),
                         AOLabel(atom_index=3, element="H", n=1, l="S", m=0, coeff=-0.7071)]),
        ]),
        IrrepSAO(irrep="B2", irrep_index=4, norb=1, saos=[
            SAOLine(label="B2|1C1", irrep="B2", irrep_index=4, component=1,
                    aos=[AOLabel(atom_index=1, element="O", n=2, l="P", m=-1, coeff=1.0)]),
        ]),
    ],
)


class TestOrbitalClassifier:
    def test_water_c2v(self):
        classifier = OrbitalClassifier()
        cls = classifier.classify(SAO_WATER_C2V, atoms=["O", "H", "H"])

        assert cls.n_electrons == 10
        assert cls.n_outer_core_electrons == 2  # O 1s² (He max_n=1, frozen=∅)
        assert cls.n_valence_electrons == 8      # O 2s² 2p⁴ + H 1s²
        assert cls.point_group == "C(2V)"

        a1 = cls.per_irrep[0]
        assert a1.irrep == "A1"
        assert a1.norb == 4
        assert a1.n_outer_core == 1     # O1S0
        assert a1.n_valence == 4

        b1 = cls.per_irrep[2]
        assert b1.irrep == "B1"
        assert b1.n_outer_core == 0
        assert b1.n_valence == 3    # O2P1 + 2×H1S0

        b2 = cls.per_irrep[3]
        assert b2.irrep == "B2"
        assert b2.n_outer_core == 0
        assert b2.n_valence == 1    # O2P-1

    def test_shell_tiers(self):
        """验证三层级判定逻辑"""
        classifier = OrbitalClassifier()
        fz, oc, vl = classifier._get_shell_tiers("C")
        assert fz == set()                            # He max_n=1, no n<1
        assert oc == {(1, "S")}                       # n=1 in [He]
        assert vl == {(2, "S"), (2, "P")}             # 2s2 2p2

        fz, oc, vl = classifier._get_shell_tiers("H")
        assert fz == set() and oc == set()            # no core
        assert vl == {(1, "S")}                       # 1s1

        fz, oc, vl = classifier._get_shell_tiers("Fe")
        assert (1, "S") in fz                         # n<3 in [Ar]
        assert (3, "S") in oc                         # n=3 in [Ar]
        assert (3, "D") in vl                         # 3d6 active

        fz, oc, vl = classifier._get_shell_tiers("U")
        assert (4, "F") in fz                         # n<6 in [Rn]
        assert (6, "S") in oc                         # n=6 in [Rn]
        assert (5, "F") in vl                         # 5f3 valence

    def test_no_sao_data(self):
        """空 SAO 数据的情况 — 仍然能统计电子数"""
        classifier = OrbitalClassifier()
        cls = classifier.classify(
            SAOParseResult(),
            atoms=["C", "O"],
        )
        assert cls.n_electrons == 14  # C(6) + O(8)
        assert cls.per_irrep == []
