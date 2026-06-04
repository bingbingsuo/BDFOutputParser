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
        assert cls.n_core_electrons == 2       # O 1s²
        assert cls.n_active_electrons == 8     # O 2s² 2p⁴ + H 1s²
        assert cls.point_group == "C(2V)"

        a1 = cls.per_irrep[0]
        assert a1.irrep == "A1"
        assert a1.norb == 4
        assert a1.n_core == 1     # O1S0
        assert a1.n_active == 4

        b1 = cls.per_irrep[2]
        assert b1.irrep == "B1"
        assert b1.n_core == 0
        assert b1.n_active == 3    # O2P1 + 2×H1S0

        b2 = cls.per_irrep[3]
        assert b2.irrep == "B2"
        assert b2.n_core == 0
        assert b2.n_active == 1    # O2P-1

    def test_core_shells(self):
        """验证芯层判定逻辑"""
        classifier = OrbitalClassifier()
        core, valence = classifier._get_core_valence_shells("C")
        assert core == {(1, "S")}                     # [He] core
        assert valence == {(2, "S"), (2, "P")}        # 2s2 2p2

        core, valence = classifier._get_core_valence_shells("H")
        assert core == set()                           # no core
        assert valence == {(1, "S")}                   # 1s1

        core, valence = classifier._get_core_valence_shells("Fe")
        assert (1, "S") in core                        # [Ar] core
        assert (3, "D") in valence                     # 3d6 active

    def test_no_sao_data(self):
        """空 SAO 数据的情况"""
        classifier = OrbitalClassifier()
        cls = classifier.classify(
            SAOParseResult(),
            atoms=["C", "O"],
        )
        assert cls.n_electrons > 0
        assert cls.per_irrep == []
