"""OrbitalClassifier 测试"""

import pytest
from bdf_output_parser import OrbitalClassifier, SAOParseResult, IrrepSAO, SAOLine, AOLabel


SAO_WATER_C2V = SAOParseResult(
    point_group="C(2V)", n_irreps=4, n_basis=7,
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
        """O: [He]2s2 2p4 → oc={1S}, inactive={2S}, active={2P} + H:vl=1S"""
        cls = OrbitalClassifier().classify(SAO_WATER_C2V, atoms=["O", "H", "H"])

        assert cls.n_basis == 7
        assert cls.n_electrons == 10
        assert cls.n_frozen_core_electrons == 0    # no deep core (He max_n=1)
        assert cls.n_outer_core_electrons == 2      # O 1s²
        assert cls.n_inactive_electrons == 2        # O 2s²
        assert cls.n_active_electrons == 6          # O 2p⁴ + H 1s²

    def test_shell_tiers(self):
        """验证五层级判定"""
        c = OrbitalClassifier()
        fz, oc, vl = c._get_shell_tiers("U")
        inactive, active = c._split_valence("U", vl)
        assert (5, "F") in active       # 5f³ — partial
        assert (6, "D") in active       # 6d¹ — partial
        assert (7, "S") in inactive     # 7s² — full

        fz, oc, vl = c._get_shell_tiers("F")
        inactive, active = c._split_valence("F", vl)
        assert (2, "S") in inactive     # 2s² — full
        assert (2, "P") in active       # 2p⁵ — partial

    def test_overrides(self):
        """覆盖: F 2s2p → frozen_core"""
        cls = OrbitalClassifier().classify(SAO_WATER_C2V, atoms=["O", "H", "H"],
            overrides={"frozen_core": ["H:1s"]})
        # H 1s goes from active to frozen_core
        assert cls.n_active_electrons == 4    # only O 2p⁴ left active
        assert cls.n_frozen_core_electrons == 2  # H 1s² moved to frozen_core

    def test_no_sao_data(self):
        """空 SAO 数据"""
        cls = OrbitalClassifier().classify(SAOParseResult(), atoms=["C", "O"])
        assert cls.n_electrons == 14
        assert cls.per_irrep == []
