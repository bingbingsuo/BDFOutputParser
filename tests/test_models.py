"""模型验证测试"""

import pytest

from bdf_output_parser.models import (
    Atom,
    BDFParseResult,
    EnergyData,
    ExcitedState,
    FrequencyData,
    GeometryData,
    OptimizationData,
    ParseStatus,
    SCFData,
    TaskType,
    TDDFTBlock,
)


class TestAtom:
    def test_create(self):
        a = Atom(element="C", x=1.0, y=2.0, z=3.0)
        assert a.element == "C"
        assert a.x == 1.0
        assert a.y == 2.0
        assert a.z == 3.0
        assert a.units == "bohr"

    def test_angstrom(self):
        a = Atom(element="O", x=0.0, y=0.0, z=0.0, units="angstrom")
        assert a.units == "angstrom"


class TestEnergyData:
    def test_defaults(self):
        e = EnergyData()
        assert e.total_energy is None
        assert e.scf_energy is None

    def test_from_values(self):
        e = EnergyData(total_energy=-76.12, electronic_energy=-76.54)
        assert e.total_energy == -76.12
        assert e.electronic_energy == -76.54


class TestGeometryData:
    def test_defaults(self):
        g = GeometryData()
        assert g.atoms == []
        assert g.natoms == 0

    def test_with_atoms(self):
        g = GeometryData(atoms=[
            Atom(element="O", x=0, y=0, z=0),
            Atom(element="H", x=0, y=0.76, z=-0.47),
            Atom(element="H", x=0, y=-0.76, z=-0.47),
        ])
        assert g.natoms == 3
        assert g.formula == "H2O"


class TestFrequencyData:
    def test_stable(self):
        f = FrequencyData(frequencies=[500.0, 600.0, 1200.0])
        assert f.is_stable is True
        assert f.n_imaginary == 0
        assert f.imaginary_frequencies == []

    def test_imaginary(self):
        f = FrequencyData(frequencies=[-200.0, 500.0, 1200.0])
        assert f.is_stable is False
        assert f.n_imaginary == 1
        assert f.imaginary_frequencies == [-200.0]


class TestExcitedState:
    def test_create(self):
        s = ExcitedState(index=1, energy_ev=4.23, wavelength_nm=292.34, oscillator_strength=0.1234)
        assert s.index == 1
        assert s.energy_ev == 4.23


class TestTDDFTBlock:
    def test_default(self):
        b = TDDFTBlock()
        assert b.states == []
        assert b.isf is None

    def test_with_states(self):
        s1 = ExcitedState(index=1, energy_ev=4.0, wavelength_nm=310.0, oscillator_strength=0.1)
        b = TDDFTBlock(isf=0, states=[s1])
        assert b.isf == 0
        assert len(b.states) == 1


class TestOptimizationData:
    def test_default(self):
        o = OptimizationData()
        assert o.converged is False
        assert o.n_steps == 0


class TestSCFData:
    def test_default(self):
        s = SCFData()
        assert s.converged is False
        assert s.n_iterations == 0


class TestBDFParseResult:
    def test_empty(self):
        r = BDFParseResult()
        assert r.status == ParseStatus.EMPTY
        assert r.is_success is False

    def test_excited_states_flat(self):
        s1 = ExcitedState(index=1, energy_ev=4.0, wavelength_nm=310.0, oscillator_strength=0.1)
        s2 = ExcitedState(index=2, energy_ev=5.0, wavelength_nm=248.0, oscillator_strength=0.2)
        b1 = TDDFTBlock(isf=0, states=[s1])
        b2 = TDDFTBlock(isf=1, states=[s2])
        r = BDFParseResult(
            status=ParseStatus.SUCCESS,
            tddft_blocks=[b1, b2],
        )
        assert len(r.excited_states) == 2
        assert r.excited_states[0].index == 1
        assert r.excited_states[1].index == 2

    def test_json_serialization(self):
        r = BDFParseResult(
            status=ParseStatus.SUCCESS,
            task_type=TaskType.SINGLE_POINT,
            energies=EnergyData(total_energy=-76.12),
            geometry=GeometryData(atoms=[Atom(element="O", x=0, y=0, z=0)]),
        )
        d = r.model_dump(exclude_none=True)
        assert d["status"] == "success"
        assert d["task_type"] == "single_point"
        assert d["energies"]["total_energy"] == -76.12
        assert len(d["geometry"]["atoms"]) == 1

        # JSON 可序列化
        json_str = r.model_dump_json(exclude_none=True)
        assert "success" in json_str
        assert "-76.12" in json_str


class TestTaskType:
    def test_values(self):
        assert TaskType.SINGLE_POINT.value == "single_point"
        assert TaskType.GEOMETRY_OPT.value == "geometry_optimization"
        assert TaskType.FREQUENCY.value == "frequency"
        assert TaskType.TDDFT.value == "tddft"
