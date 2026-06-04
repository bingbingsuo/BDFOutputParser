"""BDFOutputParser 解析测试"""

import pytest

from bdf_output_parser.models import ParseStatus, TaskType


class TestEmptyInput:
    def test_empty_string(self, parser):
        r = parser.parse("")
        assert r.status == ParseStatus.EMPTY

    def test_bogus_text(self, parser, bogus_output):
        r = parser.parse(bogus_output)
        assert r.status == ParseStatus.EMPTY


class TestEnergyExtraction:
    def test_total_energy_e_tot(self, parser, energy_output):
        r = parser.parse(energy_output)
        assert r.energies is not None
        assert r.energies.total_energy == pytest.approx(-76.12345678, rel=1e-6)

    def test_total_energy_bdf(self, parser):
        r = parser.parse("E_tot = -100.0")
        assert r.energies.total_energy == -100.0

    def test_total_energy_generic(self, parser):
        r = parser.parse("Total energy = -50.0")
        assert r.energies.total_energy == -50.0

    def test_electronic_energy(self, parser, energy_output):
        r = parser.parse(energy_output)
        assert r.energies.electronic_energy == pytest.approx(-76.54321000, rel=1e-6)

    def test_nuclear_repulsion(self, parser, energy_output):
        r = parser.parse(energy_output)
        assert r.energies.nuclear_repulsion == pytest.approx(0.41975322, rel=1e-6)

    def test_exchange(self, parser, energy_output):
        r = parser.parse(energy_output)
        assert r.energies.exchange == pytest.approx(-5.12345678, rel=1e-6)

    def test_correlation(self, parser, energy_output):
        r = parser.parse(energy_output)
        assert r.energies.correlation == pytest.approx(-0.23456789, rel=1e-6)

    def test_scf_energy(self, parser):
        r = parser.parse("SCF energy = -99.5")
        assert r.energies.scf_energy == -99.5

    def test_mp2_energy(self, parser):
        r = parser.parse("MP2 total energy = -105.5")
        assert r.energies.mp2_energy == -105.5

    def test_kinetic_energy(self, parser):
        r = parser.parse("Kinetic energy = 50.0\nPotential energy = -150.0")
        assert r.energies.kinetic_energy == 50.0
        assert r.energies.potential_energy == -150.0

    def test_scientific_notation(self, parser):
        r = parser.parse("E_tot = -1.23456789E+02")
        assert r.energies.total_energy == pytest.approx(-123.456789, rel=1e-6)


class TestGeometryExtraction:
    def test_angstrom_coordinates(self, parser, water_opt_output):
        r = parser.parse(water_opt_output)
        assert len(r.geometry.atoms) == 3
        assert r.geometry.natoms == 3
        first = r.geometry.atoms[0]
        assert first.element == "O"
        assert first.units == "angstrom"

    def test_formula(self, parser, water_opt_output):
        r = parser.parse(water_opt_output)
        assert r.geometry.formula == "H2O"

    def test_charge_multiplicity(self, parser, water_opt_output):
        r = parser.parse(water_opt_output)
        assert r.geometry.charge == 0.0
        assert r.geometry.multiplicity == 1

    def test_bohr_coordinates(self, parser, bohr_geometry_output):
        r = parser.parse(bohr_geometry_output)
        assert len(r.geometry.atoms) == 4
        assert r.geometry.atoms[0].element == "C"
        assert r.geometry.atoms[0].units == "bohr"
        # Check correct coordinate extraction
        assert r.geometry.atoms[0].x == pytest.approx(1.12766281, rel=1e-6)

    def test_no_geometry(self, parser):
        r = parser.parse("Total energy = -76.12345678")
        assert len(r.geometry.atoms) == 0


class TestFrequencyExtraction:
    def test_frequencies(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert len(r.frequencies.frequencies) == 6
        assert r.frequencies.frequencies[0] == pytest.approx(1221.1124, rel=1e-5)
        assert r.frequencies.frequencies[-1] == pytest.approx(3105.1415, rel=1e-5)

    def test_ir_intensities(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert len(r.frequencies.ir_intensities) == 6
        assert r.frequencies.ir_intensities[0] == pytest.approx(10.5879, rel=1e-5)
        assert r.frequencies.ir_intensities[-1] == pytest.approx(113.8213, rel=1e-5)

    def test_reduced_masses(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert len(r.frequencies.reduced_masses) == 6
        assert r.frequencies.reduced_masses[0] == pytest.approx(1.3546, rel=1e-5)

    def test_force_constants(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert len(r.frequencies.force_constants) == 6
        assert r.frequencies.force_constants[0] == pytest.approx(1.1900, rel=1e-5)

    def test_irreps(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert r.frequencies.irreps == ["B1", "B2", "A1", "A1", "A1", "B2"]

    def test_zpe(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert r.frequencies.zero_point_energy == pytest.approx(0.027145, rel=1e-5)

    def test_thermal_correction(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert r.frequencies.thermal_correction == pytest.approx(0.005678, rel=1e-5)

    def test_entropy(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert r.frequencies.entropy == pytest.approx(0.023456, rel=1e-5)

    def test_stable(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert r.frequencies.is_stable is True
        assert r.frequencies.n_imaginary == 0

    def test_translations_rotations(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert len(r.frequencies.translations_rotations) == 6
        assert r.frequencies.translations_rotations[0] == 0.0

    def test_legacy_freq_format(self, parser):
        output = """
Zero-point Energy = 0.012345 Hartree
Frequencies (cm-1):
   500.23  750.45  890.12
"""
        r = parser.parse(output)
        assert r.frequencies.zero_point_energy == pytest.approx(0.012345, rel=1e-5)
        # 旧格式 (cm-1) 标题行不会被当成频率值


class TestTDDFTExtraction:
    def test_excited_states(self, parser, tddft_output):
        r = parser.parse(tddft_output)
        assert len(r.excited_states) == 3

        s = r.excited_states[0]
        assert s.index == 1
        assert s.energy_ev == pytest.approx(10.4879, rel=1e-5)
        assert s.wavelength_nm == pytest.approx(118.22, rel=1e-5)
        assert s.oscillator_strength == pytest.approx(0.0922, rel=1e-5)
        assert s.symmetry == "A1"
        assert s.dominant_transition == "HOMO -> LUMO"

    def test_soc_multi_block(self, parser, tddft_soc_output):
        r = parser.parse(tddft_soc_output)
        assert len(r.tddft_blocks) == 2
        assert r.tddft_blocks[0].isf == 0
        assert r.tddft_blocks[1].isf == 1
        assert len(r.tddft_blocks[0].states) == 1
        assert len(r.tddft_blocks[1].states) == 1


class TestOptimizationExtraction:
    def test_converged(self, parser, water_opt_output):
        r = parser.parse(water_opt_output)
        assert r.optimization.converged is True
        assert r.optimization.n_steps == 3

    def test_not_converged(self, parser):
        output = """
 Geometry Optimization step : 1
 Geometry Optimization not converged
 E_tot = -76.0
"""
        r = parser.parse(output)
        assert r.optimization.n_steps == 1
        assert r.optimization.converged is False


class TestSCFExtraction:
    def test_scf_converged(self, parser, water_opt_output):
        r = parser.parse(water_opt_output)
        assert r.scf.converged is True

    def test_scf_iterations(self, parser, energy_output):
        r = parser.parse(energy_output)
        assert r.scf.n_iterations == 12

    def test_error_output(self, parser, error_output):
        r = parser.parse(error_output)
        assert r.scf.converged is False
        assert len(r.errors) > 0


class TestTaskTypeDetection:
    def test_optimization(self, parser, water_opt_output):
        r = parser.parse(water_opt_output)
        assert r.task_type == TaskType.GEOMETRY_OPT

    def test_frequency(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert r.task_type == TaskType.FREQUENCY

    def test_optfreq(self, parser, optfreq_output):
        r = parser.parse(optfreq_output)
        assert r.task_type == TaskType.OPT_FREQ

    def test_tddft(self, parser, tddft_output):
        r = parser.parse(tddft_output)
        assert r.task_type == TaskType.TDDFT

    def test_single_point(self, parser, energy_output):
        r = parser.parse(energy_output)
        assert r.task_type == TaskType.SINGLE_POINT


class TestEndToEnd:
    def test_water_opt_full(self, parser, water_opt_output):
        r = parser.parse(water_opt_output)
        assert r.status == ParseStatus.SUCCESS
        assert r.energies.total_energy is not None
        assert len(r.geometry.atoms) == 3
        assert r.optimization.converged

    def test_frequency_full(self, parser, freq_output):
        r = parser.parse(freq_output)
        assert r.status == ParseStatus.SUCCESS
        assert len(r.frequencies.frequencies) == 6
        assert r.geometry.formula == "H2O"
        assert r.frequencies.is_stable

    def test_json_output(self, parser, water_opt_output):
        r = parser.parse(water_opt_output)
        json_str = r.model_dump_json(exclude_none=True, indent=2)
        assert "success" in json_str
        assert "geometry_optimization" in json_str
        assert "total_energy" in json_str
        assert "O" in json_str
