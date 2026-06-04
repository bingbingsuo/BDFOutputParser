"""Markdown 报告测试"""

import pytest

from bdf_output_parser.reporters.markdown import MarkdownReporter
from bdf_output_parser import BDFOutputParser


@pytest.fixture
def reporter_zh():
    return MarkdownReporter(language="zh")

@pytest.fixture
def reporter_en():
    return MarkdownReporter(language="en")


class TestEmptyReport:
    def test_zh(self, reporter_zh):
        r = BDFOutputParser().parse("")
        md = reporter_zh.render(r)
        assert "无计算结果" in md

    def test_en(self, reporter_en):
        r = BDFOutputParser().parse("")
        md = reporter_en.render(r)
        assert "No Results" in md


class TestHeader:
    def test_zh(self, reporter_zh, water_opt_output):
        r = BDFOutputParser().parse(water_opt_output)
        md = reporter_zh.render(r)
        assert "BDF 计算结果" in md
        assert "几何优化" in md

    def test_en(self, reporter_en, water_opt_output):
        r = BDFOutputParser().parse(water_opt_output)
        md = reporter_en.render(r)
        assert "BDF Calculation Result" in md
        assert "Geometry Optimization" in md


class TestEnergySection:
    def test_zh(self, reporter_zh, energy_output):
        r = BDFOutputParser().parse(energy_output)
        md = reporter_zh.render(r)
        assert "能量" in md
        assert "-76.12345678" in md

    def test_en(self, reporter_en, energy_output):
        r = BDFOutputParser().parse(energy_output)
        md = reporter_en.render(r)
        assert "Energy" in md
        assert "-76.12345678" in md


class TestSCFSection:
    def test_zh(self, reporter_zh, energy_output):
        r = BDFOutputParser().parse(energy_output)
        md = reporter_zh.render(r)
        assert "SCF 收敛" in md

    def test_en(self, reporter_en, energy_output):
        r = BDFOutputParser().parse(energy_output)
        md = reporter_en.render(r)
        assert "SCF Convergence" in md


class TestGeometrySection:
    def test_zh(self, reporter_zh, water_opt_result):
        md = reporter_zh.render(water_opt_result)
        assert "几何结构" in md
        assert "O" in md
        assert "H2O" in md

    def test_en(self, reporter_en, water_opt_result):
        md = reporter_en.render(water_opt_result)
        assert "Geometry" in md


class TestFrequencySection:
    def test_zh(self, reporter_zh, freq_result):
        md = reporter_zh.render(freq_result)
        assert "频率分析" in md
        assert "1221" in md
        assert "无虚频" in md

    def test_en(self, reporter_en, freq_result):
        md = reporter_en.render(freq_result)
        assert "Frequency Analysis" in md
        assert "Stable minimum" in md

    def test_imaginary_warning(self, reporter_zh, parser):
        r = parser.parse("""
Zero-point Energy = 0.01 Hartree
Results of vibrations:
     Frequencies      -200.0000     500.0000    1200.0000
""")
        md = reporter_zh.render(r)
        assert "虚频" in md


class TestTDDFTSection:
    def test_zh(self, reporter_zh, tddft_output):
        r = BDFOutputParser().parse(tddft_output)
        md = reporter_zh.render(r)
        assert "TDDFT 激发态" in md
        assert "118.22" in md

    def test_en(self, reporter_en, tddft_output):
        r = BDFOutputParser().parse(tddft_output)
        md = reporter_en.render(r)
        assert "TDDFT Excited States" in md

    def test_soc_multi_block(self, reporter_zh, tddft_soc_output):
        r = BDFOutputParser().parse(tddft_soc_output)
        md = reporter_zh.render(r)
        assert "TDDFT 激发态" in md
        assert "isf=0" in md
        assert "isf=1" in md


class TestOptimizationSection:
    def test_zh(self, reporter_zh, water_opt_result):
        md = reporter_zh.render(water_opt_result)
        assert "几何优化" in md
        assert "已收敛" in md

    def test_en(self, reporter_en, water_opt_result):
        md = reporter_en.render(water_opt_result)
        assert "Geometry Optimization" in md
        assert "Converged" in md


class TestWarnings:
    def test_error(self, reporter_zh, error_output):
        r = BDFOutputParser().parse(error_output)
        md = reporter_zh.render(r)
        assert "警告" in md or "错误" in md
