"""BDFOutputParser — 共享测试 fixtures

提供各类 BDF 输出文本样本，覆盖所有主要计算类型。
"""

import pytest

from bdf_output_parser import BDFOutputParser


# =============================================================================
# BDF 输出样本
# =============================================================================

@pytest.fixture
def water_opt_output():
    """水分子几何优化"""
    return """
 BDF (Beijing Density Functional)  Version 2024.01
 ============================================================

 Molecular Cartesian Coordinates (X,Y,Z) in Angstrom :
  No.  Element    X (Angstrom)       Y (Angstrom)       Z (Angstrom)
 ------------------------------------------------------------------------
    1       O        0.00000000          0.00000000          0.11730000
    2       H        0.00000000          0.75720000         -0.46920000
    3       H        0.00000000         -0.75720000         -0.46920000

 Geometry Optimization step : 1
 Geometry Optimization step : 2
 Geometry Optimization step : 3

 Molecular Cartesian Coordinates (X,Y,Z) in Angstrom :
  No.  Element    X (Angstrom)       Y (Angstrom)       Z (Angstrom)
 ------------------------------------------------------------------------
    1       O        0.00000000          0.00000000          0.11680000
    2       H        0.00000000          0.76150000         -0.46710000
    3       H        0.00000000         -0.76150000         -0.46710000

 Good Job, Geometry Optimization converged!

 Total energy = -76.56789012
 E_tot = -76.56789012

 Charge = 0
 Multiplicity = 1

 diis/vshift is closed at iter = 8
 Final scf result

 Congratulations! BDF normal termination
"""


@pytest.fixture
def water_opt_result(water_opt_output):
    """解析后的优化结果"""
    return BDFOutputParser().parse(water_opt_output)


@pytest.fixture
def energy_output():
    """单点能计算"""
    return """
 BDF (Beijing Density Functional)  Version 2024.01
 ============================================================

 E_tot = -76.12345678
 E_ele = -76.54321000
 E_nn = 0.41975322
 Exchange = -5.12345678
 Correlation = -0.23456789

 Charge = 0
 Multiplicity = 1

 diis/vshift is closed at iter = 12
 Final scf result

 Total energy = -76.12345678

 Congratulations! BDF normal termination
"""


@pytest.fixture
def freq_output():
    """频率计算（真实 BDF 格式）"""
    return """
 BDF Version 2024.01
 ============================================================

 Zero-point Energy = 0.027145 Hartree
 Thermal Correction = 0.005678 Hartree
 Entropy = 0.023456 Hartree/K

 Molecular Cartesian Coordinates (X,Y,Z) in Angstrom :
  No.  Element    X (Angstrom)       Y (Angstrom)       Z (Angstrom)
 ------------------------------------------------------------------------
    1       O        0.00000000          0.00000000          0.11730000
    2       H        0.00000000          0.75720000         -0.46920000
    3       H        0.00000000         -0.75720000         -0.46920000

 Results of vibrations:

 Normal frequencies (cm^-1), reduced masses (AMU), force constants (mDyn/A), IR intensities (km/mol)

                                                   1                                 2                                 3
          Irreps                                  B1                                B2                                A1
     Frequencies                           1221.1124                         1274.7794                         1535.9150
  Reduced masses                              1.3546                            1.3253                            1.4549
 Force constants                              1.1900                            1.2689                            2.0221
  IR intensities                             10.5879                           18.3339                           70.1071

                                                   4                                 5                                 6
          Irreps                                  A1                                A1                                B2
     Frequencies                           1698.5232                         3016.2873                         3105.1415
  Reduced masses                              2.8044                            1.0480                            1.1189
 Force constants                              4.7669                            5.6177                            6.3560
  IR intensities                             98.2889                           35.0447                          113.8213

 Results of translations and rotations:
     Frequencies      0.0000      0.0000      0.0000      0.0000      0.0000      0.0000

 E_tot = -76.56789012
 Total energy = -76.56789012
 Charge = 0
 Multiplicity = 1

 Congratulations! BDF normal termination
"""


@pytest.fixture
def freq_result(freq_output):
    """解析后的频率计算结果"""
    return BDFOutputParser().parse(freq_output)


@pytest.fixture
def tddft_output():
    """TDDFT 计算"""
    return """
 BDF Version 2024.01
 ============================================================

 Spin change :

 No. Pair   ExSym   ExEnergies     Wavelengths      f       dS2     Dominant
   1   A1    2  A1   10.4879 eV        118.22 nm   0.0922   0.0000   HOMO -> LUMO
   2   B1    3  A1   12.3456 eV        100.45 nm   0.1456   0.0000   HOMO-1 -> LUMO
   3   A1    4  A1   14.7890 eV         83.90 nm   0.2013   0.0000   HOMO -> LUMO+1

 E_tot = -76.12345678
 Total energy = -76.12345678
"""


@pytest.fixture
def tddft_soc_output():
    """TDDFT+SOC 多 block"""
    return """
 BDF Version 2024.01
 ============================================================

 isf = 0

 Spin change :

 No. Pair   ExSym   ExEnergies     Wavelengths      f       dS2     Dominant
   1   A1    2  A1    4.2345 eV        292.34 nm   0.1234   0.0000   HOMO -> LUMO

 isf = 1

 Spin change :

 No. Pair   ExSym   ExEnergies     Wavelengths      f       dS2     Dominant
   1   B1    2  B1    2.1000 eV        590.47 nm   0.0000   0.0000   HOMO -> LUMO

 Congratulations! BDF normal termination
"""


@pytest.fixture
def bogus_output():
    """空/无效输出"""
    return ""


@pytest.fixture
def error_output():
    """计算错误输出"""
    return """
 BDF Version 2024.01

 ERROR: SCF did not converge after 50 iterations
 FATAL error: Calculation aborted

 Total energy = -76.12345678
"""


@pytest.fixture
def bohr_geometry_output():
    """Bohr 坐标系几何"""
    return """
 Atom  Cartcoord(Bohr)

 Atom   Charge   Basis
 ------------------------------------------------------------------------
   C        6.00     1.12766281      -0.06079459       1.22640622
   N        7.00    -0.48840199       0.98939510      -0.44852350
   H        1.00     2.09856789       0.11890456       1.00345678
   H        1.00     0.75689012      -0.84056789       1.88901234

 E_tot = -210.12345678
"""


@pytest.fixture
def optfreq_output():
    """优化+频率"""
    return """
 BDF Version 2024.01

 Geometry Optimization step : 1
 Geometry Optimization step : 2

 Molecular Cartesian Coordinates (X,Y,Z) in Angstrom :
  No.  Element    X (Angstrom)       Y (Angstrom)       Z (Angstrom)
 ------------------------------------------------------------------------
    1       C        0.00000000          0.00000000          0.00000000
    2       O        0.00000000          0.00000000          1.20000000

 Good Job, Geometry Optimization converged!

 Zero-point Energy = 0.010000 Hartree

 Results of vibrations:
     Frequencies      500.0000    1100.0000    2100.0000

 E_tot = -113.45678901
 Charge = 0
 Multiplicity = 1

 Congratulations! BDF normal termination
"""


# =============================================================================
# Parser fixture
# =============================================================================

@pytest.fixture
def parser():
    return BDFOutputParser()
