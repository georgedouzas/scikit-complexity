"""Define experiments, the functions that study a system through many runs.

They are sweeps, ensembles, scenarios, sensitivity, basin stability, orbit diagrams, calibration, posterior coverage,
and identifiability. Each is a function of a system, as the model selection functions of scikit-learn are, and returns a
frozen result.
"""

from ._basin_stability import BasinStabilityResult, basin_stability
from ._calibration import CalibrationResult, calibrate
from ._coverage import CoverageResult, posterior_coverage
from ._ensemble import ensemble
from ._identifiability import IdentifiabilityResult, identifiability
from ._orbit_diagram import OrbitDiagramResult, orbit_diagram
from ._results import ExperimentResult
from ._scenarios import compare_scenarios
from ._sensitivity import SensitivityResult, sensitivity
from ._sweep import sweep

__all__: list[str] = [
    'BasinStabilityResult',
    'CalibrationResult',
    'CoverageResult',
    'ExperimentResult',
    'IdentifiabilityResult',
    'OrbitDiagramResult',
    'SensitivityResult',
    'basin_stability',
    'calibrate',
    'compare_scenarios',
    'ensemble',
    'identifiability',
    'orbit_diagram',
    'posterior_coverage',
    'sensitivity',
    'sweep',
]
