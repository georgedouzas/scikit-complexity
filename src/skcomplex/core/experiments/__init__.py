"""Define experiments, the functions that study a system through many runs.

They are sweeps, ensembles, scenarios, sensitivity, basin stability, orbit diagrams, calibration, posterior coverage,
identifiability, virtual populations, and the calibration of individuals. Each is a function of a system, as the model
selection functions of scikit-learn are, and returns a frozen result.
"""

from ._basin_stability import BasinStabilityResult, basin_stability
from ._calibration import CalibrationResult, calibrate
from ._coverage import CoverageResult, posterior_coverage
from ._ensemble import ensemble
from ._identifiability import IdentifiabilityResult, identifiability
from ._individuals import IndividualCalibrationResult, calibrate_individuals
from ._orbit_diagram import OrbitDiagramResult, orbit_diagram
from ._results import ExperimentResult
from ._scenarios import compare_scenarios
from ._sensitivity import SensitivityResult, sensitivity
from ._sweep import sweep
from ._virtual_population import VirtualPopulationResult, virtual_population

__all__: list[str] = [
    'BasinStabilityResult',
    'CalibrationResult',
    'CoverageResult',
    'ExperimentResult',
    'IdentifiabilityResult',
    'IndividualCalibrationResult',
    'OrbitDiagramResult',
    'SensitivityResult',
    'VirtualPopulationResult',
    'basin_stability',
    'calibrate',
    'calibrate_individuals',
    'compare_scenarios',
    'ensemble',
    'identifiability',
    'orbit_diagram',
    'posterior_coverage',
    'sensitivity',
    'sweep',
    'virtual_population',
]
