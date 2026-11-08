"""Study systems through many runs.

They are sweeps, ensembles, scenarios, sensitivity, basin stability, and calibration. The experiments are functions of a
system, in the role of the model selection functions of scikit-learn, and return frozen results. Calibration returns
the fitted system in its result.
"""

from ._basin_stability import BasinStabilityResult, basin_stability
from ._calibration import CalibrationResult, calibrate
from ._ensemble import ensemble
from ._orbit_diagram import OrbitDiagramResult, orbit_diagram
from ._results import ExperimentResult
from ._scenarios import compare_scenarios
from ._sensitivity import SensitivityResult, sensitivity
from ._sweep import sweep

__all__: list[str] = [
    'BasinStabilityResult',
    'CalibrationResult',
    'ExperimentResult',
    'OrbitDiagramResult',
    'SensitivityResult',
    'basin_stability',
    'calibrate',
    'compare_scenarios',
    'ensemble',
    'orbit_diagram',
    'sensitivity',
    'sweep',
]
