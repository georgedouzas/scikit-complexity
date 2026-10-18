"""Study models through many runs: designs, scenarios, replicates, sensitivity analysis, and calibration."""

from ._calibration import CalibrationResult, calibrate
from ._designs import Design, Grid, LatinHypercube, RandomDesign, Rows, SobolDesign
from ._results import ExperimentResult, RunResult
from ._runner import Experiment, Task, run_task
from ._scenarios import Scenario
from ._sensitivity import SensitivityResult, sensitivity

__all__: list[str] = [
    'CalibrationResult',
    'Design',
    'Experiment',
    'ExperimentResult',
    'Grid',
    'LatinHypercube',
    'RandomDesign',
    'Rows',
    'RunResult',
    'Scenario',
    'SensitivityResult',
    'SobolDesign',
    'Task',
    'calibrate',
    'run_task',
    'sensitivity',
]
