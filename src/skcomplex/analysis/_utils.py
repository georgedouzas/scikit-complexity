"""Share the input checks of the analyses: real values, sample columns, kinds with their settings, and systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError, ValidationError
from ..core.system import System
from ..engine.problems import BaseProblem
from ..engine.runtime import Trajectory


def system_of(source: object) -> System | BaseProblem:
    """Return the system or the problem of a source: the problem of a trajectory, and a system or a problem as it is.

    Args:
        source:
            A system, a problem compiled from one, or a trajectory.

    Returns:
        system:
            The system, or the problem.

    Raises:
        AnalysisError:
            The source is not a system, a problem, or a trajectory.
    """
    if isinstance(source, Trajectory):
        return source.problem
    if isinstance(source, System | BaseProblem):
        return source
    error_msg = (
        f'The analysis reads a system, a problem, or a trajectory, not {type(source).__name__}. Pass one of them.'
    )
    raise AnalysisError(error_msg)


def real_array(values: npt.ArrayLike, argument: str) -> npt.NDArray[np.float64]:
    """Return values as a float array after rejecting complex values, whose imaginary part a cast would drop.

    Args:
        values:
            The values an analysis function receives.

        argument:
            The name of the argument that holds the values, used in the error message.

    Returns:
        array:
            The values as floats.

    Raises:
        ValidationError:
            The values are complex.
    """
    array = np.asarray(values)
    if np.iscomplexobj(array):
        error_msg = (
            f'The {argument} are complex, and the analysis takes real values. Pass one real view of them: the real '
            'part `values.real`, the imaginary part `values.imag`, the modulus `np.abs(values)`, or the argument '
            '`np.angle(values)`.'
        )
        raise ValidationError(error_msg)
    return np.asarray(array, dtype=float)


def sample_columns(data: npt.ArrayLike) -> npt.NDArray[Any]:
    """Return data as one row per sample and one column per variable, a series as a single column.

    Args:
        data:
            A series, or one row per sample with the variables along the remaining axes.

    Returns:
        columns:
            The data as a two-dimensional array.
    """
    array = np.asarray(data)
    return array.reshape(-1, 1) if array.ndim == 1 else array.reshape(array.shape[0], -1)
