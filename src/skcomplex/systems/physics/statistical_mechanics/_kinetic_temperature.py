"""Define the kinetic temperature of moving agents or particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np

from ....core.analysis import alignment
from ....core.analysis.types import Source
from ....core.engine.runtime import Trajectory
from ....core.errors import AnalysisError
from ....core.types import Array

SNAPSHOT_DIMENSIONS = 2


@dataclass(frozen=True)
class KineticTemperatureResult:
    """Kinetic temperature of the agents of every frame.

    Args:
        values:
            The velocities, frames by agents by dimensions.

        temperature:
            The kinetic temperature of every frame.
    """

    values: Array
    temperature: Array


def kinetic_temperature(source: Source, *, group: str | None = None, mass: float = 1.0) -> KineticTemperatureResult:
    """Return the kinetic temperature `m <|v|²> / d` of the agents of every frame in `d` dimensions, with `k_B = 1`.

    It is the second moment of the speeds of the agents scaled by the mass over the dimension, as equipartition gives
    `k_B T / 2` per degree of freedom. It reads the velocities of the agents or particles of a trajectory, as the
    alignment of the agents reads them, or an array of velocities of one snapshot, one row per agent, or of one
    snapshot per frame along its first axis.

    Args:
        source:
            The trajectory, or an array of velocities.

        group:
            The path of the agent or particle group of a trajectory, or `None` for the first one.

        mass:
            The mass of every agent.

    Returns:
        result:
            The velocities and the kinetic temperature of every frame.

    Raises:
        AnalysisError:
            The mass is not positive, the source is neither a trajectory nor an array, or the agents have no
            velocities.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import kinetic_temperature
        >>> kinetic_temperature([[1.0, 0.0], [0.0, 1.0]]).temperature
        array([0.5])
    """
    if mass <= 0:
        error_msg = f'kinetic_temperature has mass={mass!r}. Use a positive mass.'
        raise AnalysisError(error_msg)
    velocities = _read_velocities(source, group, 'kinetic_temperature')
    return KineticTemperatureResult(velocities, _kinetic_temperature(velocities, mass))


def _read_velocities(source: Source, group: str | None, analysis: str) -> Array:
    """Return the velocities of the agents of a trajectory, or an array of one snapshot or of frames."""
    if not isinstance(source, Trajectory | np.ndarray | list | tuple):
        error_msg = f'{analysis} reads a trajectory or an array of velocities, not {source!r}.'
        raise AnalysisError(error_msg)
    if not isinstance(source, Trajectory):
        array = np.asarray(source, dtype=float)
        return array[None] if array.ndim == SNAPSHOT_DIMENSIONS else array
    velocities = alignment(source, group=group).vectors
    if velocities is None:
        error_msg = f'The agents of {group or "the first group"} have no velocities. Use moving agents.'
        raise AnalysisError(error_msg)
    return velocities


def _kinetic_temperature(velocities: Array, mass: float) -> Array:
    """Return `m <|v|²> / d` of the velocities of the agents of every frame in `d` dimensions."""
    return mass * np.mean(np.sum(velocities**2, axis=-1), axis=-1) / velocities.shape[-1]
