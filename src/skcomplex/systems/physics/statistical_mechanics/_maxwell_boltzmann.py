"""Define the Maxwell-Boltzmann test of the speeds of agents or particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from ....core.analysis.types import Source
from ....core.errors import AnalysisError
from ....core.types import Array
from ._kinetic_temperature import _kinetic_temperature, _read_velocities


@dataclass(frozen=True)
class MaxwellBoltzmannResult:
    """Kolmogorov-Smirnov test of the speeds of every frame.

    The test compares the speeds with the Maxwell-Boltzmann law.

    Args:
        values:
            The velocities, frames by agents by dimensions.

        temperature:
            The temperature of the tested law at every frame.

        statistic:
            The Kolmogorov-Smirnov statistic of every frame.

        p_value:
            The p-value of every frame.
    """

    values: Array
    temperature: Array
    statistic: Array
    p_value: Array


def maxwell_boltzmann(
    source: Source,
    *,
    group: str | None = None,
    mass: float = 1.0,
    temperature: float | None = None,
) -> MaxwellBoltzmannResult:
    """Return the Kolmogorov-Smirnov test of the speeds of every frame.

    The test compares the speeds with the Maxwell-Boltzmann law.

    It is the test of the named law `'maxwell'` of `fit_law`, generalized to `d` dimensions: the speeds of an ideal gas
    at temperature `T` follow a chi law of `d` degrees of freedom and scale `sqrt(T / m)`, the Maxwell law in three
    dimensions and the Rayleigh law in two. The temperature is the kinetic temperature of every frame unless it is
    given. It reads the velocities of the agents or particles of a trajectory, or an array of velocities of one
    snapshot, one row per agent, or of one snapshot per frame along its first axis.

    Args:
        source:
            The trajectory, or an array of velocities.

        group:
            The path of the agent or particle group of a trajectory, or `None` for the first one.

        mass:
            The mass of every agent.

        temperature:
            The temperature of the law, or `None` for the kinetic temperature of every frame.

    Returns:
        result:
            The velocities, the temperature of the law, and the test of every frame.

    Raises:
        AnalysisError:
            The mass or the temperature is not positive, the source is neither a trajectory nor an array, or the agents
            have no velocities.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.physics.statistical_mechanics import maxwell_boltzmann
        >>> velocities = np.random.default_rng(0).standard_normal((2000, 3))
        >>> bool(maxwell_boltzmann(velocities).p_value[0] > 0.05)
        True
    """
    if mass <= 0 or (temperature is not None and temperature <= 0):
        error_msg = f"maxwell_boltzmann has mass={mass!r} and temperature={temperature!r}. Use positive values."
        raise AnalysisError(error_msg)
    velocities = _read_velocities(source, group, "maxwell_boltzmann")
    tests = [_maxwell_boltzmann(frame, mass, temperature) for frame in velocities]
    temperatures, statistics, p_values = np.asarray(tests).T
    return MaxwellBoltzmannResult(velocities, temperatures, statistics, p_values)


def _maxwell_boltzmann(frame: npt.ArrayLike, mass: float, temperature: float | None) -> tuple[float, float, float]:
    """Return the temperature and the Kolmogorov-Smirnov test of one frame's speeds.

    The test compares the speeds with the chi law.
    """
    velocities = np.asarray(frame, dtype=float)
    chosen = float(_kinetic_temperature(velocities, mass)) if temperature is None else temperature
    speeds = np.linalg.norm(velocities, axis=-1)
    law = stats.chi(df=velocities.shape[-1], scale=math.sqrt(chosen / mass))
    test = stats.kstest(speeds, law.cdf)
    return chosen, float(test.statistic), float(test.pvalue)
