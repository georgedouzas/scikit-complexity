"""Measure the diffusion of particles: mean squared displacements and their scaling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import tidynamics

from ..core.errors import AnalysisError
from ._kinds import DataKind, analysis
from ._utils import real_array

type Array = npt.NDArray[np.float64]
FIT_POINTS = 20


def _particle_axes(positions: npt.ArrayLike) -> Array:
    """Return positions as frames by particles by dimensions."""
    array = real_array(positions, 'positions')
    return array.reshape(array.shape[0], array.shape[1] if array.ndim > 1 else 1, -1)


def _fit_lags(lags: Array, fit_range: tuple[float, float] | None) -> npt.NDArray[np.intp]:
    """Return the indices of about 20 lags spaced logarithmically within the fit range."""
    low, high = (lags[0], lags[-1] / 4) if fit_range is None else fit_range
    inside = np.flatnonzero((lags >= low) & (lags <= high))
    if inside.size < 2:  # noqa: PLR2004
        error_msg = f'The fit range {fit_range} holds fewer than 2 lags. Widen it within {lags[0]} to {lags[-1]}.'
        raise AnalysisError(error_msg)
    chosen = np.unique(np.geomspace(1, inside.size, FIT_POINTS).astype(int) - 1)
    return inside[chosen]


@dataclass(frozen=True)
class Displacement:
    """Mean squared displacement over lags, its scaling exponent, and the diffusion coefficient.

    Args:
        lags:
            The lags in time.

        msd:
            The mean squared displacement at every lag.

        exponent:
            The slope of `log msd` against `log lag` over the fit range, 1 for normal diffusion, below 1 for
            subdiffusion, and above 1 for superdiffusion.

        diffusion:
            The coefficient `D` of `msd = 2 d D lag` fitted over the fit range in `d` dimensions.
    """

    lags: Array
    msd: Array
    exponent: float
    diffusion: float = float('nan')


@analysis(DataKind.AGENTS)
def mean_squared_displacement(
    positions: npt.ArrayLike,
    dt: float = 1.0,
    lags: npt.ArrayLike | None = None,
    *,
    fit_range: tuple[float, float] | None = None,
) -> Displacement:
    """Return the mean squared displacement of particles over all lags, its scaling exponent, and the diffusion.

    The displacement of every particle is computed for all lags at once with the FFT, so positions must be unwrapped
    across periodic boundaries, as `snapshots_of` returns them.

    Args:
        positions:
            The unwrapped positions with time as the first axis, then particles, then optionally dimensions.

        dt:
            The time between frames.

        lags:
            The lags in frames to return, or `None` for every lag from 1.

        fit_range:
            The smallest and largest lags in time of the fits, or `None` for one frame to a quarter of the run.

    Returns:
        displacement:
            The lags, the mean squared displacement, its exponent, and the diffusion coefficient.

    Raises:
        AnalysisError:
            The fit range holds fewer than 2 lags.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import mean_squared_displacement
        >>> walks = np.cumsum(np.random.default_rng(0).standard_normal((2000, 500)), axis=0)
        >>> displacement = mean_squared_displacement(walks)
        >>> round(displacement.exponent, 1), round(displacement.diffusion, 1)
        (1.0, 0.5)
    """
    array = _particle_axes(positions)
    msd = np.mean([tidynamics.msd(array[:, particle]) for particle in range(array.shape[1])], axis=0)[1:]
    all_lags = np.arange(1, array.shape[0], dtype=np.float64) * dt
    chosen = _fit_lags(all_lags, fit_range)
    exponent = float(np.polyfit(np.log(all_lags[chosen]), np.log(msd[chosen]), 1)[0])
    slope = float(np.polyfit(all_lags[chosen], msd[chosen], 1)[0])
    diffusion = slope / (2 * array.shape[2])
    if lags is None:
        return Displacement(all_lags, msd, exponent, diffusion)
    steps = np.asarray(lags, dtype=int)
    return Displacement(steps * dt, msd[steps - 1], exponent, diffusion)
