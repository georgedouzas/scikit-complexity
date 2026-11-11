"""Measure the mean squared displacement of particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import fft

from ..engine.runtime import Trajectory
from ..enums import DataKind
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array
from ._base import _read_real, _read_snapshots
from .types import Source

MIN_LINE_POINTS = 2
FIT_POINTS = 20
TAGS = AnalysisTags(DataKind.AGENTS)


@dataclass(frozen=True)
class DisplacementResult:
    """A result of `displacement`, the mean squared displacement over lags.

    It comes with its scaling exponent and the diffusion coefficient.

    Args:
        positions:
            The unwrapped positions, frames by particles by dimensions.

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

    positions: Array
    lags: Array
    msd: Array
    exponent: float
    diffusion: float


def displacement(
    source: Source,
    *,
    group: str | None = None,
    dt: float = 1.0,
    lags: Sequence[int] | None = None,
    fit_range: tuple[float, float] | None = None,
) -> DisplacementResult:
    """Return the mean squared displacement of particles over all lags.

    It comes with its scaling exponent and the diffusion. The displacement of every particle is computed for all lags at
    once with an FFT, so positions must be unwrapped across periodic boundaries, as a trajectory gives them.

    It appears in many fields. It tells diffusion from flight in the paths of particles in a fluid, of cells in a
    tissue, and of animals that forage.

    Args:
        source:
            The trajectory, or an array of positions with time as the first axis.

        group:
            The path of the agent group of a trajectory, or `None` for the first one.

        dt:
            The time between frames.

        lags:
            The lags in frames to keep, or `None` for every lag from 1.

        fit_range:
            The smallest and largest lags in time of the fits, or `None` for one frame to a quarter of the run.

    Returns:
        result:
            The positions, the mean squared displacement at every lag, its exponent, and the diffusion coefficient.

    Raises:
        AnalysisError:
            The time step or a lag is not positive, the agents have no positions, or the fit range holds too few lags.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import displacement
        >>> walks = np.cumsum(np.random.default_rng(0).standard_normal((2000, 500)), axis=0)
        >>> result = displacement(walks)
        >>> round(result.exponent, 1), round(result.diffusion, 1)
        (1.0, 0.5)
    """
    if dt <= 0:
        error_msg = f'displacement has dt={dt!r}. Use a positive time step.'
        raise AnalysisError(error_msg)
    if lags is not None and min(lags, default=1) < 1:
        error_msg = f'displacement has lags={lags!r}. Use lags of 1 frame or more.'
        raise AnalysisError(error_msg)
    if isinstance(source, Trajectory):
        read = _read_snapshots(source, group).positions
        if read is None:
            error_msg = f'The agents of {group or "the first group"} have no positions. Use agents in a space.'
            raise AnalysisError(error_msg)
        positions = _particle_axes(read)
    else:
        positions = _particle_axes(np.asarray(source))
    msd = np.mean([_get_msd(positions[:, particle]) for particle in range(positions.shape[1])], axis=0)
    all_lags = np.arange(1, positions.shape[0], dtype=np.float64) * dt
    chosen = _fit_lags(all_lags, fit_range)
    exponent = float(np.polyfit(np.log(all_lags[chosen]), np.log(msd[chosen]), 1)[0])
    slope = float(np.polyfit(all_lags[chosen], msd[chosen], 1)[0])
    diffusion = slope / (2 * positions.shape[2])
    if lags is None:
        return DisplacementResult(positions, all_lags, msd, exponent, diffusion)
    steps = np.asarray(lags, dtype=int)
    return DisplacementResult(positions, steps * dt, msd[steps - 1], exponent, diffusion)


def _get_msd(positions: Array) -> Array:
    """Return the mean squared displacement of one particle at every lag from 1.

    It is computed by fast correlation. The displacement `<(r[t + m] - r[t])²>` splits into the mean of `r[t]² + r[t +
    m]²`, from cumulative sums, and twice the autocorrelation at lag `m`, from an FFT of the positions padded against
    circular wrapping.
    """
    frames = positions.shape[0]
    size = fft.next_fast_len(2 * frames, real=True)
    spectrum = fft.rfft(positions, n=size, axis=0)
    correlation = fft.irfft(np.abs(spectrum) ** 2, n=size, axis=0)[1:frames].sum(axis=1)
    squares = np.sum(positions**2, axis=1)
    endpoints = 2 * squares.sum() - np.cumsum(squares)[:-1] - np.cumsum(squares[::-1])[:-1]
    return (endpoints - 2 * correlation) / (frames - np.arange(1, frames))


def _particle_axes(positions: npt.ArrayLike) -> Array:
    """Return positions as frames by particles by dimensions."""
    array = _read_real(positions, 'positions')
    return array.reshape(array.shape[0], array.shape[1] if array.ndim > 1 else 1, -1)


def _fit_lags(lags: Array, fit_range: tuple[float, float] | None) -> npt.NDArray[np.intp]:
    """Return the indices of about 20 lags spaced logarithmically within the fit range."""
    low, high = (lags[0], lags[-1] / 4) if fit_range is None else fit_range
    inside = np.flatnonzero((lags >= low) & (lags <= high))
    if inside.size < MIN_LINE_POINTS:
        error_msg = f'The fit range {fit_range} holds fewer than 2 lags. Widen it within {lags[0]} to {lags[-1]}.'
        raise AnalysisError(error_msg)
    chosen = np.unique(np.geomspace(1, inside.size, FIT_POINTS).astype(int) - 1)
    return inside[chosen]
