"""Measure the structure of fields: domains, patterns, and fluctuations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ._correlation import _correlation_function
from ._kinds import DataKind, analysis
from ._spectrum import _mode_power, _structure_factor

type Array = npt.NDArray[np.float64]
DOMAIN_METHODS = ('moment', 'zero')


@analysis(DataKind.FIELD)
def domain_size(field: npt.ArrayLike, *, method: str = 'moment', spacing: float = 1.0) -> float:
    """Return the characteristic domain size of one frame of a field.

    Args:
        field:
            The values on a periodic grid.

        method:
            `'moment'` for `2 pi / <k>` over the structure factor, or `'zero'` for the first zero of the correlation
            function.

        spacing:
            The distance between neighbouring sites.

    Returns:
        size:
            The domain size.

    Raises:
        AnalysisError:
            The method is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import domain_size
        >>> x = np.arange(64)
        >>> stripes = np.sign(np.cos(2 * np.pi * x / 16))
        >>> domain_size(stripes, method='zero')
        4.0
    """
    if method not in DOMAIN_METHODS:
        error_msg = f'Unknown method {method!r}. Use one of {", ".join(DOMAIN_METHODS)}.'
        raise AnalysisError(error_msg)
    if method == 'zero':
        distances, values = _correlation_function(field)
        crossing = np.flatnonzero(values <= 0)
        index = int(crossing[0]) if crossing.size else distances.size - 1
        return float(distances[index] * spacing)
    wavenumbers, power = _structure_factor(np.asarray(field, dtype=float)[None], spacing)
    return float(2 * np.pi * power.sum() / np.sum(wavenumbers * power))


@dataclass(frozen=True)
class DomainGrowth:
    """Domain size over time and its growth exponent.

    Args:
        times:
            The times of the frames.

        sizes:
            The domain size of every frame.

        exponent:
            The slope of `log size` against `log time` over the fit range, 1/2 for non-conserved and 1/3 for
            conserved order parameters.
    """

    times: Array
    sizes: Array
    exponent: float


@analysis(DataKind.FIELD)
def domain_growth(
    times: npt.ArrayLike,
    frames: npt.ArrayLike,
    *,
    method: str = 'moment',
    fit_range: tuple[float, float] | None = None,
) -> DomainGrowth:
    """Return the domain size of every frame of a coarsening field and the exponent of its growth in time.

    Args:
        times:
            The times of the frames.

        frames:
            The frames, with time as the first axis.

        method:
            The domain size method, `'moment'` or `'zero'`.

        fit_range:
            The first and last times of the fit, or `None` for every positive time.

    Returns:
        growth:
            The times, the sizes, and the exponent.

    Raises:
        AnalysisError:
            The fit range holds fewer than 2 frames.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import domain_growth
        >>> x = np.arange(256)
        >>> stack = [np.sign(np.cos(2 * np.pi * x / (8 * t))) for t in (1, 2, 4)]
        >>> round(domain_growth([1, 2, 4], stack, method='zero').exponent, 2)
        1.0
    """
    clock = np.asarray(times, dtype=float)
    sizes = np.asarray([domain_size(frame, method=method) for frame in np.asarray(frames, dtype=float)])
    low, high = (0.0, np.inf) if fit_range is None else fit_range
    inside = (clock >= low) & (clock <= high) & (clock > 0) & (sizes > 0)
    if inside.sum() < 2:  # noqa: PLR2004
        error_msg = f'The fit range {fit_range} holds fewer than 2 frames with a positive time and size.'
        raise AnalysisError(error_msg)
    exponent = float(np.polyfit(np.log(clock[inside]), np.log(sizes[inside]), 1)[0])
    return DomainGrowth(clock, sizes, exponent)


@dataclass(frozen=True)
class Pattern:
    """Dominant wavelength and orientation of a pattern.

    Args:
        wavelength:
            The wavelength of the peak of the structure factor.

        wavenumber:
            The wavenumber of the peak.

        orientation:
            The nematic order of the wave vectors near the peak, 1 for parallel stripes and near 0 for isotropic
            patterns such as labyrinths or spots.
    """

    wavelength: float
    wavenumber: float
    orientation: float


@analysis(DataKind.FIELD)
def pattern_wavelength(field: npt.ArrayLike, *, spacing: float = 1.0) -> Pattern:
    """Return the dominant wavelength of a two-dimensional pattern and the orientation order of its wave vectors.

    Args:
        field:
            The values on a periodic grid.

        spacing:
            The distance between neighbouring sites.

    Returns:
        pattern:
            The wavelength, the wavenumber, and the orientation order.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import pattern_wavelength
        >>> x = np.arange(64)
        >>> pattern = pattern_wavelength(np.cos(2 * np.pi * x / 8)[:, None] * np.ones(64))
        >>> pattern.wavelength, round(pattern.orientation, 2)
        (8.0, 1.0)
    """
    grid = np.asarray(field, dtype=float)
    wavenumbers, power = _structure_factor(grid[None], spacing)
    peak = float(wavenumbers[power.argmax()])
    power = _mode_power(grid[None])
    axes = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(size, spacing) for size in grid.shape), indexing='ij')
    moduli = np.sqrt(sum(axis**2 for axis in axes))
    width = 2 * np.pi / (max(grid.shape) * spacing)
    near = np.abs(moduli - peak) <= width
    angles = np.arctan2(axes[1], axes[0]) if grid.ndim > 1 else np.zeros_like(moduli)
    weights = power[near]
    orientation = float(np.abs(np.sum(weights * np.exp(2j * angles[near]))) / weights.sum())
    return Pattern(2 * np.pi / peak, peak, orientation)


@dataclass(frozen=True)
class NumberFluctuations:
    """Number fluctuations of a density field in boxes of growing size.

    Args:
        means:
            The mean number in boxes of every size.

        deviations:
            The standard deviation of the number in boxes of every size.

        exponent:
            The slope of `log deviation` against `log mean`, 1/2 for independent particles and above for giant
            number fluctuations.
    """

    means: Array
    deviations: Array
    exponent: float


@analysis(DataKind.FIELD)
def number_fluctuations(density: npt.ArrayLike, boxes: npt.ArrayLike | None = None) -> NumberFluctuations:
    """Return the number fluctuations of a two-dimensional density field in square boxes of growing side.

    Args:
        density:
            The number of particles in every cell, such as `coarse_grain` returns, or a stack of frames of them.

        boxes:
            The box sides in cells, or `None` for powers of 2 up to a quarter of the smallest side.

    Returns:
        fluctuations:
            The mean and standard deviation of the number in every box size, and the exponent.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import number_fluctuations
        >>> counts = np.random.default_rng(0).poisson(5.0, (20, 64, 64))
        >>> round(number_fluctuations(counts).exponent, 1)
        0.5
    """
    array = np.asarray(density, dtype=float)
    grids = array if array.ndim == 3 else array[None]  # noqa: PLR2004
    smallest = min(grids.shape[1:])
    sides = 2 ** np.arange(int(np.log2(max(smallest // 4, 1))) + 1) if boxes is None else np.asarray(boxes, dtype=int)
    means, deviations = [], []
    for side in sides:
        rows, columns = grids.shape[1] // side, grids.shape[2] // side
        trimmed = grids[:, : rows * side, : columns * side]
        counts = trimmed.reshape(grids.shape[0], rows, side, columns, side).sum(axis=(2, 4)).ravel()
        means.append(counts.mean())
        deviations.append(counts.std())
    exponent = float(np.polyfit(np.log(means), np.log(deviations), 1)[0])
    return NumberFluctuations(np.asarray(means), np.asarray(deviations), exponent)
