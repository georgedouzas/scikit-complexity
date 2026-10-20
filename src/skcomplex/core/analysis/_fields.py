"""Measure the structure of fields: structure factors, correlation lengths, domains, patterns, and fluctuations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize

from ..base import AnalysisError
from ._kinds import DataKind, analysis
from ._spatial import correlation_function

type Array = npt.NDArray[np.float64]
LENGTH_METHODS = ('moment', 'fit')
DOMAIN_METHODS = ('moment', 'zero')
MOMENT_ITERATIONS = 8


def _frames(field: npt.ArrayLike, *, frames: bool) -> Array:
    """Return a field as a stack of frames."""
    array = np.asarray(field, dtype=float)
    return array if frames else array[None]


def _wavenumbers(shape: tuple[int, ...], spacing: float) -> Array:
    """Return the modulus of the wave vector of every Fourier mode of a periodic grid."""
    axes = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(size, spacing) for size in shape), indexing='ij')
    return np.sqrt(sum(axis**2 for axis in axes))


def _spectrum(grids: Array) -> Array:
    """Return the power of every Fourier mode of the fluctuations, averaged over frames."""
    fluctuations = grids - grids.mean(axis=tuple(range(1, grids.ndim)), keepdims=True)
    axes = tuple(range(1, grids.ndim))
    return np.mean(np.abs(np.fft.fftn(fluctuations, axes=axes)) ** 2, axis=0) / np.prod(grids.shape[1:])


@dataclass(frozen=True)
class StructureFactor:
    """Radially averaged structure factor of a field.

    Args:
        wavenumbers:
            The wavenumbers of the shells, without 0.

        power:
            The mean power of the Fourier modes of the fluctuations in every shell.
    """

    wavenumbers: Array
    power: Array


@analysis(DataKind.FIELD)
def structure_factor(field: npt.ArrayLike, *, spacing: float = 1.0, frames: bool = False) -> StructureFactor:
    """Return the radially averaged structure factor `S(k)` of a field on a periodic grid of any dimension.

    Args:
        field:
            The values on the grid, or a stack of frames when `frames` is true.

        spacing:
            The distance between neighbouring sites.

        frames:
            Whether the first axis indexes frames, whose structure factors are averaged.

    Returns:
        factor:
            The wavenumbers and the power in every shell of width `2 pi / (side spacing)`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import structure_factor
        >>> x = np.arange(64)
        >>> stripes = np.cos(2 * np.pi * x / 8)[:, None] * np.ones(64)
        >>> factor = structure_factor(stripes)
        >>> round(float(2 * np.pi / factor.wavenumbers[factor.power.argmax()]), 1)
        8.0
    """
    grids = _frames(field, frames=frames)
    power = _spectrum(grids)
    moduli = _wavenumbers(grids.shape[1:], spacing)
    width = 2 * np.pi / (max(grids.shape[1:]) * spacing)
    shells = np.rint(moduli / width).astype(int).ravel()
    sums = np.bincount(shells, power.ravel())
    counts = np.bincount(shells)
    used = counts > 0
    used[0] = False
    return StructureFactor(np.flatnonzero(used) * width, sums[used] / counts[used])


def _ornstein_zernike_length(hat: Array, power: Array, limit: float) -> float:
    """Return `sqrt(slope / intercept)` of a fit of `1 / S` against `q̂²` over the modes with `q̂²` up to a limit."""
    inside = (hat <= limit) & (power > 0)
    if inside.sum() < 2:  # noqa: PLR2004
        return float('nan')
    slope, intercept = np.polyfit(hat[inside], 1 / power[inside], 1)
    if slope <= 0:
        return 0.0
    if intercept <= 0:
        return float('inf')
    return float(np.sqrt(slope / intercept))


def _moment_length(grids: Array, spacing: float) -> float:
    """Return the second-moment length from fits of `1 / S(q) ∝ 1 + xi² q̂²` over the modes with `q̂² xi² <= 1`."""
    shape = grids.shape[1:]
    axes = tuple(range(1, grids.ndim))
    fluctuations = grids - (grids.mean() if grids.shape[0] > 1 else grids.mean(axis=axes, keepdims=True))
    power = (np.mean(np.abs(np.fft.fftn(fluctuations, axes=axes)) ** 2, axis=0) / np.prod(shape)).ravel()
    lattice = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(size) for size in shape), indexing='ij')
    hat = np.sum([4 * np.sin(axis / 2) ** 2 for axis in lattice], axis=0).ravel()
    if grids.shape[0] == 1:
        hat, power = hat[1:], power[1:]
    smallest = 4 * np.sin(np.pi / max(shape)) ** 2
    limit = 1.0
    length = float('nan')
    for _ in range(MOMENT_ITERATIONS):
        length = _ornstein_zernike_length(hat, power, limit)
        if not np.isfinite(length) or length <= 0:
            break
        limit = float(np.clip(1 / length**2, 2.5 * smallest, 1.0))
    return length * spacing


def _fitted_length(grids: Array, spacing: float) -> float:
    """Return the length `xi` of a fit of `C(r) = r^-a exp(-r / xi)` to the correlation function."""
    curves = [correlation_function(grid) for grid in grids]
    distances = curves[0][0][1:] * spacing
    values = np.mean([curve[1][1:] for curve in curves], axis=0)
    positive = values > 0
    if positive.sum() < 3:  # noqa: PLR2004
        error_msg = 'The correlation function is positive at fewer than 3 distances, so no length can be fitted.'
        raise AnalysisError(error_msg)

    def logarithm(distance: Array, power: float, length: float, scale: float) -> Array:
        """Return the logarithm of the fitted form."""
        return scale - power * np.log(distance) - distance / length

    guess = (0.0, float(distances[positive].mean()), 0.0)
    parameters, _ = optimize.curve_fit(
        logarithm,
        distances[positive],
        np.log(values[positive]),
        p0=guess,
        bounds=([-5.0, 1e-6, -50.0], [5.0, 1e6, 50.0]),
    )
    return float(parameters[1])


@analysis(DataKind.FIELD)
def correlation_length(
    field: npt.ArrayLike,
    *,
    method: str = 'moment',
    spacing: float = 1.0,
    frames: bool = False,
) -> float:
    """Return the correlation length of a field on a periodic grid, averaged over frames.

    The second-moment estimate fits the Ornstein-Zernike form `1 / S(q) ∝ 1 + xi² Σ 4 sin²(q_i / 2)` over the modes
    with `q̂² xi² <= 1`, narrowing the window from `q̂² <= 1` in a few iterations. Several frames are
    taken as fluctuations about their common mean, so `S(0)` enters the fit. The fit method adjusts
    `C(r) ~ r^-a exp(-r / xi)`.

    Args:
        field:
            The values on the grid, or a stack of frames when `frames` is true.

        method:
            `'moment'` or `'fit'`.

        spacing:
            The distance between neighbouring sites.

        frames:
            Whether the first axis indexes frames.

    Returns:
        length:
            The correlation length.

    Raises:
        AnalysisError:
            The method is unknown, or the correlation function cannot be fitted.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import correlation_length
        >>> noise = np.random.default_rng(0).standard_normal((20, 64, 64))
        >>> bool(correlation_length(noise, frames=True) < 0.5)
        True
    """
    if method not in LENGTH_METHODS:
        error_msg = f'Unknown method {method!r}. Use one of {", ".join(LENGTH_METHODS)}.'
        raise AnalysisError(error_msg)
    grids = _frames(field, frames=frames)
    if method == 'fit':
        return _fitted_length(grids, spacing)
    return _moment_length(grids, spacing)


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
        >>> from skcomplex.core.analysis import domain_size
        >>> x = np.arange(64)
        >>> stripes = np.sign(np.cos(2 * np.pi * x / 16))
        >>> domain_size(stripes, method='zero')
        4.0
    """
    if method not in DOMAIN_METHODS:
        error_msg = f'Unknown method {method!r}. Use one of {", ".join(DOMAIN_METHODS)}.'
        raise AnalysisError(error_msg)
    if method == 'zero':
        distances, values = correlation_function(field)
        crossing = np.flatnonzero(values <= 0)
        index = int(crossing[0]) if crossing.size else distances.size - 1
        return float(distances[index] * spacing)
    factor = structure_factor(field, spacing=spacing)
    return float(2 * np.pi * factor.power.sum() / np.sum(factor.wavenumbers * factor.power))


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
        >>> from skcomplex.core.analysis import domain_growth
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
        >>> from skcomplex.core.analysis import pattern_wavelength
        >>> x = np.arange(64)
        >>> pattern = pattern_wavelength(np.cos(2 * np.pi * x / 8)[:, None] * np.ones(64))
        >>> pattern.wavelength, round(pattern.orientation, 2)
        (8.0, 1.0)
    """
    grid = np.asarray(field, dtype=float)
    factor = structure_factor(grid, spacing=spacing)
    peak = float(factor.wavenumbers[factor.power.argmax()])
    power = _spectrum(grid[None])
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
        >>> from skcomplex.core.analysis import number_fluctuations
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
