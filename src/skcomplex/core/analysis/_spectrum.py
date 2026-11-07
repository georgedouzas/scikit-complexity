"""Measure spectra: power spectra of series and structure factors of fields."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array
from ._base import _check_choice, _check_settings, _read_fields, _read_real, _read_series
from .types import Source

KINDS = ('power', 'structure-factor')
# Settings every kind reads, besides `kind` and `variables`.
SETTINGS = {
    'power': {'dt'},
    'structure-factor': {'spacing', 'frames'},
}


@dataclass(frozen=True)
class SpectrumResult:
    """Spectrum of every series of a source, or of a field.

    Args:
        values:
            The series, one per column, or the field.

        labels:
            The label of every series, or of the field.

        power:
            One density per series and frequency for `'power'`, and the mean power in every shell for
            `'structure-factor'`.

        frequencies:
            The frequencies, for `'power'`.

        wavenumbers:
            The wavenumbers of the shells, without 0, for `'structure-factor'`.
    """

    values: Array
    labels: list[str]
    power: Array
    frequencies: Array | None = None
    wavenumbers: Array | None = None


def spectrum(
    source: Source,
    *,
    kind: str = 'power',
    variables: Sequence[str] | None = None,
    dt: float = 1.0,
    spacing: float = 1.0,
    frames: bool = False,
) -> SpectrumResult:
    """Return the spectrum of every series of a source, or of a field.

    The kinds read series or fields:

    - `'power'`: the power spectral density of every series by Welch's method.
    - `'structure-factor'`: the radially averaged structure factor `S(k)` of a field on a periodic grid of any
      dimension, in shells of width `2 pi / (side spacing)`, averaged over frames when `frames` is true.

    A structure factor reads the last frame of a trajectory, or every frame when `frames` is true.

    Args:
        source:
            The trajectory, or an array of series or of a field.

        kind:
            The spectrum to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series, or the first field for a field.

        dt:
            The time between samples.

        spacing:
            The distance between neighbouring sites.

        frames:
            Whether the first axis of an array indexes frames, and whether to read every frame of a trajectory.

    Returns:
        result:
            The data read and its spectrum.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, or a setting is not positive.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import spectrum
        >>> times = np.arange(4096) * 0.01
        >>> result = spectrum(np.sin(2 * np.pi * 5 * times), dt=0.01)
        >>> round(float(result.frequencies[result.power[0].argmax()]))
        5
        >>> stripes = np.cos(2 * np.pi * np.arange(64) / 8)[:, None] * np.ones(64)
        >>> factor = spectrum(stripes, kind='structure-factor')
        >>> round(float(2 * np.pi / factor.wavenumbers[factor.power.argmax()]), 1)
        8.0
    """
    _check_choice('spectrum', 'kind', kind, KINDS)
    _check_settings(
        spectrum,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        dt=dt,
        spacing=spacing,
        frames=frames,
    )
    if dt <= 0 or spacing <= 0:
        error_msg = f'spectrum has dt={dt!r} and spacing={spacing!r}. Use positive values.'
        raise AnalysisError(error_msg)
    if kind == 'power':
        series = _read_series(source, variables)
        spectra = [_power_spectrum(column, dt) for column in series.values.T]
        power = np.asarray([density for _, density in spectra])
        return SpectrumResult(series.values, series.labels, power, frequencies=spectra[0][0])
    name = variables[0] if variables else None
    field = _read_fields(source, name)
    if isinstance(source, Trajectory) and not frames:
        field = field[-1]
    wavenumbers, power = _structure_factor(field if frames else field[None], spacing)
    return SpectrumResult(field, [name or 'field'], power, wavenumbers=wavenumbers)


def _power_spectrum(values: npt.ArrayLike, dt: float = 1.0) -> tuple[Array, Array]:
    """Return the frequencies and the power spectral density of a series by Welch's method."""
    array = _read_real(values, 'values')
    return signal.welch(array, fs=1 / dt, nperseg=min(array.size, 1024))


def _wavenumbers(shape: tuple[int, ...], spacing: float) -> Array:
    """Return the modulus of the wave vector of every Fourier mode of a periodic grid."""
    axes = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(size, spacing) for size in shape), indexing='ij')
    return np.sqrt(sum(axis**2 for axis in axes))


def _mode_power(grids: Array) -> Array:
    """Return the power of every Fourier mode of the fluctuations, averaged over frames."""
    fluctuations = grids - grids.mean(axis=tuple(range(1, grids.ndim)), keepdims=True)
    axes = tuple(range(1, grids.ndim))
    return np.mean(np.abs(np.fft.fftn(fluctuations, axes=axes)) ** 2, axis=0) / np.prod(grids.shape[1:])


def _structure_factor(grids: Array, spacing: float = 1.0) -> tuple[Array, Array]:
    """Return the wavenumbers and the radially averaged structure factor of frames, in shells of `2 pi / side`."""
    power = _mode_power(grids)
    moduli = _wavenumbers(grids.shape[1:], spacing)
    width = 2 * np.pi / (max(grids.shape[1:]) * spacing)
    shells = np.rint(moduli / width).astype(int).ravel()
    sums = np.bincount(shells, power.ravel())
    counts = np.bincount(shells)
    used = counts > 0
    used[0] = False
    return np.flatnonzero(used) * width, sums[used] / counts[used]
