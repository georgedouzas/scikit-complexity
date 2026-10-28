"""Measure spectra: power spectra of series and structure factors of fields."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy import signal

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

KINDS = ('power', 'structure-factor')
SETTINGS = {
    'power': {'dt'},
    'structure-factor': {'spacing', 'frames'},
}
"""Settings every kind reads, besides `kind` and `variables`."""


class SpectrumAnalysis(BaseAnalysis):
    """Spectrum of every series of a source, or of a field.

    The kinds read series or fields:

    - `'power'`: the power spectral density of every series by Welch's method.
    - `'structure-factor'`: the radially averaged structure factor `S(k)` of a field on a periodic grid of any
      dimension, in shells of width `2 pi / (side spacing)`, averaged over frames when `frames` is true.

    A structure factor reads the last frame of a trajectory, or every frame when `frames` is true.

    Args:
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

    Attributes:
        values_:
            The series, one per column, or the field.

        labels_:
            The label of every series, or of the field.

        frequencies_:
            The frequencies of `'power'`.

        power_:
            One density per series and frequency for `'power'`, and the mean power in every shell for
            `'structure-factor'`.

        wavenumbers_:
            The wavenumbers of the shells, without 0, for `'structure-factor'`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import SpectrumAnalysis
        >>> times = np.arange(4096) * 0.01
        >>> spectrum = SpectrumAnalysis(dt=0.01).check().prepare(np.sin(2 * np.pi * 5 * times)).run()
        >>> round(float(spectrum.frequencies_[spectrum.power_[0].argmax()]))
        5
        >>> stripes = np.cos(2 * np.pi * np.arange(64) / 8)[:, None] * np.ones(64)
        >>> factor = SpectrumAnalysis('structure-factor').check().prepare(stripes).run()
        >>> round(float(2 * np.pi / factor.wavenumbers_[factor.power_.argmax()]), 1)
        8.0
    """

    def __init__(
        self,
        kind: str = 'power',
        variables: Sequence[str] | None = None,
        *,
        dt: float = 1.0,
        spacing: float = 1.0,
        frames: bool = False,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.dt = dt
        self.spacing = spacing
        self.frames = frames

    def _check(self) -> Self:
        """Validate the kind, the numeric settings, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.dt <= 0 or self.spacing <= 0:
            error_msg = f'{type(self).__name__} has dt={self.dt!r} and spacing={self.spacing!r}. Use positive values.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the series kind for power spectra and the field kind for fields."""
        return DataKind.SERIES if self.kind == 'power' else DataKind.FIELD

    def _prepare(self, source: Source) -> Self:
        """Read the series as columns, or the frames of a field."""
        if self.kind == 'power':
            series = self._read_series(source, self.variables)
            self.values_ = series.values
            self.labels_ = series.labels
            return self
        name = self.variables[0] if self.variables else None
        field = self._read_fields(source, name)
        if isinstance(source, Trajectory) and not self.frames:
            field = field[-1]
        self.values_ = field
        self.labels_ = [name or 'field']
        return self

    def _run(self, inputs: object = None) -> Self:
        """Compute the spectrum of the kind."""
        if self.kind == 'power':
            spectra = [self._power_spectrum(column, self.dt) for column in self.values_.T]
            self.frequencies_ = spectra[0][0]
            self.power_ = np.asarray([density for _, density in spectra])
        else:
            grids = self.values_ if self.frames else self.values_[None]
            self.wavenumbers_, self.power_ = self._structure_factor(grids, self.spacing)
        return self

    @staticmethod
    def _power_spectrum(values: npt.ArrayLike, dt: float = 1.0) -> tuple[Array, Array]:
        """Return the frequencies and the power spectral density of a series by Welch's method."""
        array = BaseAnalysis._read_real(values, 'values')
        return signal.welch(array, fs=1 / dt, nperseg=min(array.size, 1024))

    @staticmethod
    def _wavenumbers(shape: tuple[int, ...], spacing: float) -> Array:
        """Return the modulus of the wave vector of every Fourier mode of a periodic grid."""
        axes = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(size, spacing) for size in shape), indexing='ij')
        return np.sqrt(sum(axis**2 for axis in axes))

    @staticmethod
    def _mode_power(grids: Array) -> Array:
        """Return the power of every Fourier mode of the fluctuations, averaged over frames."""
        fluctuations = grids - grids.mean(axis=tuple(range(1, grids.ndim)), keepdims=True)
        axes = tuple(range(1, grids.ndim))
        return np.mean(np.abs(np.fft.fftn(fluctuations, axes=axes)) ** 2, axis=0) / np.prod(grids.shape[1:])

    @staticmethod
    def _structure_factor(grids: Array, spacing: float = 1.0) -> tuple[Array, Array]:
        """Return the wavenumbers and the radially averaged structure factor of frames, in shells of `2 pi / side`."""
        power = SpectrumAnalysis._mode_power(grids)
        moduli = SpectrumAnalysis._wavenumbers(grids.shape[1:], spacing)
        width = 2 * np.pi / (max(grids.shape[1:]) * spacing)
        shells = np.rint(moduli / width).astype(int).ravel()
        sums = np.bincount(shells, power.ravel())
        counts = np.bincount(shells)
        used = counts > 0
        used[0] = False
        return np.flatnonzero(used) * width, sums[used] / counts[used]
