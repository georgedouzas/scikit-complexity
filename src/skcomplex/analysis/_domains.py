"""Measure the domains of fields: their size, its growth in time, pattern wavelengths, and interface densities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._correlation import CorrelationAnalysis
from ._kinds import DataKind
from ._neighbors import NeighborAnalysis
from ._spectrum import SpectrumAnalysis

type Array = npt.NDArray[np.float64]
KINDS = ('size', 'growth', 'wavelength', 'interface-density')
METHODS = ('moment', 'zero')
SETTINGS = {
    'size': {'method', 'spacing', 'times'},
    'growth': {'method', 'spacing', 'fit_range', 'times'},
    'wavelength': {'spacing', 'times'},
    'interface-density': {'attribute', 'times'},
}
"""Settings every kind reads, besides `kind` and `variables`."""


class DomainAnalysis(BaseAnalysis):
    """Domains of every frame of a field, or of the states on a network, of one kind.

    The kinds read frames of a field on a periodic grid:

    - `'size'`: the characteristic domain size, `2 pi / <k>` over the structure factor with `method='moment'`, or the
      first zero of the correlation function with `method='zero'`.
    - `'growth'`: the domain size of every frame and the exponent of its growth in time, the slope of `log size`
      against `log time` over the fit range, 1/2 for non-conserved and 1/3 for conserved order parameters.
    - `'wavelength'`: the dominant wavelength and wavenumber of a pattern, the peak of its structure factor, and the
      nematic order of the wave vectors near the peak, 1 for parallel stripes and near 0 for labyrinths or spots.
    - `'interface-density'`: the share of neighbouring pairs, on a periodic grid without corners or along the edges of
      a network, whose states differ, 0 for one domain and 1/2 for random binary states.

    A trajectory gives the frames of a lattice state and their times. An array is one frame, or one frame per entry of
    `times` along its first axis. A graph carries the states on its nodes for `'interface-density'`.

    Args:
        kind:
            The domain measure to compute, one of the kinds above.

        variables:
            The lattice state of a trajectory to read, or `None` for the first one.

        method:
            The domain size method, `'moment'` or `'zero'`.

        spacing:
            The distance between neighbouring sites.

        fit_range:
            The first and last times of the growth fit, or `None` for every positive time.

        times:
            The time of every frame of an array, or `None` for an array of one frame.

        attribute:
            The node attribute of the states of a graph.

    Attributes:
        frames_:
            The frames of the field, or the graph.

        times_:
            The time of every frame.

        size_:
            The domain size of every frame.

        exponent_:
            The growth exponent of the domain size.

        wavelength_:
            The wavelength of the peak of every frame.

        wavenumber_:
            The wavenumber of the peak of every frame.

        orientation_:
            The orientation order of the wave vectors near the peak of every frame.

        density_:
            The interface density of every frame.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import DomainAnalysis
        >>> x = np.arange(256)
        >>> stripes = np.sign(np.cos(2 * np.pi * x / 16))
        >>> DomainAnalysis(method='zero').check().prepare(stripes[:64]).run().size_.tolist()
        [4.0]
        >>> stack = [np.sign(np.cos(2 * np.pi * x / (8 * t))) for t in (1, 2, 4)]
        >>> growth = DomainAnalysis('growth', method='zero', times=[1, 2, 4]).check().prepare(stack).run()
        >>> round(growth.exponent_, 2)
        1.0
        >>> DomainAnalysis('interface-density').check().prepare([[1, 1, -1, -1]] * 4).run().density_.tolist()
        [0.25]
    """

    def __init__(
        self,
        kind: str = 'size',
        variables: Sequence[str] | None = None,
        *,
        method: str = 'moment',
        spacing: float = 1.0,
        fit_range: tuple[float, float] | None = None,
        times: Sequence[float] | None = None,
        attribute: str = 'state',
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.method = method
        self.spacing = spacing
        self.fit_range = fit_range
        self.times = times
        self.attribute = attribute

    def _check(self) -> Self:
        """Validate the kind, the method, the spacing, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('method', self.method, METHODS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.spacing <= 0:
            error_msg = f'{type(self).__name__} has spacing={self.spacing!r}. Use a positive spacing.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the field kind."""
        return DataKind.FIELD

    def _prepare(self, source: Source) -> Self:
        """Read the frames and their times from a trajectory, an array, or a graph."""
        if isinstance(source, nx.Graph):
            if self.kind != 'interface-density':
                error_msg = f'{type(self).__name__} of kind {self.kind!r} reads a field, not a graph.'
                raise AnalysisError(error_msg)
            self.frames_ = self._read_networks(source)
            self.times_ = np.zeros(1)
            return self
        name = self.variables[0] if self.variables else None
        frames = self._read_fields(source, name)
        if isinstance(source, Trajectory):
            self.frames_, self.times_ = frames, np.asarray(source.times, dtype=float)
        elif self.times is None:
            self.frames_, self.times_ = frames[None], np.zeros(1)
        else:
            self.frames_, self.times_ = frames, np.asarray(self.times, dtype=float)
        return self

    def _run(self) -> Self:
        """Compute the domain measure of every frame."""
        if self.kind in {'size', 'growth'}:
            self.size_ = np.asarray([self._domain_size(frame, self.method, self.spacing) for frame in self.frames_])
            if self.kind == 'growth':
                self.exponent_ = self._growth_exponent(self.times_, self.size_, self.fit_range)
            return self
        if self.kind == 'wavelength':
            patterns = np.asarray([self._pattern(frame, self.spacing) for frame in self.frames_]).reshape(-1, 3)
            self.wavelength_, self.wavenumber_, self.orientation_ = patterns.T
            return self
        self.density_ = np.asarray([self._interface_density(frame, self.attribute) for frame in self.frames_])
        return self

    @staticmethod
    def _domain_size(field: npt.ArrayLike, method: str, spacing: float) -> float:
        """Return `2 pi / <k>` over the structure factor of a frame, or the first zero of its correlation function."""
        if method == 'zero':
            distances, values = CorrelationAnalysis._correlation_function(field)
            crossing = np.flatnonzero(values <= 0)
            index = int(crossing[0]) if crossing.size else distances.size - 1
            return float(distances[index] * spacing)
        wavenumbers, power = SpectrumAnalysis._structure_factor(np.asarray(field, dtype=float)[None], spacing)
        return float(2 * np.pi * power.sum() / np.sum(wavenumbers * power))

    @staticmethod
    def _growth_exponent(times: Array, sizes: Array, fit_range: tuple[float, float] | None) -> float:
        """Return the slope of `log size` against `log time` over the frames of the fit range."""
        low, high = (0.0, np.inf) if fit_range is None else fit_range
        inside = (times >= low) & (times <= high) & (times > 0) & (sizes > 0)
        if inside.sum() < 2:  # noqa: PLR2004
            error_msg = (
                f'The fit range {fit_range} holds fewer than 2 frames with a positive time and size. Widen it, or '
                'prepare on more frames.'
            )
            raise AnalysisError(error_msg)
        return float(np.polyfit(np.log(times[inside]), np.log(sizes[inside]), 1)[0])

    @staticmethod
    def _pattern(field: npt.ArrayLike, spacing: float) -> tuple[float, float, float]:
        """Return the wavelength and the wavenumber of the structure factor peak and the order of its orientation."""
        grid = np.asarray(field, dtype=float)
        wavenumbers, shells = SpectrumAnalysis._structure_factor(grid[None], spacing)
        peak = float(wavenumbers[shells.argmax()])
        power = SpectrumAnalysis._mode_power(grid[None])
        axes = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(size, spacing) for size in grid.shape), indexing='ij')
        moduli = np.sqrt(sum(axis**2 for axis in axes))
        width = 2 * np.pi / (max(grid.shape) * spacing)
        near = np.abs(moduli - peak) <= width
        angles = np.arctan2(axes[1], axes[0]) if grid.ndim > 1 else np.zeros_like(moduli)
        weights = power[near]
        orientation = float(np.abs(np.sum(weights * np.exp(2j * angles[near]))) / weights.sum())
        return 2 * np.pi / peak, peak, orientation

    @staticmethod
    def _interface_density(data: npt.ArrayLike | nx.Graph, attribute: str = 'state') -> float:
        """Return the share of neighbouring pairs, on a periodic grid without corners or along edges, that differ."""
        own, other = NeighborAnalysis._pairs(data, attribute, diagonal=False)
        return float(np.mean(own != other))
