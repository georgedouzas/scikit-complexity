"""Measure the domains of fields: their size, its growth in time, pattern wavelengths, and interface densities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _check_choice, _check_settings, _read_fields, _read_networks
from ._correlation import _correlation_function
from ._neighbors import _pairs
from ._spectrum import _mode_power, _structure_factor
from .types import Source

KINDS = ('size', 'growth', 'wavelength', 'interface-density')
METHODS = ('moment', 'zero')
SETTINGS = {
    'size': {'method', 'spacing', 'times'},
    'growth': {'method', 'spacing', 'fit_range', 'times'},
    'wavelength': {'spacing', 'times'},
    'interface-density': {'attribute', 'times'},
}
"""Settings every kind reads, besides `kind` and `variables`."""

MIN_LINE_POINTS = 2

TAGS = AnalysisTags(DataKind.FIELD)


@dataclass(frozen=True)
class DomainsResult:
    """Domains of every frame of a field, or of the states on a network.

    Args:
        frames:
            The frames of the field, or the graph.

        times:
            The time of every frame.

        size:
            The domain size of every frame, for `'size'` and `'growth'`.

        exponent:
            The growth exponent of the domain size, for `'growth'`.

        wavelength:
            The wavelength of the peak of every frame, for `'wavelength'`.

        wavenumber:
            The wavenumber of the peak of every frame, for `'wavelength'`.

        orientation:
            The orientation order of the wave vectors near the peak of every frame, for `'wavelength'`.

        density:
            The interface density of every frame, for `'interface-density'`.
    """

    frames: list[nx.Graph] | Array
    times: Array
    size: Array | None = None
    exponent: float | None = None
    wavelength: Array | None = None
    wavenumber: Array | None = None
    orientation: Array | None = None
    density: Array | None = None


def domains(
    source: Source,
    *,
    kind: str = 'size',
    variables: Sequence[str] | None = None,
    method: str = 'moment',
    spacing: float = 1.0,
    fit_range: tuple[float, float] | None = None,
    times: Sequence[float] | None = None,
    attribute: str = 'state',
) -> DomainsResult:
    """Return the domains of every frame of a field, or of the states on a network, of one kind.

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
        source:
            The trajectory, an array of frames, or a graph.

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

    Returns:
        result:
            The frames, their times, and the domain measure of every frame.

    Raises:
        AnalysisError:
            The kind or method is unknown, a setting the kind does not read is changed, the spacing is not positive,
            or a kind that reads a field is given a graph.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import domains
        >>> x = np.arange(256)
        >>> stripes = np.sign(np.cos(2 * np.pi * x / 16))
        >>> domains(stripes[:64], method='zero').size.tolist()
        [4.0]
        >>> stack = [np.sign(np.cos(2 * np.pi * x / (8 * t))) for t in (1, 2, 4)]
        >>> round(domains(stack, kind='growth', method='zero', times=[1, 2, 4]).exponent, 2)
        1.0
        >>> domains([[1, 1, -1, -1]] * 4, kind='interface-density').density.tolist()
        [0.25]
    """
    _check_choice('domains', 'kind', kind, KINDS)
    _check_choice('domains', 'method', method, METHODS)
    _check_settings(
        domains,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        method=method,
        spacing=spacing,
        fit_range=fit_range,
        times=times,
        attribute=attribute,
    )
    if spacing <= 0:
        error_msg = f'domains has spacing={spacing!r}. Use a positive spacing.'
        raise AnalysisError(error_msg)
    frames, frame_times = _read_frames(source, kind, variables, times)
    if kind in {'size', 'growth'}:
        size = np.asarray([_domain_size(frame, method, spacing) for frame in frames])
        exponent = _growth_exponent(frame_times, size, fit_range) if kind == 'growth' else None
        return DomainsResult(frames, frame_times, size=size, exponent=exponent)
    if kind == 'wavelength':
        patterns = np.asarray([_pattern(frame, spacing) for frame in frames]).reshape(-1, 3)
        wavelength, wavenumber, orientation = patterns.T
        return DomainsResult(
            frames,
            frame_times,
            wavelength=wavelength,
            wavenumber=wavenumber,
            orientation=orientation,
        )
    density = np.asarray([_interface_density(frame, attribute) for frame in frames])
    return DomainsResult(frames, frame_times, density=density)


def _read_frames(
    source: Source,
    kind: str,
    variables: Sequence[str] | None,
    times: Sequence[float] | None,
) -> tuple[list[nx.Graph] | Array, Array]:
    """Return the frames and their times from a trajectory, an array, or a graph."""
    if isinstance(source, nx.Graph):
        if kind != 'interface-density':
            error_msg = f'domains of kind {kind!r} reads a field, not a graph.'
            raise AnalysisError(error_msg)
        return _read_networks(source), np.zeros(1)
    name = variables[0] if variables else None
    frames = _read_fields(source, name)
    if isinstance(source, Trajectory):
        return frames, np.asarray(source.times, dtype=float)
    if times is None:
        return frames[None], np.zeros(1)
    return frames, np.asarray(times, dtype=float)


def _domain_size(field: npt.ArrayLike, method: str, spacing: float) -> float:
    """Return `2 pi / <k>` over the structure factor of a frame, or the first zero of its correlation function."""
    if method == 'zero':
        distances, values = _correlation_function(field)
        crossing = np.flatnonzero(values <= 0)
        index = int(crossing[0]) if crossing.size else distances.size - 1
        return float(distances[index] * spacing)
    wavenumbers, power = _structure_factor(np.asarray(field, dtype=float)[None], spacing)
    return float(2 * np.pi * power.sum() / np.sum(wavenumbers * power))


def _growth_exponent(times: Array, sizes: Array, fit_range: tuple[float, float] | None) -> float:
    """Return the slope of `log size` against `log time` over the frames of the fit range."""
    low, high = (0.0, np.inf) if fit_range is None else fit_range
    inside = (times >= low) & (times <= high) & (times > 0) & (sizes > 0)
    if inside.sum() < MIN_LINE_POINTS:
        error_msg = (
            f'The fit range {fit_range} holds fewer than 2 frames with a positive time and size. Widen it, or '
            'prepare on more frames.'
        )
        raise AnalysisError(error_msg)
    return float(np.polyfit(np.log(times[inside]), np.log(sizes[inside]), 1)[0])


def _pattern(field: npt.ArrayLike, spacing: float) -> tuple[float, float, float]:
    """Return the wavelength and the wavenumber of the structure factor peak and the order of its orientation."""
    grid = np.asarray(field, dtype=float)
    wavenumbers, shells = _structure_factor(grid[None], spacing)
    peak = float(wavenumbers[shells.argmax()])
    power = _mode_power(grid[None])
    axes = np.meshgrid(*(2 * np.pi * np.fft.fftfreq(size, spacing) for size in grid.shape), indexing='ij')
    moduli = np.sqrt(sum(axis**2 for axis in axes))
    width = 2 * np.pi / (max(grid.shape) * spacing)
    near = np.abs(moduli - peak) <= width
    angles = np.arctan2(axes[1], axes[0]) if grid.ndim > 1 else np.zeros_like(moduli)
    weights = power[near]
    orientation = float(np.abs(np.sum(weights * np.exp(2j * angles[near]))) / weights.sum())
    return 2 * np.pi / peak, peak, orientation


def _interface_density(data: npt.ArrayLike | nx.Graph, attribute: str = 'state') -> float:
    """Return the share of neighbouring pairs, on a periodic grid without corners or along edges, that differ."""
    own, other = _pairs(data, attribute, diagonal=False)
    return float(np.mean(own != other))
