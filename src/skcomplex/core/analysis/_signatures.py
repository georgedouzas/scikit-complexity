"""Compute one default analysis per kind of data of a trajectory, with curves, labels, and values ready to plot."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import JumpPathSpec, SeriesSpec, SignatureSpec, SnapshotsSpec
from ..types import Array, DataKind, FormKind
from ._alignment import _mean_rotation, _mean_vector_length
from ._base import _get_data_kinds, _read_fields, _read_networks, _read_path, _read_series, _read_snapshots
from ._correlation import _autocorrelation_time, _correlation_length
from ._distribution import _ccdf, _dispersion_index
from ._neighbors import _nearest_neighbor_distances
from ._networks import _degrees, _summary
from ._spectrum import _power_spectrum, _structure_factor
from .types import Source

SERIES_COLUMNS = 3
PLANE = 2


@dataclass(frozen=True)
class SignaturesResult:
    """Default analysis of every kind of data of a trajectory, with its curves, axis labels, title, and values.

    Args:
        kinds:
            The data kinds of the trajectory, in the order agents, field, network, path, series.

        discrete:
            Whether the trajectory is that of a map, whose series give a return map.

        signatures:
            One signature per data kind, in the order of `kinds`, each with its kind, title, axis labels and scales,
            curves, and stated values.

        snapshots:
            The agents of the first group at every frame, for the agents kind.

        frames:
            The frames of the first lattice state, for the field kind.

        graphs:
            The network of every frame, for the network kind.

        path:
            The jump path of the first discrete state, for the path kind.

        jumps:
            The size of the jumps of that state between frames, for the path kind.

        series:
            The scalar series, for the series kind.
    """

    kinds: list[DataKind]
    discrete: bool
    signatures: list[SignatureSpec]
    snapshots: SnapshotsSpec | None = None
    frames: Array | None = None
    graphs: list[nx.Graph] | None = None
    path: JumpPathSpec | None = None
    jumps: Array | None = None
    series: SeriesSpec | None = None


def signatures(source: Source) -> SignaturesResult:
    """Return the default analysis of every kind of data of a trajectory, with its curves, axis labels, and values.

    A series gives its power spectrum with the autocorrelation time, or its return map for a map. A jump path gives
    the survival function of its waiting times with the dispersion index of jumps per frame. A field gives the structure
    factor of its last frame with the peak wavelength or the correlation length. Agents give their polar and milling
    orders over time, or their nearest-neighbour distances against the Poisson law. A network gives its degree
    distribution.

    Args:
        source:
            The trajectory.

    Returns:
        result:
            The data of every kind and one signature per kind.

    Raises:
        AnalysisError:
            The source is not a trajectory.

    Examples:
        >>> from skcomplex.core.analysis import signatures
        >>> from skcomplex.systems import create_simulation
        >>> trajectory = create_simulation('lotka_volterra').prepare().run(until=20.0).trajectory_
        >>> [signature.title for signature in signatures(trajectory).signatures]
        ['Power spectrum']
    """
    if not isinstance(source, Trajectory):
        error_msg = f'signatures reads a trajectory, not {type(source).__name__}.'
        raise AnalysisError(error_msg)
    kinds = _get_data_kinds(source)
    discrete = source.problem.form == FormKind.DISCRETE
    snapshots = _read_snapshots(source) if DataKind.AGENTS in kinds else None
    frames = _read_fields(source) if DataKind.FIELD in kinds else None
    graphs = _read_networks(source) if DataKind.NETWORK in kinds else None
    path = jumps = None
    if DataKind.PATH in kinds:
        path = _read_path(source)
        values = _read_series(source, [path.label]).values
        jumps = np.abs(np.diff(values, axis=0)).sum(axis=1)
    series = _read_series(source) if DataKind.SERIES in kinds else None
    found = []
    for kind in kinds:
        if kind == DataKind.AGENTS and snapshots is not None:
            found.append(_get_agents_signature(snapshots))
        elif kind == DataKind.FIELD and frames is not None:
            found.append(_get_field_signature(frames))
        elif kind == DataKind.NETWORK and graphs is not None:
            found.append(_get_network_signature(graphs))
        elif kind == DataKind.PATH and path is not None and jumps is not None:
            found.append(_get_path_signature(path, jumps))
        elif series is not None:
            found.append(_get_series_signature(series, discrete=discrete))
    return SignaturesResult(kinds, discrete, found, snapshots, frames, graphs, path, jumps, series)


def _get_series_signature(series: SeriesSpec, *, discrete: bool) -> SignatureSpec:
    """Return the power spectra of the series and the autocorrelation time, or the return map of a map."""
    columns = series.values[:, :SERIES_COLUMNS]
    labels = series.labels[:SERIES_COLUMNS]
    times = series.times
    dt = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    if discrete:
        curves = {label: (column[:-1], column[1:]) for label, column in zip(labels, columns.T, strict=True)}
        return SignatureSpec(DataKind.SERIES, 'Return map', 'x(t)', 'x(t + 1)', curves, points=True)
    curves = {}
    for label, column in zip(labels, columns.T, strict=True):
        frequencies, density = _power_spectrum(column, dt)
        keep = (frequencies > 0) & (density > 0)
        curves[label] = (frequencies[keep], density[keep])
    values = {f'autocorrelation time of {labels[0]}': _autocorrelation_time(columns[:, 0], dt)}
    return SignatureSpec(
        DataKind.SERIES,
        'Power spectrum',
        'frequency',
        'power spectral density',
        curves,
        values,
        xscale='log',
        yscale='log',
    )


def _get_path_signature(path: JumpPathSpec, jumps: Array) -> SignatureSpec:
    """Return the survival function of the waiting times between jumps and the dispersion index of jump counts."""
    waiting = np.diff(path.times)
    values = {'dispersion index of jumps per frame': _dispersion_index(jumps)} if jumps.mean() > 0 else {}
    values['jumps'] = float(path.times.size - 1)
    curves = {'P(waiting time >= t)': _ccdf(waiting)} if waiting.size else {}
    return SignatureSpec(
        DataKind.PATH,
        'Waiting times between jumps',
        'waiting time',
        'survival',
        curves,
        values,
        yscale='log',
    )


def _get_field_signature(frames: Array) -> SignatureSpec:
    """Return the structure factor of the last frame with its peak wavelength or its correlation length."""
    last = frames[-1]
    wavenumbers, power = _structure_factor(last[None])
    peak = int(np.argmax(power))
    values = (
        {'peak wavelength': float(2 * np.pi / wavenumbers[peak])}
        if 0 < peak < power.size - 1
        else {'correlation length': _correlation_length(last[None], 'moment', 1.0)}
    )
    return SignatureSpec(
        DataKind.FIELD,
        'Structure factor of the last frame',
        'wavenumber',
        'S(k)',
        {'S(k)': (wavenumbers, power)},
        values,
        xscale='log',
        yscale='log',
    )


def _get_order_signature(snapshots: SnapshotsSpec) -> SignatureSpec:
    """Return the polar order over time, and the milling order of agents with velocities in a plane."""
    if snapshots.velocities is not None:
        polar = _mean_vector_length(snapshots.velocities, vectors=True)
    else:
        angles = snapshots.directions if snapshots.directions is not None else snapshots.phases
        polar = _mean_vector_length(np.asarray(angles))
    curves = {'polar order': (snapshots.times, polar)}
    planar = snapshots.positions is not None and snapshots.positions.shape[-1] == PLANE
    if snapshots.velocities is not None and snapshots.positions is not None and planar:
        milling = _mean_rotation(snapshots.positions, snapshots.velocities)
        curves['milling order'] = (snapshots.times, milling)
    values = {name: float(curve[1][-1]) for name, curve in curves.items()}
    return SignatureSpec(DataKind.AGENTS, 'Order over time', 'time', 'order', curves, values)


def _get_neighbor_signature(positions: Array, box: Array | None) -> SignatureSpec:
    """Return the survival function of nearest-neighbour distances, with the Poisson law of a plane in a box."""
    distances = _nearest_neighbor_distances(positions, None if box is None else box.tolist())
    observed = _ccdf(distances)
    curves = {'agents': observed}
    if box is not None and box.size == PLANE:
        density = positions.shape[0] / np.prod(box)
        curves['Poisson'] = (observed[0], np.exp(-density * np.pi * observed[0] ** 2))
    values = {'mean nearest-neighbour distance': float(distances.mean())}
    return SignatureSpec(
        DataKind.AGENTS,
        'Nearest-neighbour distances',
        'distance',
        'P(distance >= r)',
        curves,
        values,
    )


def _get_agents_signature(snapshots: SnapshotsSpec) -> SignatureSpec:
    """Return the polar and milling orders over time, or the nearest-neighbour distances against the Poisson law."""
    if snapshots.velocities is not None or snapshots.directions is not None or snapshots.phases is not None:
        return _get_order_signature(snapshots)
    if snapshots.positions is None:
        counts = np.full(snapshots.times.size, float(snapshots.ids.size))
        return SignatureSpec(
            DataKind.AGENTS,
            'Agents present',
            'time',
            'agents',
            {'agents': (snapshots.times, counts)},
        )
    last = snapshots.positions[-1]
    if snapshots.box is not None and snapshots.periodic:
        return _get_neighbor_signature(np.mod(last, snapshots.box), snapshots.box)
    return _get_neighbor_signature(last, None)


def _get_network_signature(graphs: list[nx.Graph]) -> SignatureSpec:
    """Return the degree distribution of the last network with its moments, clustering, and path length."""
    graph = graphs[-1]
    values = _degrees(graph).astype(float)
    summary = _summary(graph)
    stated = {
        'mean degree': float(values.mean()),
        'second moment': float(np.mean(values**2)),
        'clustering': summary['clustering'],
        'mean path length': summary['path_length'],
    }
    positive = values[values > 0]
    curves = {'P(degree >= k)': _ccdf(positive)} if positive.size else {}
    return SignatureSpec(
        DataKind.NETWORK,
        'Degree distribution',
        'degree',
        'P(degree >= k)',
        curves,
        stated,
        xscale='log',
        yscale='log',
    )
