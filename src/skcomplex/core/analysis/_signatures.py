"""Compute one default analysis per kind of data, with curves, labels, and values ready to plot."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..base import Form
from ..experiments import ExperimentResult
from ..runtime import Trajectory
from ._adapters import Snapshots, data_kinds, ensemble_of, fields_of, networks_of, path_of, series_of, snapshots_of
from ._agents import milling_order, nearest_neighbor_distances, polar_order
from ._fields import correlation_length, structure_factor
from ._fluctuations import fano_factor
from ._kinds import DataKind
from ._networks import degrees, summarize_network
from ._series import autocorrelation_time, power_spectrum
from ._statistics import ccdf

type Array = npt.NDArray[np.float64]
type Curve = tuple[Array, Array]
SERIES_COLUMNS = 3
QUANTILES = (0.05, 0.95)
PLANE = 2


@dataclass(frozen=True)
class Signature:
    """Default analysis of one kind of data, ready to plot.

    Args:
        kind:
            The kind of data analysed.

        title:
            The title of the figure.

        xlabel:
            The label of the horizontal axis.

        ylabel:
            The label of the vertical axis.

        curves:
            A mapping from label to the abscissas and ordinates of every curve.

        values:
            The measured values worth stating, by name.

        xscale:
            `'linear'` or `'log'`.

        yscale:
            `'linear'` or `'log'`.

        points:
            Whether the curves are drawn as points rather than lines.

        band:
            The abscissas and the lower and upper edges of a shaded band, or `None`.
    """

    kind: DataKind
    title: str
    xlabel: str
    ylabel: str
    curves: dict[str, Curve]
    values: dict[str, float] = field(default_factory=dict)
    xscale: str = 'linear'
    yscale: str = 'linear'
    points: bool = False
    band: tuple[Array, Array, Array] | None = None


def _series_signature(trajectory: Trajectory) -> Signature:
    """Return the power spectra of the series and the autocorrelation time, or the return map of a map."""
    series = series_of(trajectory)
    columns = series.values[:, :SERIES_COLUMNS]
    labels = series.labels[:SERIES_COLUMNS]
    dt = float(np.median(np.diff(series.times))) if series.times.size > 1 else 1.0
    if trajectory.problem.form is Form.DISCRETE:
        curves = {label: (column[:-1], column[1:]) for label, column in zip(labels, columns.T, strict=True)}
        return Signature(DataKind.SERIES, 'Return map', 'x(t)', 'x(t + 1)', curves, points=True)
    curves = {}
    for label, column in zip(labels, columns.T, strict=True):
        frequencies, density = power_spectrum(column, dt)
        keep = (frequencies > 0) & (density > 0)
        curves[label] = (frequencies[keep], density[keep])
    values = {f'autocorrelation time of {labels[0]}': autocorrelation_time(columns[:, 0], dt)}
    return Signature(
        DataKind.SERIES,
        'Power spectrum',
        'frequency',
        'power spectral density',
        curves,
        values,
        xscale='log',
        yscale='log',
    )


def _path_signature(trajectory: Trajectory) -> Signature:
    """Return the survival function of the waiting times between jumps and the Fano factor of jump counts."""
    path = path_of(trajectory)
    waiting = np.diff(path.times)
    states = path.states.reshape(path.times.size, -1)
    frames = np.asarray(trajectory[path.label], dtype=float).reshape(len(trajectory), -1)
    jumps = np.abs(np.diff(frames, axis=0)).sum(axis=1)
    values = {'Fano factor of jumps per frame': fano_factor(jumps)} if jumps.mean() > 0 else {}
    values['jumps'] = float(states.shape[0] - 1)
    curves = {'P(waiting time >= t)': ccdf(waiting)} if waiting.size else {}
    return Signature(
        DataKind.PATH,
        'Waiting times between jumps',
        'waiting time',
        'survival',
        curves,
        values,
        yscale='log',
    )


def _field_signature(trajectory: Trajectory) -> Signature:
    """Return the structure factor of the last frame with its peak wavelength or its correlation length."""
    last = fields_of(trajectory)[-1]
    factor = structure_factor(last)
    peak = int(np.argmax(factor.power))
    values = (
        {'peak wavelength': float(2 * np.pi / factor.wavenumbers[peak])}
        if 0 < peak < factor.power.size - 1
        else {'correlation length': correlation_length(last)}
    )
    return Signature(
        DataKind.FIELD,
        'Structure factor of the last frame',
        'wavenumber',
        'S(k)',
        {'S(k)': (factor.wavenumbers, factor.power)},
        values,
        xscale='log',
        yscale='log',
    )


def _order_signature(snapshots: Snapshots) -> Signature:
    """Return the polar order over time, and the milling order of agents with velocities in a plane."""
    if snapshots.velocities is not None:
        polar = np.asarray(polar_order(snapshots.velocities, vectors=True))
    else:
        polar = np.asarray(polar_order(np.asarray(snapshots.headings)))
    curves = {'polar order': (snapshots.times, polar)}
    planar = snapshots.positions is not None and snapshots.positions.shape[-1] == PLANE
    if snapshots.velocities is not None and snapshots.positions is not None and planar:
        milling = np.asarray(milling_order(snapshots.positions, snapshots.velocities))
        curves['milling order'] = (snapshots.times, milling)
    values = {name: float(curve[1][-1]) for name, curve in curves.items()}
    return Signature(DataKind.AGENTS, 'Order over time', 'time', 'order', curves, values)


def _neighbor_signature(positions: Array, box: Array | None) -> Signature:
    """Return the survival function of nearest-neighbour distances, with the Poisson law of a plane in a box."""
    distances = nearest_neighbor_distances(positions, None if box is None else box.tolist())
    observed = ccdf(distances)
    curves = {'agents': observed}
    if box is not None and box.size == PLANE:
        density = positions.shape[0] / np.prod(box)
        curves['Poisson'] = (observed[0], np.exp(-density * np.pi * observed[0] ** 2))
    values = {'mean nearest-neighbour distance': float(distances.mean())}
    return Signature(DataKind.AGENTS, 'Nearest-neighbour distances', 'distance', 'P(distance >= r)', curves, values)


def _agents_signature(trajectory: Trajectory) -> Signature:
    """Return the polar and milling orders over time, or the nearest-neighbour distances against the Poisson law."""
    snapshots = snapshots_of(trajectory)
    if snapshots.velocities is not None or snapshots.headings is not None:
        return _order_signature(snapshots)
    if snapshots.positions is None:
        counts = np.full(snapshots.times.size, float(snapshots.ids.size))
        return Signature(DataKind.AGENTS, 'Agents present', 'time', 'agents', {'agents': (snapshots.times, counts)})
    last = snapshots.positions[-1]
    if snapshots.box is not None and snapshots.periodic:
        return _neighbor_signature(np.mod(last, snapshots.box), snapshots.box)
    return _neighbor_signature(last, None)


def _network_signature(trajectory: Trajectory) -> Signature:
    """Return the degree distribution of the last network with its moments, clustering, and path length."""
    graph = networks_of(trajectory)[-1]
    values = degrees(graph).astype(float)
    summary = summarize_network(graph)
    stated = {
        'mean degree': float(values.mean()),
        'second moment': float(np.mean(values**2)),
        'clustering': summary.clustering,
        'mean path length': summary.path_length,
    }
    positive = values[values > 0]
    curves = {'P(degree >= k)': ccdf(positive)} if positive.size else {}
    return Signature(
        DataKind.NETWORK,
        'Degree distribution',
        'degree',
        'P(degree >= k)',
        curves,
        stated,
        xscale='log',
        yscale='log',
    )


def _ensemble_signature(result: ExperimentResult) -> list[Signature]:
    """Return the mean and the 90 percent band over runs of the first kept series."""
    names = list(result.runs[0].series) if result.runs else []
    if not names:
        return []
    ensemble = ensemble_of(result, names[0])
    low, high = np.quantile(ensemble.values, QUANTILES, axis=0)
    mean = ensemble.values.mean(axis=0)
    return [
        Signature(
            DataKind.ENSEMBLE,
            f'{names[0]} over {ensemble.values.shape[0]} runs',
            'time',
            names[0],
            {'mean': (ensemble.times, mean)},
            {'runs': float(ensemble.values.shape[0])},
            band=(ensemble.times, low, high),
        ),
    ]


SIGNATURES = {
    DataKind.AGENTS: _agents_signature,
    DataKind.FIELD: _field_signature,
    DataKind.NETWORK: _network_signature,
    DataKind.PATH: _path_signature,
    DataKind.SERIES: _series_signature,
}


def signature_analysis(source: Trajectory | ExperimentResult) -> list[Signature]:
    """Return one default analysis per kind of data of a trajectory or an experiment result.

    A series gives its power spectrum with the autocorrelation time, or its return map for a map. A jump path gives
    the survival function of its waiting times with the Fano factor of jumps per frame. A field gives the structure
    factor of its last frame with the peak wavelength or the correlation length. Agents give their polar and milling
    orders over time, or their nearest-neighbour distances against the Poisson law. A network gives its degree
    distribution, and an ensemble the mean with a 90 percent band over runs.

    Args:
        source:
            The trajectory or the experiment result.

    Returns:
        signatures:
            One signature per data kind, in the order agents, field, network, path, series, ensemble.

    Examples:
        >>> from skcomplex.core.analysis import signature_analysis
        >>> from skcomplex.models import run
        >>> trajectory = run('lotka_volterra', settings={'until': 20.0}).trajectory_
        >>> [signature.title for signature in signature_analysis(trajectory)]
        ['Power spectrum']
    """
    if isinstance(source, ExperimentResult):
        return _ensemble_signature(source)
    return [SIGNATURES[kind](source) for kind in data_kinds(source) if kind in SIGNATURES]
