"""Compute one default analysis per kind of data of a trajectory, with curves, labels, and values ready to plot."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

import numpy as np

from ..engine.problems import Form
from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array
from ._alignment import AlignmentAnalysis
from ._base import BaseAnalysis, Signature, Source
from ._correlation import CorrelationAnalysis
from ._distribution import DistributionAnalysis
from ._kinds import DataKind
from ._neighbors import NeighborAnalysis
from ._networks import NetworkAnalysis
from ._spectrum import SpectrumAnalysis

SERIES_COLUMNS = 3
PLANE = 2


class SignatureAnalysis(BaseAnalysis):
    """Default analysis of every kind of data of a trajectory, with its curves, axis labels, title, and values.

    A series gives its power spectrum with the autocorrelation time, or its return map for a map. A jump path gives
    the survival function of its waiting times with the dispersion index of jumps per frame. A field gives the structure
    factor of its last frame with the peak wavelength or the correlation length. Agents give their polar and milling
    orders over time, or their nearest-neighbour distances against the Poisson law. A network gives its degree
    distribution.

    Attributes:
        kinds_:
            The data kinds of the trajectory, in the order agents, field, network, path, series.

        snapshots_:
            The agents of the first group at every frame, for the agents kind.

        frames_:
            The frames of the first lattice state, for the field kind.

        graphs_:
            The network of every frame, for the network kind.

        path_:
            The jump path of the first discrete state, for the path kind.

        jumps_:
            The size of the jumps of that state between frames, for the path kind.

        series_:
            The scalar series, for the series kind.

        discrete_:
            Whether the trajectory is that of a map, whose series give a return map.

        signatures_:
            One signature per data kind, in the order of `kinds_`, each with its kind, title, axis labels and scales,
            curves, and stated values.

    Examples:
        >>> from skcomplex.analysis import SignatureAnalysis
        >>> from skcomplex.models import run
        >>> trajectory = run('lotka_volterra', settings={'until': 20.0}).trajectory_
        >>> [signature.title for signature in SignatureAnalysis().check().prepare(trajectory).run().signatures_]
        ['Power spectrum']
    """

    def _get_data_kind(self) -> DataKind:
        """Return the system kind, since the analysis reads every kind of data of a trajectory."""
        return DataKind.SYSTEM

    def _prepare(self, source: Source) -> Self:
        """Read the data of every kind the trajectory holds."""
        if not isinstance(source, Trajectory):
            error_msg = f'{type(self).__name__} reads a trajectory, not {type(source).__name__}.'
            raise AnalysisError(error_msg)
        self.kinds_ = self._get_data_kinds(source)
        self.discrete_ = source.problem.form == Form.DISCRETE
        if DataKind.AGENTS in self.kinds_:
            self.snapshots_ = self._read_snapshots(source)
        if DataKind.FIELD in self.kinds_:
            self.frames_ = self._read_fields(source)
        if DataKind.NETWORK in self.kinds_:
            self.graphs_ = self._read_networks(source)
        if DataKind.PATH in self.kinds_:
            self.path_ = self._read_path(source)
            frames = self._read_series(source, [self.path_.label]).values
            self.jumps_ = np.abs(np.diff(frames, axis=0)).sum(axis=1)
        if DataKind.SERIES in self.kinds_:
            self.series_ = self._read_series(source)
        return self

    def _get_series_signature(self) -> Signature:
        """Return the power spectra of the series and the autocorrelation time, or the return map of a map."""
        columns = self.series_.values[:, :SERIES_COLUMNS]
        labels = self.series_.labels[:SERIES_COLUMNS]
        times = self.series_.times
        dt = float(np.median(np.diff(times))) if times.size > 1 else 1.0
        if self.discrete_:
            curves = {label: (column[:-1], column[1:]) for label, column in zip(labels, columns.T, strict=True)}
            return Signature(DataKind.SERIES, 'Return map', 'x(t)', 'x(t + 1)', curves, points=True)
        curves = {}
        for label, column in zip(labels, columns.T, strict=True):
            frequencies, density = SpectrumAnalysis._power_spectrum(column, dt)
            keep = (frequencies > 0) & (density > 0)
            curves[label] = (frequencies[keep], density[keep])
        values = {f'autocorrelation time of {labels[0]}': CorrelationAnalysis._autocorrelation_time(columns[:, 0], dt)}
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

    def _get_path_signature(self) -> Signature:
        """Return the survival function of the waiting times between jumps and the dispersion index of jump counts."""
        waiting = np.diff(self.path_.times)
        values = (
            {'dispersion index of jumps per frame': DistributionAnalysis._dispersion_index(self.jumps_)}
            if self.jumps_.mean() > 0
            else {}
        )
        values['jumps'] = float(self.path_.times.size - 1)
        curves = {'P(waiting time >= t)': DistributionAnalysis._ccdf(waiting)} if waiting.size else {}
        return Signature(
            DataKind.PATH,
            'Waiting times between jumps',
            'waiting time',
            'survival',
            curves,
            values,
            yscale='log',
        )

    def _get_field_signature(self) -> Signature:
        """Return the structure factor of the last frame with its peak wavelength or its correlation length."""
        last = self.frames_[-1]
        wavenumbers, power = SpectrumAnalysis._structure_factor(last[None])
        peak = int(np.argmax(power))
        values = (
            {'peak wavelength': float(2 * np.pi / wavenumbers[peak])}
            if 0 < peak < power.size - 1
            else {'correlation length': CorrelationAnalysis._correlation_length(last[None], 'moment', 1.0)}
        )
        return Signature(
            DataKind.FIELD,
            'Structure factor of the last frame',
            'wavenumber',
            'S(k)',
            {'S(k)': (wavenumbers, power)},
            values,
            xscale='log',
            yscale='log',
        )

    def _get_order_signature(self) -> Signature:
        """Return the polar order over time, and the milling order of agents with velocities in a plane."""
        snapshots = self.snapshots_
        if snapshots.velocities is not None:
            polar = AlignmentAnalysis._mean_vector_length(snapshots.velocities, vectors=True)
        else:
            polar = AlignmentAnalysis._mean_vector_length(np.asarray(snapshots.headings))
        curves = {'polar order': (snapshots.times, polar)}
        planar = snapshots.positions is not None and snapshots.positions.shape[-1] == PLANE
        if snapshots.velocities is not None and snapshots.positions is not None and planar:
            milling = AlignmentAnalysis._mean_rotation(snapshots.positions, snapshots.velocities)
            curves['milling order'] = (snapshots.times, milling)
        values = {name: float(curve[1][-1]) for name, curve in curves.items()}
        return Signature(DataKind.AGENTS, 'Order over time', 'time', 'order', curves, values)

    @staticmethod
    def _get_neighbor_signature(positions: Array, box: Array | None) -> Signature:
        """Return the survival function of nearest-neighbour distances, with the Poisson law of a plane in a box."""
        distances = NeighborAnalysis._nearest_neighbor_distances(positions, None if box is None else box.tolist())
        observed = DistributionAnalysis._ccdf(distances)
        curves = {'agents': observed}
        if box is not None and box.size == PLANE:
            density = positions.shape[0] / np.prod(box)
            curves['Poisson'] = (observed[0], np.exp(-density * np.pi * observed[0] ** 2))
        values = {'mean nearest-neighbour distance': float(distances.mean())}
        return Signature(DataKind.AGENTS, 'Nearest-neighbour distances', 'distance', 'P(distance >= r)', curves, values)

    def _get_agents_signature(self) -> Signature:
        """Return the polar and milling orders over time, or the nearest-neighbour distances against the Poisson law."""
        snapshots = self.snapshots_
        if snapshots.velocities is not None or snapshots.headings is not None:
            return self._get_order_signature()
        if snapshots.positions is None:
            counts = np.full(snapshots.times.size, float(snapshots.ids.size))
            return Signature(DataKind.AGENTS, 'Agents present', 'time', 'agents', {'agents': (snapshots.times, counts)})
        last = snapshots.positions[-1]
        if snapshots.box is not None and snapshots.periodic:
            return self._get_neighbor_signature(np.mod(last, snapshots.box), snapshots.box)
        return self._get_neighbor_signature(last, None)

    def _get_network_signature(self) -> Signature:
        """Return the degree distribution of the last network with its moments, clustering, and path length."""
        graph = self.graphs_[-1]
        values = NetworkAnalysis._degrees(graph).astype(float)
        summary = NetworkAnalysis._summary(graph)
        stated = {
            'mean degree': float(values.mean()),
            'second moment': float(np.mean(values**2)),
            'clustering': summary['clustering'],
            'mean path length': summary['path_length'],
        }
        positive = values[values > 0]
        curves = {'P(degree >= k)': DistributionAnalysis._ccdf(positive)} if positive.size else {}
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

    def _run(self, inputs: object = None) -> Self:
        """Compute the signature of every data kind of the trajectory."""
        signatures = {
            DataKind.AGENTS: self._get_agents_signature,
            DataKind.FIELD: self._get_field_signature,
            DataKind.NETWORK: self._get_network_signature,
            DataKind.PATH: self._get_path_signature,
            DataKind.SERIES: self._get_series_signature,
        }
        self.signatures_ = [signatures[kind]() for kind in self.kinds_]
        return self
