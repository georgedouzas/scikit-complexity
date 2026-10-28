"""Test the base of analyses: its verbs and the readers of trajectories, arrays, and graphs."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.analysis import BaseAnalysis, DataKind
from skcomplex.core.errors import AnalysisError, NotCheckedError
from skcomplex.models import run


class MeanAnalysis(BaseAnalysis):
    """Mean of every series of a source."""

    def __init__(self, variables=None):
        self.variables = variables

    def _get_data_kind(self):
        return DataKind.SERIES

    def _prepare(self, source):
        self.series_ = self._read_series(source, self.variables)
        return self

    def _run(self):
        self.mean_ = self.series_.values.mean(axis=0)
        return self


class DegreeAnalysis(BaseAnalysis):
    """Mean degree of the network of every frame."""

    def _get_data_kind(self):
        return DataKind.NETWORK

    def _prepare(self, source):
        self.graphs_ = self._read_networks(source)
        return self

    def _run(self):
        self.degree_ = np.asarray([2 * graph.number_of_edges() / graph.number_of_nodes() for graph in self.graphs_])
        return self


class AgentAnalysis(BaseAnalysis):
    """Number of agents of the first frame."""

    def _get_data_kind(self):
        return DataKind.AGENTS

    def _prepare(self, source):
        self.snapshots_ = self._read_snapshots(source)
        return self

    def _run(self):
        self.count_ = self.snapshots_.ids.size
        return self


def test_analysis_reads_a_trajectory():
    """Test that the series of a trajectory are read by name."""
    trajectory = run('lotka_volterra', settings={'until': 1.0}).trajectory_
    analysis = MeanAnalysis(['prey.n']).check().prepare(trajectory).run()
    assert analysis.series_.labels == ['prey.n']
    assert analysis.mean_.tolist() == pytest.approx([trajectory.values('prey.n').mean()])


def test_analysis_reads_arrays_as_columns():
    """Test that a series is one column and a two-dimensional array is one series per column."""
    assert MeanAnalysis().check().prepare([1.0, 3.0]).run().mean_.tolist() == [2.0]
    assert MeanAnalysis().check().prepare(np.array([[1.0, 2.0], [3.0, 4.0]])).run().mean_.tolist() == [2.0, 3.0]


def test_analysis_reads_a_graph():
    """Test that a graph is the only frame of a network analysis."""
    assert DegreeAnalysis().check().prepare(nx.cycle_graph(5)).run().degree_.tolist() == [2.0]


def test_missing_data_kind_names_the_kinds():
    """Test that reading agents from an ODE trajectory names the data kinds it has."""
    trajectory = run('lotka_volterra', settings={'until': 1.0}).trajectory_
    with pytest.raises(AnalysisError, match='no agent group'):
        AgentAnalysis().check().prepare(trajectory)


def test_verbs_run_in_order():
    """Test that prepare needs check and run needs prepare."""
    with pytest.raises(NotCheckedError):
        MeanAnalysis().prepare([1.0])
    with pytest.raises(AnalysisError, match='prepared with a source'):
        MeanAnalysis().check().run()
