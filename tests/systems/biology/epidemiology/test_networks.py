"""Test the `_networks` module of `skcomplex.systems.biology.epidemiology`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.systems.biology.epidemiology import HeterogeneousMeanFieldSISSystem, MetapopulationSystem

from ..conftest import _run_until


def test_heterogeneous_mean_field_threshold():
    """Test that the degree classes are group members and that SIS dies out below `<k> / <k^2>` and persists above."""
    degrees = np.arange(3, 31, dtype=float)
    weights = degrees**-2.5 / np.sum(degrees**-2.5)
    threshold = (degrees @ weights) / (degrees**2 @ weights)
    classes = HeterogeneousMeanFieldSISSystem(lam=0.5 * threshold).prepare().system_.elements[0]
    assert classes.size == degrees.size
    np.testing.assert_allclose(classes.values['k'], degrees)
    np.testing.assert_allclose(classes.values['P'], weights)
    below = _run_until(HeterogeneousMeanFieldSISSystem(lam=0.5 * threshold), 100, dt=1.0)
    above = _run_until(HeterogeneousMeanFieldSISSystem(lam=2 * threshold), 100, dt=1.0)
    assert below.get_values('network.rho')[-1].max() < 1e-6
    assert above.get_values('network.rho')[-1].min() > 1e-3
    assert np.all(np.diff(above.get_values('network.rho')[-1]) > 0)


def test_metapopulation_conserves_individuals_and_spreads():
    """Test that travellers conserve the total population and carry the epidemic to every node."""
    trajectory = _run_until(MetapopulationSystem(nx.path_graph(5), D=0.05), 200, dt=1.0)
    totals = trajectory.get_values('cities.S') + trajectory.get_values('cities.I') + trajectory.get_values('cities.R')
    assert totals.sum(axis=1) == pytest.approx(np.full(len(totals), 5000.0))
    assert trajectory.get_values('cities.R')[-1].min() > 100
