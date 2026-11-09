"""Test the `_hypergraphs` module of `skcomplex.systems.physics.synchronization`."""

import numpy as np
import pytest

from skcomplex.systems.physics.synchronization import HypergraphKuramotoSystem

from ..conftest import _run_observed
from .conftest import _random_drift


def _final_order(K2: float, phases: np.ndarray | None) -> float:
    """Return the order parameter over the last quarter of a run of 40 oscillators coupled by triangles."""
    system = HypergraphKuramotoSystem(40, K2=K2, phases=phases)
    order = np.asarray(_run_observed(system, 40.0, 0.5).get_values('order'))
    return float(order[-20:].mean())


def test_hypergraph_kuramoto_drift_matches_the_simplex_coupling():
    """Test that the drift of the triangle coupling equals a direct loop over the triangles and links."""
    system = HypergraphKuramotoSystem(8, K1=1.5, K2=2.0, links=0.5, triangles=0.5).prepare().system_
    problem, state, drift = _random_drift(system)
    space = system.elements[0].space.prepare()
    coupling = system.interactions[0]
    omega = np.asarray(system.elements[0].values['omega'])
    expected = omega.copy()
    for hyperedge in space.hyperedges_:
        for member in hyperedge:
            others = [state[other] for other in hyperedge if other != member]
            strength = coupling.K1 if len(others) == 1 else coupling.K2
            expected[member] += strength * np.sin(sum(others) - len(others) * state[member])
    assert np.allclose(problem.states.read(drift, 'oscillators.theta'), expected, atol=1e-12)


def test_hypergraph_kuramoto_shows_hysteresis_in_the_three_body_coupling():
    """Test that strong triangle coupling keeps incoherent and synchronized phases, and weak coupling loses sync."""
    synchronized = np.zeros(40)
    assert _final_order(8.0, None) < 0.3
    assert _final_order(8.0, synchronized) > 0.9
    assert _final_order(1.0, synchronized) < 0.3


@pytest.mark.parametrize(('links', 'triangles'), [(1.0, 1.0), (0.3, 0.1)])
def test_hypergraph_kuramoto_builds_its_hypergraph(links, triangles):
    """Test that the hypergraph holds links and triangles at the given densities."""
    space = HypergraphKuramotoSystem(12, links=links, triangles=triangles).prepare().system_.elements[0].space.prepare()
    sizes = [len(hyperedge) for hyperedge in space.hyperedges_]
    assert sizes.count(2) == pytest.approx(66 * links, abs=12)
    assert sizes.count(3) == pytest.approx(220 * triangles, abs=25)
