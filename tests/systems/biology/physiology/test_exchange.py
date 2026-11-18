"""Test the `_exchange` module of `skcomplex.systems.biology.physiology`."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import hill, laplacian
from skcomplex.core.specs import Role
from skcomplex.core.systems import System
from skcomplex.systems.biology.physiology import (
    ClosedContourCellExchangeSystem,
    ContourCellElement,
    ContourUptakeInteraction,
    ExchangingCellElement,
    NanotubeGapJunctionExchangeSystem,
)


class _DegreeCellElement(ExchangingCellElement):
    """Cells that also hold the degree of their node."""

    def __init__(self, label, *, x=0.0, alpha=1.0, beta=10.0, g=0.1, xi=0.0, degree=0.0, space=None):
        super().__init__(label, x=x, alpha=alpha, beta=beta, g=g, xi=xi, space=space)
        self.degree = degree


class _PreviousExchangeInteraction(Interaction):
    """Metabolism and exchange in one law through the graph Laplacian, as before the coupling interaction."""

    def __init__(self, label, *, target: Role):
        super().__init__(label)
        self.target = target

    def _prepare(self, context=None):
        """Set the metabolism of the substance and its exchange through gap junctions and nanotubes."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            exchange = (s.g + s.xi) * laplacian(s.x) + s.xi * s.degree * s.x
            self.laws_ = [RateLaw(s.x, -s.alpha * hill(s.x, s.beta) + exchange)]


def test_closed_contour_cell_exchange_synchronizes_in_the_stability_domain():
    """Test that the contour cells synchronize at a stable equilibrium in its domain, and a chaotic one does not."""
    stable = ClosedContourCellExchangeSystem(r=2.5).prepare().simulate(dt=1, until=200).get_values('cells.x')
    chaotic = ClosedContourCellExchangeSystem(r=3.9).prepare().simulate(dt=1, until=200).get_values('cells.x')
    equilibrium = stable[-1]
    assert np.ptp(stable[-20:], axis=0).max() < 1e-9
    assert np.ptp(equilibrium) < 1e-9
    assert (2.5 - 1) / 5 < equilibrium[0] < (2.5 + 1) / 5
    assert equilibrium[0] > 0.5 ** (1 / (1 - 0.5))
    assert np.ptp(chaotic[-20:], axis=1).max() > 0.1


def test_nanotube_gap_junction_exchange_destabilizes_past_the_threshold():
    """Test that the substance dies out below the nanotube threshold alpha / (beta k) and grows above it."""
    graph = nx.random_regular_graph(4, 30, seed=0)
    critical = 1.0 / (10.0 * 4)
    final = {}
    for share in (0.8, 1.2):
        system = NanotubeGapJunctionExchangeSystem(graph, xi=share * critical, x=0.1).prepare().system_
        final[share] = system.prepare().simulate(dt=0.5, until=200).get_values('cells.x')[-1].mean()
    assert final[0.8] < 0.01
    assert final[1.2] > 10


@pytest.mark.parametrize('xi', [0.0, 0.01])
def test_nanotube_gap_junction_exchange_keeps_the_trajectory_of_the_coupling_in_the_element(xi):
    """Test that the rebuilt exchange follows the previous element, which wrote it with the graph Laplacian."""
    system = NanotubeGapJunctionExchangeSystem(xi=xi).prepare().system_
    field = system.elements[0]
    graph = field.space.graph
    degrees = np.array([graph.degree[node] for node in graph.nodes], dtype=float)
    cells = _DegreeCellElement(
        'cells',
        x=field.x,
        alpha=field.alpha,
        beta=field.beta,
        g=field.g,
        xi=field.xi,
        degree=degrees,
        space=field.space,
    )
    previous = System(elements=[cells], interactions=[_PreviousExchangeInteraction('exchange', target='cells')])
    trajectory = system.prepare().simulate(dt=0.5, until=100)
    expected = previous.prepare().simulate(dt=0.5, until=100)
    assert np.abs(trajectory.get_values('cells.x') - expected.get_values('cells.x')).max() < 1e-10


def test_contour_uptake_interaction_mixes_logistic_map_and_next_cell():
    """Test that every cell mixes its logistic map with the power of the next cell, the last with the first."""
    x, r, c, p = np.array([0.2, 0.5, 0.7]), 3.0, np.array([0.1, 0.2, 0.3]), np.array([0.5, 1.0, 2.0])
    cells = ContourCellElement('cells', x=x, r=r, c=c, p=p, size=3)
    uptake = ContourUptakeInteraction('uptake', target='cells')
    system = System(elements=[cells], interactions=[uptake])
    found = system.prepare().simulate(until=1).get_values('cells.x')[-1]
    expected = (1 - c) * r * x * (1 - x) + c * np.roll(x, -1) ** p
    np.testing.assert_allclose(found, expected)
