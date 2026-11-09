"""Test the `_exchange` module of `skcomplex.systems.biology.physiology`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import FieldElement, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import hill, laplacian
from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, StateSpec
from skcomplex.systems.biology.physiology import (
    ClosedContourCellExchangeSystem,
    ContourCellElement,
    ContourUptakeInteraction,
    ExchangingCellElement,
    NanotubeGapJunctionExchangeSystem,
)


class _PreviousCellElement(ExchangingCellElement):
    """Cell with the exchange in its own law through the graph Laplacian, as before the coupling interaction."""

    def __init__(
        self,
        label,
        visual=None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        alpha: Annotated[float, ParameterSpec()] = 1.0,
        beta: Annotated[float, ParameterSpec()] = 10.0,
        g: Annotated[float, ParameterSpec()] = 0.1,
        xi: Annotated[float, ParameterSpec()] = 0.0,
        degree: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label, visual=visual, x=x, alpha=alpha, beta=beta, g=g, xi=xi)
        self.degree = degree

    def _get_laws(self, s):
        """Return the metabolism of the substance and its exchange through gap junctions and nanotubes."""
        exchange = (s.g + s.xi) * laplacian(s.x) + s.xi * s.degree * s.x
        return [RateLaw(s.x, -s.alpha * hill(s.x, s.beta) + exchange)]


def test_closed_contour_cell_exchange_synchronizes_in_the_stability_domain():
    """Test that the contour cells synchronize at a stable equilibrium in its domain, and a chaotic one does not."""
    stable = simulate(ClosedContourCellExchangeSystem(r=2.5), dt=1, until=200).values('cells.x')
    chaotic = simulate(ClosedContourCellExchangeSystem(r=3.9), dt=1, until=200).values('cells.x')
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
        final[share] = simulate(system, dt=0.5, until=200).values('cells.x')[-1].mean()
    assert final[0.8] < 0.01
    assert final[1.2] > 10


@pytest.mark.parametrize('xi', [0.0, 0.01])
def test_nanotube_gap_junction_exchange_keeps_the_trajectory_of_the_coupling_in_the_element(xi):
    """Test that the rebuilt exchange follows the previous element, which wrote it with the graph Laplacian."""
    system = NanotubeGapJunctionExchangeSystem(xi=xi).prepare().system_
    field = system.elements[0]
    graph = field.space.graph
    degrees = np.array([graph.degree[node] for node in graph.nodes], dtype=float)
    cells = FieldElement(
        'cells',
        element=_PreviousCellElement,
        space=field.space,
        values={**field.values, 'degree': degrees},
    )
    previous = System(elements=[cells], time=ContinuousTime())
    trajectory = simulate(system, dt=0.5, until=100)
    expected = simulate(previous, dt=0.5, until=100)
    assert np.abs(trajectory.values('cells.x') - expected.values('cells.x')).max() < 1e-10


def test_contour_uptake_interaction_mixes_logistic_map_and_next_cell():
    """Test that every cell mixes its logistic map with the power of the next cell, the last with the first."""
    x, r, c, p = np.array([0.2, 0.5, 0.7]), 3.0, np.array([0.1, 0.2, 0.3]), np.array([0.5, 1.0, 2.0])
    cells = GroupElement('cells', element=ContourCellElement, size=3, values={'x': x, 'r': r, 'c': c, 'p': p})
    uptake = ContourUptakeInteraction('uptake', target='cells')
    system = System(elements=[cells], interactions=[uptake], time=DiscreteTime())
    found = simulate(system, until=1).values('cells.x')[-1]
    expected = (1 - c) * r * x * (1 - x) + c * np.roll(x, -1) ** p
    np.testing.assert_allclose(found, expected)
