"""Test the `_exchange` module of `skcomplex.models.biology.physiology`."""

import networkx as nx
import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.physiology import (
    closed_contour_cell_exchange,
    nanotube_gap_junction_exchange,
)


def test_closed_contour_cell_exchange_synchronizes_in_the_stability_domain():
    """Test that the contour cells synchronize at a stable equilibrium in its domain, and a chaotic one does not."""
    stable = Simulation(closed_contour_cell_exchange(r=2.5), dt=1).check().run(200).trajectory_.values('cells.x')
    chaotic = Simulation(closed_contour_cell_exchange(r=3.9), dt=1).check().run(200).trajectory_.values('cells.x')
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
        system = nanotube_gap_junction_exchange(graph, xi=share * critical, x=0.1)
        final[share] = Simulation(system, dt=0.5).check().run(200).trajectory_.values('cells.x')[-1].mean()
    assert final[0.8] < 0.01
    assert final[1.2] > 10
