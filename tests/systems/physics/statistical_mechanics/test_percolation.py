"""Test the `_percolation` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np

from skcomplex.core.analysis import bind_analysis, clusters
from skcomplex.core.components.system import System
from skcomplex.core.experiments import sweep
from skcomplex.systems.physics.statistical_mechanics import ContinuumPercolationSystem

FILLINGS = [0.8, 1.0, 1.25, 1.5]
SEEDS = list(range(40))


def _disks(**parameters: object) -> System:
    """Return the disks of a continuum percolation system for its model parameters, as an experiment runs them."""
    return ContinuumPercolationSystem(**parameters).prepare().system_


def _spanning_shares(side: float) -> np.ndarray:
    """Return the share of seeds with a spanning cluster at every filling factor, for a side of the square."""
    bounds = {'side': [side], 'filling': FILLINGS, 'random_state': SEEDS}
    result = sweep(
        _disks,
        bounds,
        until=1.0,
        analyses=[bind_analysis(clusters, kind='percolation', radius=1.0)],
        outputs=[],
    )
    spans = result.column('clusters.spanning').astype(float)
    return spans.reshape(len(FILLINGS), len(SEEDS)).mean(axis=1)


def test_continuum_percolation_threshold() -> None:
    """Test that the spanning probabilities of two sizes cross between filling factors 1.0 and 1.25."""
    small, large = _spanning_shares(20.0), _spanning_shares(60.0)
    assert large[1] < small[1]
    assert large[2] > small[2]


def test_continuum_percolation_number_of_disks() -> None:
    """Test that the number of disks gives the filling factor."""
    disks = ContinuumPercolationSystem(1.2, side=60.0, radius=1.0, random_state=0).prepare().system_.elements[0]
    assert abs(disks.size * np.pi / 60.0**2 - 1.2) < 1e-3
