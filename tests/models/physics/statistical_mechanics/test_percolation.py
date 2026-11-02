"""Test the `_percolation` module of `skcomplex.models.physics.statistical_mechanics`."""

import numpy as np

from skcomplex.analysis import ClusterAnalysis
from skcomplex.experiments import Experiment, GridDesign
from skcomplex.models.physics.statistical_mechanics import continuum_percolation

FILLINGS = [0.8, 1.0, 1.25, 1.5]
SEEDS = list(range(40))


def _spanning_shares(side: float) -> np.ndarray:
    """Return the share of seeds with a spanning cluster at every filling factor, for a side of the square."""
    bounds = {'side': [side], 'filling': FILLINGS, 'random_state': SEEDS}
    experiment = Experiment(
        continuum_percolation,
        GridDesign(),
        bounds,
        until=1.0,
        analyses=[ClusterAnalysis('percolation', radius=1.0)],
        outputs=[],
    )
    spans = experiment.check().prepare().run().result_.column('clusteranalysis.spanning').astype(float)
    return spans.reshape(len(FILLINGS), len(SEEDS)).mean(axis=1)


def test_continuum_percolation_threshold() -> None:
    """Test that the spanning probabilities of two sizes cross between filling factors 1.0 and 1.25."""
    small, large = _spanning_shares(20.0), _spanning_shares(60.0)
    assert large[1] < small[1]
    assert large[2] > small[2]


def test_continuum_percolation_number_of_disks() -> None:
    """Test that the number of disks gives the filling factor."""
    disks = continuum_percolation(1.2, side=60.0, radius=1.0, random_state=0).elements[0]
    assert abs(disks.size * np.pi / 60.0**2 - 1.2) < 1e-3
