"""Test the `_pollution` module of `skcomplex.systems.physics.climate`."""

import numpy as np
import pytest

from skcomplex.systems import create_simulation

from ..conftest import _series


def test_population_pollution_follows_the_published_run():
    """Test the run of Fig.

    4.33 of Bala et al.: pollution near 9e7, population doubled, food near subsistence.
    """
    trajectory = create_simulation('population_pollution').prepare().run().trajectory_
    pollution, population, food = _series(trajectory, 'region.pollution', 'region.population', 'region.food_per_capita')
    assert np.all(np.diff(pollution) > 0)
    assert pollution[-1] == pytest.approx(9e7, rel=0.1)
    assert population[-1] == pytest.approx(2.1e8, rel=0.1)
    assert food[0] > 280
    assert food[-1] == pytest.approx(180.0, rel=0.05)


def test_population_pollution_absorbs_pollution_without_emissions():
    """Test that without emissions the pollution decays within its absorption time of under a year."""
    simulation = create_simulation('population_pollution', pollution_normal=0.0, pollution=5e7)
    trajectory = simulation.prepare().run(until=20.0).trajectory_
    pollution = _series(trajectory, 'region.pollution')[0]
    assert np.all(np.diff(pollution) < 0)
    assert pollution[-1] < 1e6
