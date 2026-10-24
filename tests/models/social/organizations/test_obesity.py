"""Test the `_obesity` module of `skcomplex.models.social.organizations`."""

from skcomplex.models.social.organizations import (
    basal_metabolic_rate,
    obesity,
)

from ..conftest import _run, _values


def test_obesity_weight_follows_meals():
    """Test that people stay within the grid and that heavy eaters near every restaurant gain weight."""
    trajectory = _run(obesity(30, n_places=10, size=50.0, reach=100.0, random_state=0), 28)
    locations = trajectory.agents('people', 28).locations
    assert locations.min() >= 0 and locations.max() <= 50  # noqa: PT018
    habits = _values(trajectory, 'people', 'habit')
    gain = _values(trajectory, 'people', 'weight') - _values(trajectory, 'people', 'weight', 0)
    assert gain[habits == 2].mean() > gain[habits == 0].mean()
    assert basal_metabolic_rate(70.0, 175.0, 30.0, 0.0) < basal_metabolic_rate(70.0, 175.0, 30.0, 1.0)


def test_obesity_gains_follow_habits_whatever_the_access():
    """Test that people with no restaurant within reach still eat, so mean gains rise from low to high habits."""
    trajectory = _run(obesity(random_state=0), 70)
    habits = _values(trajectory, 'people', 'habit')
    meals = _values(trajectory, 'restaurants', 'sold').sum()
    assert meals == 3 * 70 * len(habits)
    gain = _values(trajectory, 'people', 'weight') - _values(trajectory, 'people', 'weight', 0)
    means = [gain[habits == habit].mean() for habit in range(3)]
    assert means[0] < means[1] < means[2]
