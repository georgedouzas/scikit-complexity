"""Test the `_reducing` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import numpy as np
import pytest

from skcomplex.systems.mathematics.stochastic_processes import (
    CascadingSampleSpaceReducingSystem,
    FragmentationSystem,
    SampleSpaceReducingSystem,
)

from ..conftest import _run


def test_sample_space_reducing_visits_follow_zipf():
    """Test that a sample-space-reducing process visits state `i` with a frequency proportional to `1 / i`."""
    trajectory = _run(SampleSpaceReducingSystem(200, n_states=10), 200)
    visits = np.concatenate([trajectory.get_agents('balls', frame).values['level'] for frame in range(1, 201)])
    counts = np.bincount(visits.astype(int), minlength=11)
    assert counts[1] / counts[2] == pytest.approx(2, rel=0.1)
    assert counts[1] / counts[5] == pytest.approx(5, rel=0.15)


def test_fragmentation_conserves_the_total_size():
    """Test that fragments keep breaking while their total size stays that of the initial object."""
    trajectory = _run(FragmentationSystem(500), 20)
    lengths = trajectory.get_agents('fragments', 20).values['length']
    assert lengths.sum() == 500
    assert trajectory.get_values('fragments.count')[-1] > trajectory.get_values('fragments.count')[5] > 1


@pytest.mark.parametrize(('mu', 'n_cascades'), [(0.5, 3000), (1.0, 1000), (2.0, 50)])
def test_cascading_sample_space_reducing_visits_follow_the_multiplication_exponent(mu, n_cascades):
    """Test that the visits of cascades that multiply by `mu` fall off as `i^(-mu)`."""
    trajectory = _run(CascadingSampleSpaceReducingSystem(n_cascades, n_states=100, mu=mu), 101)
    visits = np.concatenate([trajectory.get_agents('balls', frame).values['level'] for frame in range(102)])
    counts = np.bincount(visits.astype(int), minlength=101)
    states = np.arange(2, 31)
    slope = np.polyfit(np.log(states), np.log(counts[states]), 1)[0]
    assert slope == pytest.approx(-mu, abs=0.15)
    assert trajectory.get_values('balls.count')[-1] == 0
