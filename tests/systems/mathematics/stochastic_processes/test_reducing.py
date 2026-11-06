"""Test the `_reducing` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import numpy as np
import pytest

from skcomplex.systems.mathematics.stochastic_processes import (
    fragmentation,
    sample_space_reducing,
)

from ..conftest import _run


def test_sample_space_reducing_visits_follow_zipf():
    """Test that a sample-space-reducing process visits state `i` with a frequency proportional to `1 / i`."""
    trajectory = _run(sample_space_reducing(200, n_states=10), 200)
    visits = np.concatenate([trajectory.agents('balls', frame).values['level'] for frame in range(1, 201)])
    counts = np.bincount(visits.astype(int), minlength=11)
    assert counts[1] / counts[2] == pytest.approx(2, rel=0.1)
    assert counts[1] / counts[5] == pytest.approx(5, rel=0.15)


def test_fragmentation_conserves_the_total_size():
    """Test that fragments keep breaking while their total size stays that of the initial object."""
    trajectory = _run(fragmentation(500), 20)
    lengths = trajectory.agents('fragments', 20).values['length']
    assert lengths.sum() == 500
    assert trajectory.values('fragments.count')[-1] > trajectory.values('fragments.count')[5] > 1
