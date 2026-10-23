"""Test the `_synapses` module of `skcomplex.models.biology.neuroscience`."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.models.biology.neuroscience import (
    adaptrode,
)


def test_adaptrode_keeps_long_term_memory():
    """Test that the Adaptrode weights order by time scale during input and that the slow weight outlasts it."""
    trajectory = Simulation(adaptrode(), dt=1).run(until=400).trajectory_
    w0, w1, w2 = (trajectory[f'synapse.w{index}'] for index in range(3))
    during = (trajectory.times > 100) & (trajectory.times < 150)
    assert np.all(w0[during] > w1[during])
    assert np.all(w1[during] > w2[during])
    after = trajectory.times == 200
    assert w0[after][0] - w1[after][0] < 0.05
    assert w2[-1] > 0.5 * w2[trajectory.times == 150][0]
