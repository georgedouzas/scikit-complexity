"""Test the `_coevolution` module of `skcomplex.models.biology.evolution`."""

import numpy as np
import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.models.biology.evolution import bak_sneppen

from ...conftest import Events


@pytest.mark.benchmark
def test_bak_sneppen_self_organises():
    """Test that Bak-Sneppen on a ring of 200 gathers its fitnesses above 0.75 with largest minima near 0.667."""
    observer = Events()
    system = bak_sneppen(200, updates=10_000)
    simulation = Simulation(system, dt=100, record=False, backend='numba', random_state=0, observers=[observer])
    simulation.run(until=200)
    minima = np.asarray([event.value for event in observer.events if event.time > 100])
    assert minima.size == 1_000_000
    assert np.median(simulation.state_) > 0.75
    assert 0.6 < minima.max() < 0.72
