"""Test the `_coevolution` module of `skcomplex.systems.biology.evolution`."""

import numpy as np
import pytest

from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.evolution import BakSneppenSystem

from ...conftest import EventsObserver


@pytest.mark.benchmark
def test_bak_sneppen_self_organises():
    """Test that Bak-Sneppen on a ring of 200 gathers its fitnesses above 0.75 with largest minima near 0.667."""
    observer = EventsObserver()
    system = BakSneppenSystem(200, updates=10_000).prepare().system_
    trajectory = simulate(system, until=200, dt=100, backend=NumbaBackend(), random_state=0, observers=[observer])
    minima = np.asarray([event.value for event in observer.events_ if event.time > 100])
    assert minima.size == 1_000_000
    assert np.median(trajectory.states[-1]) > 0.75
    assert 0.6 < minima.max() < 0.72
