"""Test the `_faults` module of `skcomplex.models.earth.seismology`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.earth.seismology import earthquakes

from ...conftest import Events


def test_earthquakes_have_broad_sizes():
    """Test that the earthquake model records earthquakes of sizes spanning two decades."""
    observer = Events()
    Simulation(earthquakes((30, 30)), dt=100, record=False, random_state=0, observers=[observer]).run(until=5000)
    sizes = np.asarray([event.size for event in observer.events])
    assert sizes.min() == 1
    assert sizes.max() > 100
