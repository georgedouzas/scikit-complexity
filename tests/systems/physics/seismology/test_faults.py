"""Test the `_faults` module of `skcomplex.systems.physics.seismology`."""

import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.physics.seismology import EarthquakesSystem

from ...conftest import EventsObserver


def test_earthquakes_have_broad_sizes():
    """Test that the earthquake system records earthquakes of sizes spanning two decades."""
    observer = EventsObserver()
    (
        Simulation(
            EarthquakesSystem((30, 30)).prepare().system_,
            dt=100,
            record=False,
            random_state=0,
            observers=[observer],
        )
        .prepare()
        .run(until=5000)
    )
    sizes = np.asarray([event.size for event in observer.events_])
    assert sizes.min() == 1
    assert sizes.max() > 100
