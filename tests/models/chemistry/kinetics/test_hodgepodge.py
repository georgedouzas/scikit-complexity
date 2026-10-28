"""Test the `_hodgepodge` module of `skcomplex.models.chemistry.kinetics`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.chemistry.kinetics import hodgepodge


def test_hodgepodge_stays_in_range():
    """Test that the hodgepodge machine keeps every level between healthy and ill, with waves of infection."""
    levels = (
        Simulation(hodgepodge((50, 50)), dt=1, random_state=0)
        .check()
        .run(until=100)
        .trajectory_
        .values('cells.level')
    )
    assert levels.min() >= 0
    assert levels.max() <= 100
    assert np.ptp(levels[-1]) > 50


def test_hodgepodge_keeps_waves():
    """Test that the default hodgepodge machine keeps travelling waves, with more than a tenth of the cells ill."""
    levels = Simulation(hodgepodge(), dt=1, random_state=0).check().run(until=300).trajectory_.values('cells.level')
    assert (levels[-1] == 100).mean() > 0.1
