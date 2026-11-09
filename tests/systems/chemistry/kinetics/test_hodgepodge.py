"""Test the `_hodgepodge` module of `skcomplex.systems.chemistry.kinetics`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.chemistry.kinetics import HodgepodgeSystem


def test_hodgepodge_stays_in_range():
    """Test that the hodgepodge machine keeps every level between healthy and ill, with waves of infection."""
    levels = simulate(HodgepodgeSystem((50, 50)), until=100, random_state=0).get_values('cells.level')
    assert levels.min() >= 0
    assert levels.max() <= 100
    assert np.ptp(levels[-1]) > 50


def test_hodgepodge_keeps_waves():
    """Test that the default hodgepodge machine keeps travelling waves, with more than a tenth of the cells ill."""
    levels = simulate(HodgepodgeSystem(), until=300, random_state=0).get_values('cells.level')
    assert (levels[-1] == 100).mean() > 0.1


def test_hodgepodge_passes_its_parameters_to_its_parts():
    """Test that a model parameter changed with `set_params` changes the cells of the next `prepare`."""
    system = HodgepodgeSystem((10, 10)).set_params(g=20.0)
    assert system.get_params()['g'] == 20.0
    assert system.prepare().system_.elements[0].values['g'] == 20.0
