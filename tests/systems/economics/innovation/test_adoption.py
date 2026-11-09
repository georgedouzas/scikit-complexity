"""Test the `_adoption` module of `skcomplex.systems.economics.innovation`."""

import numpy as np

from skcomplex.systems.economics.innovation import (
    PhoneAdoptionSystem,
)

from ..conftest import _agents, _run


def test_phone_adoption_spreads_among_households_that_can_afford_it():
    """Test that adoption never falls, spreads, and reaches only households above the income threshold."""
    trajectory = _run(
        PhoneAdoptionSystem((20, 20)).prepare().system_,
        50,
        observables=PhoneAdoptionSystem().get_run_settings()['observables'],
    )
    share = trajectory.get_values('share')
    assert np.all(np.diff(share) >= 0)
    assert share[-1] > 2 * share[0]
    households = _agents(trajectory, 'households', 50)
    assert np.all(households['income'][households['phone'] == 1] >= 1.2)
    stuck = _run(
        PhoneAdoptionSystem((20, 20), threshold=100.0).prepare().system_,
        20,
        observables=PhoneAdoptionSystem().get_run_settings()['observables'],
    ).get_values('share')
    assert set(stuck.tolist()) == {0.0}
