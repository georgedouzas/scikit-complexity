"""Test the `_adoption` module of `skcomplex.models.economics.innovation`."""

import numpy as np

from skcomplex.models import describe
from skcomplex.models.economics.innovation import (
    phone_adoption,
)

from ..conftest import _agents, _run


def test_phone_adoption_spreads_among_households_that_can_afford_it():
    """Test that adoption never falls, spreads, and reaches only households above the income threshold."""
    trajectory = _run(
        built := phone_adoption((20, 20)),
        50,
        observables=describe('phone_adoption').observables_of(built),
    )
    share = trajectory.values('share')
    assert np.all(np.diff(share) >= 0)
    assert share[-1] > 2 * share[0]
    households = _agents(trajectory, 'households', 50)
    assert np.all(households['income'][households['phone'] == 1] >= 1.2)
    stuck = _run(
        built := phone_adoption((20, 20), threshold=100.0),
        20,
        observables=describe('phone_adoption').observables_of(built),
    ).values('share')
    assert set(stuck.tolist()) == {0.0}
