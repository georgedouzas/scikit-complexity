"""Test the `_trafficking` module of `skcomplex.models.social.organizations`."""

import numpy as np

from skcomplex.models.social.organizations import (
    trafficking,
)

from ..conftest import _run

TRAFFICKING_TRIPS = '011101220001123101500402242222122031341221'


TRAFFICKING_BILLS = (
    '15.39067 17.901823 30.5131 8.432013 37.175629 29.030561 7.355628 18.482139 6.048363 12.712 7.70364 '
    '32.673267 25.123793 27.780338 16.653185 17.55624 7.035796 33.565717 6.491482 5.844644 39.924158 '
    '44.366192 26.418483 20.797019 4.643588 40.198269 7.879553 17.431179 36.342778 22.506267 25.992822 '
    '29.253357 26.328384 19.891423 4.752967 18.14027 4.919435 42.940525 40.552128 43.123515 70.027132 '
    '86.314599 78.100177 '
)


def test_trafficking_binds_both_groups_and_matches_recorded_run():
    """Test that the daily interaction reads both groups through roles and the grid size from the model."""
    system = trafficking(random_state=0)
    day = system.interactions[0]
    assert (day.women, day.premises) == ('women', 'premises')
    assert not {'width', 'height'} & set(day.get_params())
    trajectory = _run(system, 90)
    assert ''.join(str(int(trips)) for trips in trajectory.agents('women', 90).values['trips']) == TRAFFICKING_TRIPS
    bills = np.asarray(TRAFFICKING_BILLS.split(), dtype=float)
    np.testing.assert_allclose(trajectory.agents('premises', 90).values['last_bills'], bills, atol=1e-6)


def test_trafficking_victims_stay_in_brothels():
    """Test that victims are always in brothels and that brothels run higher bills than houses."""
    trajectory = _run(trafficking(random_state=0), 180)
    premises, women = trajectory.agents('premises', 180), trajectory.agents('women', 180)
    brothels = premises.values['brothel'] > 0
    cells = {tuple(location) for location in premises.locations[brothels].tolist()}
    victims = women.locations[women.values['victim'] > 0].tolist()
    assert all(tuple(location) in cells for location in victims)
    assert premises.values['last_bills'][brothels].mean() > premises.values['last_bills'][~brothels].mean()
