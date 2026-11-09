"""Test the `_trafficking` module of `skcomplex.systems.social.organizations`."""

import numpy as np

from skcomplex.systems.social.organizations import (
    TraffickingSystem,
)

from ..conftest import _run

TRAFFICKING_TRIPS = '210002022212131123123131102321211124232231'


TRAFFICKING_BILLS = (
    '15.39067 17.901823 30.5131 8.432013 37.175629 29.030561 7.355628 18.482139 6.048363 12.712 7.820885 '
    '32.673267 25.123793 21.972537 16.653185 17.55624 7.035796 33.565717 7.003061 5.844644 39.924158 44.366192 '
    '26.418483 20.797019 4.643588 40.438545 7.879553 17.431179 36.342778 22.506267 21.764179 29.253357 26.328384 '
    '19.891423 4.752967 18.14027 4.919435 42.940525 40.552128 43.123515 82.435316 88.269587 54.87078 '
)


def test_trafficking_binds_both_groups_and_matches_recorded_run():
    """Test that the daily interaction reads both groups through roles and the grid size from the system."""
    system = TraffickingSystem(random_state=0).prepare().system_
    day = system.interactions[0]
    assert (day.women, day.premises) == ('women', 'premises')
    assert not {'width', 'height'} & set(day.get_params())
    trajectory = _run(system, 90)
    assert ''.join(str(int(trips)) for trips in trajectory.get_agents('women', 90).values['trips']) == TRAFFICKING_TRIPS
    bills = np.asarray(TRAFFICKING_BILLS.split(), dtype=float)
    np.testing.assert_allclose(trajectory.get_agents('premises', 90).values['last_bills'], bills, atol=1e-6)


def test_trafficking_victims_stay_in_brothels():
    """Test that victims are always in brothels and that brothels run higher bills than houses."""
    trajectory = _run(TraffickingSystem(random_state=0), 180)
    premises, women = trajectory.get_agents('premises', 180), trajectory.get_agents('women', 180)
    brothels = premises.values['brothel'] > 0
    cells = {tuple(location) for location in premises.locations[brothels].tolist()}
    victims = women.locations[women.values['victim'] > 0].tolist()
    assert all(tuple(location) in cells for location in victims)
    assert premises.values['last_bills'][brothels].mean() > premises.values['last_bills'][~brothels].mean()
