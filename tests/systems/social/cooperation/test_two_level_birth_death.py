"""Test the `_two_level_birth_death` module of `skcomplex.systems.social.cooperation`."""

import numpy as np

from skcomplex.systems.social.cooperation import TwoLevelBirthDeathSystem

from ..conftest import _run, _values


def _punishers(trajectory, frame=-1):
    """Return the mean fraction of punishers of the groups at a frame."""
    return float(np.mean(_values(trajectory, 'groups', 'punishers', frame)) / 20)


def test_conflicts_between_groups_sustain_the_punishers():
    """Test that pairwise group conflict raises the punishers and their absence lets the defectors gain."""
    fighting = _run(TwoLevelBirthDeathSystem(), 100)
    assert _punishers(fighting) > 0.9
    peaceful = _run(TwoLevelBirthDeathSystem(conflict_rate=0.0), 100)
    assert _punishers(peaceful) < _punishers(peaceful, 0)


def test_groups_keep_their_size():
    """Test that the number of punishers of every group stays between 0 and the size of the group."""
    trajectory = _run(TwoLevelBirthDeathSystem(), 20)
    for frame in range(21):
        punishers = _values(trajectory, 'groups', 'punishers', frame)
        assert np.all((punishers >= 0) & (punishers <= 20))
