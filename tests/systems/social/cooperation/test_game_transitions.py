"""Test the `_game_transitions` module of `skcomplex.systems.social.cooperation`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.social.cooperation import GameTransitionsSystem

from ..conftest import _run, _values


def _cooperation(trajectory):
    """Return the final share of cooperators."""
    return float(np.mean(_values(trajectory, 'players', 'strategy')))


def test_transitions_favour_cooperation_under_pairwise_comparison():
    """Test that cooperation wins with `b1 - b2 > 2 c` and transitions, and loses with game 1 alone."""
    settings = {'update': 'pairwise', 'selection': 1.0, 'first_benefit': 6.0, 'second_benefit': 2.0}
    assert _cooperation(_run(GameTransitionsSystem(**settings), 50)) == 1.0
    assert _cooperation(_run(GameTransitionsSystem(**settings, transitions=False), 50)) == 0.0


def test_mutual_cooperation_keeps_the_valuable_game():
    """Test that the share of edges of game 1 is reported and is one once everyone cooperates."""
    trajectory = _run(
        GameTransitionsSystem(update='pairwise', selection=1.0, first_benefit=6.0, second_benefit=2.0),
        50,
    )
    assert trajectory.get_values('evolution.game_one')[-1] == 1.0
    assert trajectory.get_values('evolution.game_one')[1] < 1.0


def test_game_transitions_rejects_an_unknown_update():
    """Test that the update must be death-birth or pairwise comparison."""
    with pytest.raises(ValidationError, match='Unknown update'):
        _run(GameTransitionsSystem(update='birth-death'), 1)
