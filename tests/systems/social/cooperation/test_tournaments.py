"""Test the `_tournaments` module of `skcomplex.systems.social.cooperation`."""

import numpy as np

from skcomplex.systems.social.cooperation import (
    PrisonersDilemmaTournamentSystem,
    play,
)

from ..conftest import _run, _values


def test_iterated_games_and_a_cooperative_population():
    """Test the payoffs of iterated games and that a population of cooperators keeps fitness 3 without mutation."""
    tit_for_tat = [1, 1, 0, 1, 0, 1, 0]
    assert play(tit_for_tat, [0] * 7) == (5.0, 10.0)
    assert play([1] * 7, [1] * 7) == (18.0, 18.0)
    system = PrisonersDilemmaTournamentSystem(20, mutation=0.0, random_state=0).prepare().system_
    system.elements[0].values['genome'] = np.ones((20, 7))
    assert set(_values(_run(system, 5), 'agents', 'fitness').tolist()) == {3.0}
