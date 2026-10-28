"""Test the `_minority` module of `skcomplex.models.economics.econophysics`."""

from skcomplex.engine.runtime import Simulation
from skcomplex.models.economics.econophysics import (
    minority_game,
)


def _volatility(memory, n_agents=51, **options):
    """Return `sigma^2 / N` of the attendance of a minority game after a transient."""
    system = minority_game(n_agents, memory=memory, random_state=0, **options)
    attendance = Simulation(system, random_state=0).check().run(until=400).trajectory_.values('game.attendance')
    return attendance[100:].var() / n_agents


def test_minority_game_volatility_minimum():
    """The volatility exceeds random play for small `alpha` and falls well below it near `alpha_c ~ 0.34`."""
    assert _volatility(1) > 1
    assert _volatility(4) < 0.6
    assert _volatility(4, evolutionary=True, period=50) < 1
