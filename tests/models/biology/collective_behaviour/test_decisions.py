"""Test the `_decisions` module of `skcomplex.models.biology.collective_behaviour`."""

import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.collective_behaviour import (
    binary_ant_decision,
    rats,
)


def test_binary_ant_decision_breaks_symmetry():
    """Test that ants pick one branch in most stochastic runs, while the deterministic mode stays symmetric."""
    shares = []
    for seed in range(5):
        state = Simulation(binary_ant_decision(), dt=10.0, random_state=seed).check().prepare().run(until=1000).state_
        shares.append(state.max() / state.sum())
    assert sum(share > 0.75 for share in shares) >= 4
    state = Simulation(binary_ant_decision(stochastic=False), dt=10.0).check().prepare().run(until=1000).state_
    assert state[0] == pytest.approx(state[1])


def test_rats_phase_transition():
    """Test that nervousness dies out below the transition and settles at `1 - calming / contagion` above it."""
    below = (
        Simulation(rats(spontaneous=0.0, contagion=0.5, stochastic=False), dt=1.0)
        .check()
        .prepare()
        .run(until=50)
        .state_
    )
    above = (
        Simulation(rats(spontaneous=0.0, contagion=2.0, stochastic=False), dt=1.0)
        .check()
        .prepare()
        .run(until=50)
        .state_
    )
    assert below[1] < 0.1
    assert above[1] / above.sum() == pytest.approx(0.5, rel=1e-3)
