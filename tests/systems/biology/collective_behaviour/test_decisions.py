"""Test the `_decisions` module of `skcomplex.systems.biology.collective_behaviour`."""

import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.collective_behaviour import BinaryAntDecisionSystem, RatsSystem


def test_binary_ant_decision_breaks_symmetry():
    """Test that ants pick one branch in most stochastic runs, while the deterministic mode stays symmetric."""
    shares = []
    for seed in range(5):
        state = simulate(BinaryAntDecisionSystem().prepare().system_, until=1000, dt=10.0, random_state=seed).states[-1]
        shares.append(state.max() / state.sum())
    assert sum(share > 0.75 for share in shares) >= 4
    state = simulate(BinaryAntDecisionSystem(stochastic=False).prepare().system_, until=1000, dt=10.0).states[-1]
    assert state[0] == pytest.approx(state[1])


def test_rats_phase_transition():
    """Test that nervousness dies out below the transition and settles at `1 - calming / contagion` above it."""
    below = simulate(
        RatsSystem(spontaneous=0.0, contagion=0.5, stochastic=False).prepare().system_,
        until=50,
        dt=1.0,
    ).states[-1]
    above = simulate(
        RatsSystem(spontaneous=0.0, contagion=2.0, stochastic=False).prepare().system_,
        until=50,
        dt=1.0,
    ).states[-1]
    assert below[1] < 0.1
    assert above[1] / above.sum() == pytest.approx(0.5, rel=1e-3)
