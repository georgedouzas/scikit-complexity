"""Test the `_spatial_games` module of `skcomplex.systems.social.cooperation`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.social.cooperation import (
    IteratedDilemmaInteraction,
    IteratedLatticeDilemmaSystem,
    PublicGoodsGameSystem,
    SpatialPrisonersDilemmaSystem,
)

from ..conftest import _run


def _cooperation(trajectory):
    """Return the share of cooperators at every frame of a lattice game."""
    strategies = trajectory.values('players.strategy')
    return strategies.reshape(len(strategies), -1).mean(axis=1)


def test_spatial_prisoners_dilemma_regimes():
    """Test that cooperators and defectors coexist at temptation 1.85 and defectors win at temptation 3."""
    coexisting = _cooperation(_run(SpatialPrisonersDilemmaSystem(temptation=1.85), 50, dt=1))
    assert coexisting[-10:].min() > 0.1
    assert coexisting[-10:].max() < 0.6
    assert _cooperation(_run(SpatialPrisonersDilemmaSystem(temptation=3.0), 20, dt=1))[-1] == 0
    fermi = _run(SpatialPrisonersDilemmaSystem(temptation=1.05, imitation='fermi', share=0.5), 50, dt=1)
    assert _cooperation(fermi)[-1] > 0.9


def test_spatial_cooperation_survives_in_clusters():
    """Test that with imitate-the-best and temptation 1.6 cooperators survive in clusters for 200 steps."""
    trajectory = _run(SpatialPrisonersDilemmaSystem((40, 40), temptation=1.6), 200, dt=1)
    shares = _cooperation(trajectory)
    assert shares.min() > 0.2
    lattice = trajectory.values('players.strategy')[-1].reshape(40, 40)
    shifts = [np.roll(lattice, shift, axis) for shift in (-1, 1) for axis in (0, 1)]
    cooperating_neighbours = np.mean(shifts, axis=0)[lattice == 1]
    assert cooperating_neighbours.mean() > shares[-1]


def test_spatial_prisoners_dilemma_rejects_unknown_imitation():
    """Test that an unknown imitation rule raises a validation error when the system compiles."""
    with pytest.raises(ValidationError, match='Unknown imitation'):
        _run(SpatialPrisonersDilemmaSystem((10, 10), imitation='copy'), 1, dt=1)


def test_public_goods_game_depends_on_synergy():
    """Test that cooperation dies out at synergy 2 and takes over at synergy 6."""
    assert _cooperation(_run(PublicGoodsGameSystem(synergy=2.0), 50, dt=1))[-1] == 0
    assert _cooperation(_run(PublicGoodsGameSystem(synergy=6.0), 50, dt=1))[-1] > 0.9


def test_best_reply_to_a_strategy_is_to_defect_one_round_earlier():
    """Test that against a strategy `m` the fewest years in prison come from the strategy `m - 1`."""
    for opponent in range(1, 11):
        years = IteratedDilemmaInteraction.years(
            np.arange(11.0),
            np.full(11, float(opponent)),
            rounds=10,
            reward=0.5,
            sucker=1.5,
            temptation=0.0,
            punishment=1.0,
        )
        assert int(np.argmin(years)) == opponent - 1


def test_iterated_dilemma_with_two_strategies_has_three_regimes():
    """Test that cooperation dominates at `R = 0.82`, coexists at `R = 0.84`, and vanishes at `R = 0.86`."""
    shares = {}
    for reward in (0.82, 0.84, 0.86):
        system = IteratedLatticeDilemmaSystem(reward=reward, extremes=True).prepare().system_
        strategies = simulate(system, until=100, random_state=0).values('players.strategy')
        shares[reward] = np.mean(strategies[-20:] == 7)
        assert set(np.unique(strategies)) <= {0.0, 7.0}
    assert shares[0.82] > 0.8
    assert 0.3 < shares[0.84] < 0.8
    assert shares[0.86] < 0.2


def test_iterated_dilemma_with_every_strategy_cycles():
    """Test that in the coexistence regime cooperative and defecting strategies alternate in dominance."""
    system = IteratedLatticeDilemmaSystem().prepare().system_
    strategies = simulate(system, until=200, random_state=0).values('players.strategy')
    means = strategies.reshape(len(strategies), -1).mean(axis=1)
    assert means[50:].max() - means[50:].min() > 2.0
    assert len(np.unique(strategies[-1])) > 3
