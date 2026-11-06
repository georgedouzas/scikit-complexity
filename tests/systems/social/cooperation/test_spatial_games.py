"""Test the `_spatial_games` module of `skcomplex.systems.social.cooperation`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.social.cooperation import (
    public_goods_game,
    spatial_prisoners_dilemma,
)

from ..conftest import _run


def _cooperation(trajectory):
    """Return the share of cooperators at every frame of a lattice game."""
    strategies = trajectory.values('players.strategy')
    return strategies.reshape(len(strategies), -1).mean(axis=1)


def test_spatial_prisoners_dilemma_regimes():
    """Test that cooperators and defectors coexist at temptation 1.85 and defectors win at temptation 3."""
    coexisting = _cooperation(_run(spatial_prisoners_dilemma(temptation=1.85), 50, dt=1))
    assert coexisting[-10:].min() > 0.1
    assert coexisting[-10:].max() < 0.6
    assert _cooperation(_run(spatial_prisoners_dilemma(temptation=3.0), 20, dt=1))[-1] == 0
    fermi = _run(spatial_prisoners_dilemma(temptation=1.05, imitation='fermi', share=0.5), 50, dt=1)
    assert _cooperation(fermi)[-1] > 0.9


def test_spatial_cooperation_survives_in_clusters():
    """Test that with imitate-the-best and temptation 1.6 cooperators survive in clusters for 200 steps."""
    trajectory = _run(spatial_prisoners_dilemma((40, 40), temptation=1.6), 200, dt=1)
    shares = _cooperation(trajectory)
    assert shares.min() > 0.2
    lattice = trajectory.values('players.strategy')[-1].reshape(40, 40)
    shifts = [np.roll(lattice, shift, axis) for shift in (-1, 1) for axis in (0, 1)]
    cooperating_neighbours = np.mean(shifts, axis=0)[lattice == 1]
    assert cooperating_neighbours.mean() > shares[-1]


def test_spatial_prisoners_dilemma_rejects_unknown_imitation():
    """Test that an unknown imitation rule raises a validation error when the system compiles."""
    with pytest.raises(ValidationError, match='Unknown imitation'):
        _run(spatial_prisoners_dilemma((10, 10), imitation='copy'), 1, dt=1)


def test_public_goods_game_depends_on_synergy():
    """Test that cooperation dies out at synergy 2 and takes over at synergy 6."""
    assert _cooperation(_run(public_goods_game(synergy=2.0), 50, dt=1))[-1] == 0
    assert _cooperation(_run(public_goods_game(synergy=6.0), 50, dt=1))[-1] > 0.9
