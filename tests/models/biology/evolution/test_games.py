"""Test the `_games` module of `skcomplex.models.biology.evolution`."""

import numpy as np
import pytest

from skcomplex.models.biology.evolution import (
    replicator_dynamics,
    rock_paper_scissors,
)

from ..conftest import _run_until
from .conftest import PRECISE


def test_hawks_and_doves_settle_at_share_v_over_c():
    """Test that hawks settle at the share `V / C` when fights cost more than the resource is worth."""
    trajectory = _run_until(replicator_dynamics(V=2.0, C=3.0), 50, dt=1.0, solver=PRECISE)
    assert trajectory['population.x'][-1][0] == pytest.approx(2 / 3, abs=1e-4)
    assert trajectory['population.x'][-1].sum() == pytest.approx(1)


def test_replicator_rock_paper_scissors_conserves_product():
    """Test that rock, paper, and scissors, three members of a group, keep the simplex and the product of the shares."""
    game = ((0, -1, 1), (1, 0, -1), (-1, 1, 0))
    system = replicator_dynamics([0.5, 0.3, 0.2], payoff=game)
    assert system.elements[0].size == 3
    trajectory = _run_until(system, 100, dt=0.5, solver=PRECISE)
    shares = trajectory['population.x']
    assert np.abs(shares.sum(axis=1) - 1).max() < 1e-10
    assert np.abs(np.prod(shares, axis=1) - 0.03).max() < 1e-6


def test_rock_paper_scissors_lattice_keeps_three_strategies():
    """Test that cyclic invasion on a lattice keeps the three strategies at about a third each."""
    trajectory = _run_until(rock_paper_scissors((40, 40)), 30, dt=1, random_state=0)
    final = trajectory['lattice.strategy'][-1]
    shares = np.bincount(final.astype(int), minlength=3) / final.size
    assert shares.min() > 0.1
    assert not np.array_equal(final, trajectory['lattice.strategy'][0])
