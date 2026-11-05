"""Test the `_hypergraph_games` module of `skcomplex.models.social.cooperation`."""

import numpy as np
import pytest

from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.models.social.cooperation import hypergraph_public_goods, simplicial_social_dilemmas

from ..conftest import _run

GAMES = {
    'sucker': -0.2,
    'temptation': 1.2,
    'sucker_allied': 0.5,
    'temptation_allied': 0.8,
    'sucker_opposed': -0.3,
    'temptation_opposed': 1.1,
}


def _cooperation(trajectory):
    """Return the share of cooperators at every frame of a game."""
    return trajectory.values('players.strategy').mean(axis=1)


def _first_payoffs(system, backend):
    """Return the payoffs of the play of the first step."""
    simulation = Simulation(system, random_state=0, dt=1, backend=backend).prepare().run(until=1)
    return simulation.trajectory_.values('players.payoff')[1]


def _game(own, other, sucker, temptation):
    """Return the payoff of a two-player game of reward 1 and punishment 0."""
    return own * other + own * (1 - other) * sucker + (1 - own) * other * temptation


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_hypergraph_public_goods_payoffs_sum_the_shares_of_every_hyperedge(backend):
    """Test that the payoffs of a step equal the shares of the pots of every hyperedge minus the contributions."""
    system = hypergraph_public_goods(60, synergy=3.5, seed=2)
    players = system.elements[0]
    strategies = players.values['strategy']
    expected = np.zeros(60)
    for hyperedge in players.space.prepare().hyperedges_:
        members = list(hyperedge)
        expected[members] += 3.5 * strategies[members].sum() / len(members) - strategies[members]
    np.testing.assert_allclose(_first_payoffs(system, backend), expected, atol=1e-12)


def test_hypergraph_public_goods_depends_on_synergy():
    """Test that cooperation dies out below the sizes of the hyperedges and takes over above them."""
    assert _cooperation(_run(hypergraph_public_goods(synergy=2.0), 100, dt=1))[-1] < 0.05
    assert _cooperation(_run(hypergraph_public_goods(synergy=6.0), 100, dt=1))[-1] > 0.95


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_simplicial_social_dilemmas_payoffs_follow_the_games_of_links_and_simplices(backend):
    """Test that the payoffs of a step equal a loop over the links of the triangles and the 2-simplices."""
    system = simplicial_social_dilemmas(40, seed=3, **GAMES)
    players = system.elements[0]
    strategies, triangles = players.values['strategy'], players.values['triangles']
    expected = np.zeros(40)
    for hyperedge in players.space.prepare().hyperedges_:
        for own in hyperedge:
            for other in hyperedge:
                if other == own:
                    continue
                games = (GAMES['sucker'], GAMES['temptation'])
                if len(hyperedge) == 3:
                    (third,) = set(hyperedge) - {own, other}
                    allied = strategies[third] == strategies[own]
                    games = (GAMES['sucker_allied'], GAMES['temptation_allied'])
                    if not allied:
                        games = (GAMES['sucker_opposed'], GAMES['temptation_opposed'])
                expected[own] += _game(strategies[own], strategies[other], *games)
    np.testing.assert_allclose(_first_payoffs(system, backend), expected / (2 * triangles), atol=1e-12)


def test_simplicial_social_dilemmas_simplices_sustain_cooperation():
    """Test that defectors win on plain triangles and an allied harmony game on 2-simplices saves cooperators."""
    plain = _cooperation(_run(simplicial_social_dilemmas(simplices=0.0), 100, dt=1))
    filled = _run(simplicial_social_dilemmas(simplices=1.0, temptation_allied=0.5, sucker_allied=0.5), 100, dt=1)
    assert plain[-1] == 0
    assert _cooperation(filled)[-10:].mean() > 0.3


def test_simplicial_social_dilemmas_counts_the_triangles_of_every_player():
    """Test that the network holds one triangle per new node besides the ten of the first clique."""
    players = simplicial_social_dilemmas(50, simplices=1.0).elements[0]
    assert players.values['triangles'].sum() == 3 * (10 + 45)
    assert len(players.space.prepare().hyperedges_) == 55
