"""Test the `_imitation` module of `skcomplex.core.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.interactions import ImitationInteraction
from skcomplex.core.spaces import LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec

SIZE = 50
RUNS = 200
FRACTION = 0.3


class PlayerElement(Element):
    """Player with a discrete strategy and a payoff."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        state: Annotated[object, StateSpec()] = 0.0,
        payoff: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.state = state
        self.payoff = payoff


def _final(space, values, rule, until, seed=0):
    """Return the strategies after imitation by a rule for some time."""
    system = System(
        elements=[LatticeElement('players', element=PlayerElement, space=space, values=values)],
        interactions=[ImitationInteraction('imitation', nodes='players', rule=rule)],
    )
    simulation = Simulation(system, dt=until, random_state=seed).check().prepare().run(until=until)
    return simulation.trajectory_.values('players.state')[-1]


def test_voters_on_a_ring_reach_consensus_with_the_initial_fraction_as_probability():
    """Test that voters on a ring of 50 always reach consensus, winning with the probability of their fraction."""
    initial = (np.arange(SIZE) < FRACTION * SIZE).astype(float)
    finals = [_final(LatticeSpace(SIZE), {'state': initial}, 'random', 5000.0, seed) for seed in range(RUNS)]
    assert all(np.ptp(final) == 0.0 for final in finals)
    wins = np.mean([final[0] for final in finals])
    assert wins == pytest.approx(FRACTION, abs=3 * np.sqrt(FRACTION * (1 - FRACTION) / RUNS))


def test_majority_flips_a_member_whose_neighbours_agree():
    """Test that a member whose two neighbours on a ring both hold the other option takes it, and blocks stay."""
    initial = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    final = _final(LatticeSpace(12), {'state': initial}, 'majority', 50.0)
    assert final.tolist() == [1.0] * 7 + [0.0] * 5


def test_best_copies_the_neighbour_of_largest_payoff_up_to_a_local_maximum():
    """Test that every member ends with the strategy of the local payoff maximum it climbs to through its neighbours."""
    graph = nx.gnp_random_graph(12, 0.3, seed=5)
    rng = np.random.default_rng(4)
    strategies, payoffs = rng.integers(0, 2, 12).astype(float), rng.normal(size=12)
    final = _final(NetworkSpace(graph), {'state': strategies, 'payoff': payoffs}, 'best', 200.0)

    def summit(node):
        """Return the node reached by moving to the neighbour of largest payoff while it pays more."""
        best = max(graph.neighbors(node), key=lambda other: payoffs[other], default=node)
        return summit(best) if payoffs[best] > payoffs[node] else node

    assert final.tolist() == [strategies[summit(node)] for node in graph]


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [({'rule': 'copy'}, 'rule of imitation'), ({'options': (1.0, 1.0)}, 'two distinct')],
)
def test_check_rejects_unknown_rules_and_equal_options(arguments, match):
    """Test that an unknown rule or options that are not distinct raise an error."""
    with pytest.raises(ValidationError, match=match):
        ImitationInteraction('imitation', nodes='players', **arguments).check()
