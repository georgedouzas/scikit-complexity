"""Test the `_imitation` module of `skcomplex.core.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from scipy.special import expit

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.interactions import ImitationInteraction
from skcomplex.core.spaces import ContinuousTime, DiscreteTime, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec

SIZE = 50
RUNS = 200
FRACTION = 0.3
DISCRETE_SIZE = 20
DISCRETE_RUNS = 100


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


def _final(space, values, rule, until, seed=0, **arguments):
    """Return the strategies after imitation by a rule for some time, in continuous or in discrete time."""
    discrete = arguments.get('form') == 'probability'
    system = System(
        elements=[LatticeElement('players', element=PlayerElement, space=space, values=values)],
        interactions=[ImitationInteraction('imitation', target='players', rule=rule, **arguments)],
        time=DiscreteTime() if discrete else ContinuousTime(),
    )
    simulation = Simulation(system, dt=1.0 if discrete else until, random_state=seed)
    return simulation.check().prepare().run(until=until).trajectory_.values('players.state')[-1]


def _alternating(size):
    """Return strategies and payoffs alternating between option 0 of payoff 0 and option 1 of payoff 1."""
    values = (np.arange(size) % 2).astype(float)
    return {'state': values, 'payoff': values}


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


def test_voters_in_discrete_time_reach_consensus_with_the_initial_fraction_as_probability():
    """Test that voters on a ring of 20 imitating with probability one half per step reach consensus."""
    initial = (np.arange(DISCRETE_SIZE) < FRACTION * DISCRETE_SIZE).astype(float)
    finals = [
        _final(
            LatticeSpace(DISCRETE_SIZE),
            {'state': initial},
            'random',
            2000.0,
            seed,
            form='probability',
            probability=0.5,
        )
        for seed in range(DISCRETE_RUNS)
    ]
    assert all(np.ptp(final) == 0.0 for final in finals)
    wins = np.mean([final[0] for final in finals])
    assert wins == pytest.approx(FRACTION, abs=3 * np.sqrt(FRACTION * (1 - FRACTION) / DISCRETE_RUNS))


def test_fermi_copies_with_the_probability_of_the_payoff_difference():
    """Test that in one step a member copies a neighbour with probability `p / (1 + exp(-beta (P_j - P_i)))`."""
    values = _alternating(20000)
    final = _final(LatticeSpace(20000), values, 'fermi', 1.0, form='probability', probability=0.8, intensity=1.0)
    switched = final != values['state']
    assert switched[::2].mean() == pytest.approx(0.8 * expit(1.0), abs=0.015)
    assert switched[1::2].mean() == pytest.approx(0.8 * expit(-1.0), abs=0.015)


def test_fermi_copies_at_the_rate_times_the_fermi_probability():
    """Test that in a short time a member copies a neighbour at the rate times its Fermi probability."""
    values = _alternating(20000)
    final = _final(LatticeSpace(20000), values, 'fermi', 0.1, intensity=1.0)
    switched = final != values['state']
    assert switched[::2].mean() == pytest.approx(1 - np.exp(-0.1 * expit(1.0)), abs=0.01)
    assert switched[1::2].mean() == pytest.approx(1 - np.exp(-0.1 * expit(-1.0)), abs=0.01)


def test_majority_in_discrete_time_flips_a_member_whose_neighbours_agree():
    """Test that in discrete time a member whose two neighbours on a ring hold the other option takes it."""
    initial = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    final = _final(LatticeSpace(12), {'state': initial}, 'majority', 3.0, form='probability')
    assert final.tolist() == [1.0] * 7 + [0.0] * 5


def test_best_in_discrete_time_climbs_to_the_local_payoff_maximum():
    """Test that in discrete time every member ends with the strategy of the payoff maximum it climbs to."""
    payoffs = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 5.0])
    strategies = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    final = _final(LatticeSpace(8), {'state': strategies, 'payoff': payoffs}, 'best', 10.0, form='probability')
    assert final.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'rule': 'copy'}, 'rule of imitation'),
        ({'form': 'events'}, 'form of imitation'),
        ({'options': (1.0, 1.0)}, 'two distinct'),
    ],
)
def test_check_rejects_unknown_rules_and_equal_options(arguments, match):
    """Test that an unknown rule or options that are not distinct raise an error."""
    with pytest.raises(ValidationError, match=match):
        ImitationInteraction('imitation', target='players', **arguments).check()
