"""Test the laws of agents: matchings and their random pairing."""

import numpy as np

from skcomplex.core.base import state
from skcomplex.core.components import Element, Group
from skcomplex.core.interactions import Interaction
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.agents import Match, Rule, matches, random_pairing


class Trader(Element):
    """Agent that counts the pairs it was in."""

    paired: float = state(0.0)


class Exchange(Interaction):
    """Market that pairs buyers with sellers and counts the pairs."""

    def laws(self, s):
        """Return the matching and the rule that reads it."""
        return [Match('buyers', 'sellers', capacity=1, label='trades'), Rule(self.count)]

    def count(self, model):
        """Count every pair of the step on both members."""
        for buyer, seller in matches(model, 'trades'):
            buyer.paired += 1
            seller.paired += 1


def test_random_matching_pairs_every_member_at_most_once():
    """Test that 50 buyers and 30 sellers form 30 pairs every step, with no member in two pairs."""
    system = System(
        components=[
            Group('buyers', element=Trader, size=50, variable=True),
            Group('sellers', element=Trader, size=30, variable=True),
        ],
        interactions=[Exchange('market')],
    )
    trajectory = Simulation(system, random_state=0).run(until=1).trajectory_
    sellers = np.asarray(trajectory.agents('sellers', 1).values['paired'])
    buyers = np.asarray(trajectory.agents('buyers', 1).values['paired'])
    assert sellers.tolist() == [1.0] * 30
    assert buyers.sum() == 30
    assert buyers.max() == 1


def test_random_pairing_within_one_group_avoids_self_pairs():
    """Test that random pairing within one group never pairs a member with itself."""
    members = list(range(9))
    pairs = random_pairing(members, members, np.random.default_rng(0), 2)
    assert all(first != second for first, second in pairs)
    assert len({frozenset(pair) for pair in pairs}) == len(pairs)
