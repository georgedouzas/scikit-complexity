"""Test the laws of agents: matchings and their random pairing."""

import numpy as np

from skcomplex.core.base import BaseInteraction, state
from skcomplex.core.components import Element, GroupElement
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.agents import MatchLaw, RuleLaw, matches, random_pairing


class TraderElement(Element):
    """Agent that counts the pairs it was in."""

    paired: float = state(0.0)


class ExchangeInteraction(BaseInteraction):
    """Market that pairs buyers with sellers and counts the pairs."""

    def laws(self, s):
        """Return the matching and the rule that reads it."""
        return [MatchLaw('buyers', 'sellers', capacity=1, label='trades'), RuleLaw(self.count)]

    def count(self, model):
        """Count every pair of the step on both members."""
        for buyer, seller in matches(model, 'trades'):
            buyer.paired += 1
            seller.paired += 1


def test_random_matching_pairs_every_member_at_most_once():
    """Test that 50 buyers and 30 sellers form 30 pairs every step, with no member in two pairs."""
    system = System(
        components=[
            GroupElement('buyers', element=TraderElement, size=50, variable=True),
            GroupElement('sellers', element=TraderElement, size=30, variable=True),
        ],
        interactions=[ExchangeInteraction('market')],
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
