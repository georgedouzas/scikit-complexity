"""Test the `_match` module of `skcomplex.core.components.laws`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import MatchLaw, RuleLaw
from skcomplex.core.components.laws.utils import matches
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PartnerSpec, StateSpec, VisualSpec


class TraderElement(Element):
    """Agent that counts the pairs it was in."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        paired: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.paired = paired


class ExchangeInteraction(Interaction):
    """Market that pairs buyers with sellers and counts the pairs."""

    def _get_laws(self, s):
        """Return the matching and the rule that reads it."""
        return [MatchLaw('buyers', 'sellers', capacity=1, label='trades'), RuleLaw(self.count)]

    def count(self, system):
        """Count every pair of the step on both members."""
        for buyer, seller in matches(system, 'trades'):
            buyer.paired += 1
            seller.paired += 1


@pytest.mark.parametrize('capacity', [0, 1.5, True])
def test_match_needs_a_positive_whole_capacity(capacity):
    """Test that a match rejects a capacity that is not a positive integer."""
    with pytest.raises(ValidationError, match='Give a positive integer'):
        MatchLaw('buyers', capacity=capacity).prepare()


def test_match_changes_no_state():
    """Test that a match targets no state, since it stores pairs for the rules of the step."""
    assert MatchLaw('buyers', 'sellers', label='trades').prepare().targets_ == ()


def test_random_matching_pairs_every_member_at_most_once():
    """Test that 50 buyers and 30 sellers form 30 pairs every step, with no member in two pairs."""
    system = System(
        elements=[
            GroupElement('buyers', element=TraderElement, size=50, variable=True),
            GroupElement('sellers', element=TraderElement, size=30, variable=True),
        ],
        interactions=[ExchangeInteraction('market')],
    )
    trajectory = Simulation(system, random_state=0).prepare().run(until=1).trajectory_
    sellers = np.asarray(trajectory.agents('sellers', 1).values['paired'])
    buyers = np.asarray(trajectory.agents('buyers', 1).values['paired'])
    assert sellers.tolist() == [1.0] * 30
    assert buyers.sum() == 30
    assert buyers.max() == 1


class PairedInteraction(Interaction):
    """Market that pairs the members of one group by a partner spec and counts the pairs."""

    def _get_laws(self, s):
        """Return the matching and the rule that reads it."""
        return [
            MatchLaw('traders', None, PartnerSpec('all', 'score', score=lambda a, b: b.paired), label='trades'),
            RuleLaw(self.count),
        ]

    def count(self, system):
        """Count every pair of the step on its second member."""
        for _, second in matches(system, 'trades'):
            second.paired += 1


def test_run_stores_the_pairs_of_a_partner_spec_for_the_rules_of_the_step():
    """Test that a match law lowers a partner spec, whose score sends every member to the one with the most pairs."""
    traders = GroupElement('traders', element=TraderElement, size=3, values={'paired': [0.0, 5.0, 0.0]})
    system = System(elements=[traders], interactions=[PairedInteraction('market')])
    trajectory = Simulation(system, random_state=0).prepare().run(until=1).trajectory_
    assert trajectory.agents('traders', 1).values['paired'].sum() == pytest.approx(8.0)
    assert trajectory.agents('traders', 1).values['paired'][1] == pytest.approx(7.0)


def test_check_validates_a_partner_spec():
    """Test that the check of a match law checks its partner spec."""
    with pytest.raises(ValidationError, match='Unknown choice'):
        MatchLaw('traders', pairing=PartnerSpec(choice='best')).prepare()
