"""Test the `_pairs` module of `skcomplex.core.components.laws.utils`."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import PartnerSpec, UpdateLaw
from skcomplex.core.components.laws.utils import pairs
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System


def _count(system):
    """Count every pair of buyers and sellers of the step on both members."""
    for buyer, seller in pairs(system.get_agents('buyers'), system.get_agents('sellers'), system.rng):
        buyer.paired += 1
        seller.paired += 1


def _count_partners(system):
    """Count every pair of traders of the step, chosen by a partner spec, on its second member."""
    spec = PartnerSpec('all', 'score', score=lambda a, b: b.paired)
    for _, second in pairs(system.get_agents('traders'), None, system.rng, spec):
        second.paired += 1


def _market(groups, rule):
    """Return the system of groups of traders and a market whose rule pairs them."""
    roles = {label: label for label in groups}
    market = Interaction('market', roles=roles, laws=lambda s: [UpdateLaw(s, rule)])
    elements = [Element(label, values={'paired': paired}, size=size) for label, (size, paired) in groups.items()]
    return System(elements=elements, interactions=[market])


def _members(size):
    """Return plain members with distinct ids."""
    return [SimpleNamespace(id=index) for index in range(size)]


@pytest.mark.parametrize('capacity', [0, 1.5, True])
def test_pairs_need_a_positive_whole_capacity(capacity):
    """Test that pairs reject a capacity that is not a positive integer."""
    with pytest.raises(ValidationError, match='positive whole number'):
        pairs(_members(2), None, np.random.default_rng(0), capacity=capacity)


def test_random_pairs_pair_every_member_at_most_once():
    """Test that 50 buyers and 30 sellers form 30 pairs every step, with no member in two pairs."""
    system = _market({'buyers': (50, 0.0), 'sellers': (30, 0.0)}, _count)
    trajectory = system.prepare().simulate(until=1, random_state=0)
    sellers = np.asarray(trajectory.get_agents('sellers', 1).values['paired'])
    buyers = np.asarray(trajectory.get_agents('buyers', 1).values['paired'])
    assert sellers.tolist() == [1.0] * 30
    assert buyers.sum() == 30
    assert buyers.max() == 1


def test_partner_spec_sends_every_member_to_the_highest_score():
    """Test that a score sends every member to the one with the most pairs."""
    system = _market({'traders': (3, [0.0, 5.0, 0.0])}, _count_partners)
    trajectory = system.prepare().simulate(until=1, random_state=0)
    assert trajectory.get_agents('traders', 1).values['paired'].sum() == pytest.approx(8.0)
    assert trajectory.get_agents('traders', 1).values['paired'][1] == pytest.approx(7.0)


def test_pairs_check_a_partner_spec():
    """Test that pairs check their partner spec."""
    with pytest.raises(ValidationError, match='Unknown choice'):
        pairs(_members(2), None, np.random.default_rng(0), PartnerSpec(choice='best'))


def test_random_pairs_within_one_group_avoid_self_pairs():
    """Test that random pairs within one group never pair a member with itself."""
    found = list(pairs(_members(9), None, np.random.default_rng(0), capacity=2))
    assert all(first is not second for first, second in found)
    assert len({frozenset((first.id, second.id)) for first, second in found}) == len(found)


def test_random_pairs_of_two_groups_give_every_member_its_capacity():
    """Test that random pairs of two groups pair every member of the smaller group at most once per capacity."""
    assert len(list(pairs(_members(5), _members(3), np.random.default_rng(0)))) == 3


def test_function_pairing_receives_both_groups_and_the_capacity():
    """Test that a pairing function receives both groups, the generator, and the capacity."""

    def pairing(left, right, rng, capacity):
        """Return the first member of the left with the second of the right, once per capacity."""
        return [(left[0], right[1])] * capacity

    found = pairs(_members(2), _members(2), np.random.default_rng(0), pairing, capacity=2)
    assert len(list(found)) == 2


def test_feasible_pairs_come_one_at_a_time():
    """Test that a feasibility reads what the pairs settled before it."""
    left, right = _members(4), [SimpleNamespace(id=10, stock=2)]
    settled = 0

    def feasible(agent, candidate):
        """Return whether the candidate still has stock."""
        return candidate.stock > 0

    for _, provider in pairs(left, right, np.random.default_rng(0), PartnerSpec(), feasible=feasible):
        provider.stock -= 1
        settled += 1
    assert settled == 2


def test_feasible_pairs_need_a_partner_spec():
    """Test that a feasibility without a partner spec raises an error."""
    with pytest.raises(ValidationError, match='need a `PartnerSpec`'):
        pairs(_members(2), None, np.random.default_rng(0), feasible=lambda a, b: True)
