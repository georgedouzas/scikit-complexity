"""Test transfers, invariants, matching, and adaptive parameters, the primitives that replace field vocabulary."""

import numpy as np
import pytest

from skcomplex.core.base import ConservationError, parameter, role, state
from skcomplex.core.components import Element, Group
from skcomplex.core.interactions import Interaction, Match, matches, random_pairing
from skcomplex.core.laws import Invariant, Rate, Rule, Transfer, Update, where
from skcomplex.core.runtime import Simulation
from skcomplex.core.space import DiscreteTime
from skcomplex.core.system import System


class Tanks(Element):
    """Two stocks with a transfer between them and the invariant of their total."""

    a: float = state(3.0)
    b: float = state(1.0)

    def laws(self, s):
        """Return the transfer and the invariant."""
        return [Transfer(s.a, s.b, s.a / 4, label='flow'), Invariant('total', s.a + s.b)]


@pytest.mark.parametrize('time', [None, DiscreteTime()])
def test_transfer_conserves_across_time_classes(time):
    """Test that a transfer keeps the total of two stocks in continuous and in discrete time."""
    trajectory = Simulation(System(components=[Tanks('t')], time=time), dt=1.0).run(until=5).trajectory_
    np.testing.assert_allclose(trajectory['t.a'] + trajectory['t.b'], 4.0, atol=1e-12)
    assert trajectory['t.a'][-1] < 3.0


def test_transfer_with_the_environment_takes_none():
    """Test that a transfer from `None` draws the quantity from the environment."""
    Filled = Element.define('Filled', states={'n': 0.0}, laws=lambda s: [Transfer(None, s.n, 1.0, label='inflow')])
    trajectory = Simulation(System(components=[Filled('f')]), dt=1.0).run(until=3).trajectory_
    assert trajectory['f.n'][-1] == pytest.approx(3.0)


def test_violated_invariant_names_itself_and_the_frame():
    """Test that a law creating a quantity from nothing breaks the declared invariant at the first frame."""
    Leaky = Element.define('Leaky', states={'a': 1.0}, laws=lambda s: [Rate(s.a, 1.0), Invariant('total', s.a)])
    with pytest.raises(ConservationError, match=r'invariant l\.total .* at frame 1'):
        Simulation(System(components=[Leaky('l')]), dt=0.5).run(until=2)


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


class Economy(Element):
    """Prices that grow every step, with an adaptive policy rate."""

    price: float = state(1.0)
    rate: float = parameter(0.02, adaptive=True)

    def laws(self, s):
        """Return the growth of prices."""
        return [Update(s.price, s.price * 1.03)]


class Policy(Interaction):
    """Rule that raises the policy rate whenever prices exceed a target."""

    target: str = role()

    def laws(self, s):
        """Return the update of the adaptive rate."""
        return [Update(s.target.rate, where(s.target.price > 1.05, s.target.rate + 0.01, s.target.rate))]


def test_adaptive_parameter_changes_only_when_the_policy_fires():
    """Test that the adaptive rate changes only at steps where prices exceed the target, and starts from its value."""
    system = System(components=[Economy('economy', rate=0.03)], interactions=[Policy('bank', target='economy')])
    trajectory = Simulation(system).run(until=4).trajectory_
    rates, prices = trajectory['economy.rate'], trajectory['economy.price']
    assert rates[0] == pytest.approx(0.03)
    np.testing.assert_array_equal(np.diff(rates) > 0, prices[:-1] > 1.05)
