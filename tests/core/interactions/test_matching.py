"""Test the `_matching` module of `skcomplex.core.interactions`."""

import numpy as np
import pytest

from skcomplex.core.elements import GroupElement
from skcomplex.core.interactions import MatchingInteraction, UniformPriceProtocol
from skcomplex.core.laws import RuleLaw
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import PhaseSpec, PhasesScheduleSpec

from .conftest import ParticipantElement


class RequesterElement(ParticipantElement):
    """Requester that quotes a random limit below its value while it still requests a unit."""

    def _get_laws(self, s):
        """Return the quoting rule."""
        return [RuleLaw(self.quote, phase='quote')]

    def quote(self, agent, model):
        """Quote a random limit below the value."""
        agent.limit = model.rng.uniform(0, agent.value) if agent.quantity > 0 else 0.0


class ProviderElement(ParticipantElement):
    """Provider that quotes a random limit above its value while it still provides a unit."""

    def _get_laws(self, s):
        """Return the quoting rule."""
        return [RuleLaw(self.quote, phase='quote')]

    def quote(self, agent, model):
        """Quote a random limit above the value."""
        agent.limit = model.rng.uniform(agent.value, 200) if agent.quantity > 0 else 200.0


def _auction(size, rng):
    """Return the values, the costs, and a uniform price matching of requesters with budgets and providers."""
    values = rng.uniform(50, 200, size)
    costs = rng.uniform(0, 150, size)
    requesters = GroupElement('buyers', element=RequesterElement, size=size, values={'value': values, 'money': values})
    providers = GroupElement('sellers', element=ProviderElement, size=size, values={'value': costs})
    matching = MatchingInteraction(
        'auction',
        requesters='buyers',
        providers='sellers',
        protocol=UniformPriceProtocol(),
        budget='money',
        transfers=(('money', -1), ('goods', 1)),
        phase='clear',
    )
    system = System(
        elements=[requesters, providers],
        interactions=[matching],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('quote'), PhaseSpec('clear')])),
    )
    return values, costs, system


def test_match_settles_transfers_and_realises_the_surplus():
    """Test that random quotes within budgets realise most of the surplus and the transfers conserve the money."""
    values, costs, system = _auction(50, np.random.default_rng(1))
    simulation = Simulation(system, random_state=0).check().prepare().run(until=30)
    requesters = simulation.trajectory_.agents('buyers', 30).values
    providers = simulation.trajectory_.agents('sellers', 30).values
    bought, sold = requesters['goods'] > 0, providers['goods'] < 0
    realised = requesters['value'][bought].sum() - providers['value'][sold].sum()
    maximum = np.sum(np.maximum(np.sort(values)[::-1] - np.sort(costs), 0))
    assert realised > 0.9 * maximum
    assert requesters['money'].sum() + providers['money'].sum() == pytest.approx(values.sum())
    assert np.nansum(simulation.trajectory_.values('auction.volume')) == bought.sum()
    assert np.all(requesters['quantity'][bought] == 0.0)


def test_match_reads_the_states_named_by_a_pair():
    """Test that a pair of names reads the requesters' limit from its first state and the providers' from its second."""
    system = System(
        elements=[
            GroupElement('providers', element=ParticipantElement, size=1, values={'limit': 5.0}),
            GroupElement('requesters', element=ParticipantElement, size=1, values={'limit': 1.0, 'value': 9.0}),
        ],
        interactions=[
            MatchingInteraction(
                'matching',
                requesters='requesters',
                providers='providers',
                protocol=UniformPriceProtocol(),
                limit=('value', 'limit'),
            ),
        ],
    )
    trajectory = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_
    assert trajectory.values('matching.price')[-1] == 7.0
    assert trajectory.values('matching.unmatched')[-1] == 0.0


def test_match_defaults_to_random_search(clear):
    """Test that a matching without a protocol matches by random search at the midpoint."""
    assert clear(None, {'limit': [10.0]}, {'limit': [2.0]}) == [(10.0, 2.0, 0.0, 1.0, 6.0)]


def test_check_rejects_a_transfer_without_a_sign():
    """Test that a transfer whose sign is neither -1 nor 1 raises an error."""
    matching = MatchingInteraction('matching', requesters='a', providers='b', transfers=(('money', 2),))
    with pytest.raises(ValidationError, match='sign'):
        matching.check()
