"""Test the `_matching` module of `skcomplex.systems.general.matching`, and its rule specs."""

import dataclasses

import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PartnerSpec, PhaseSpec, PhasesScheduleSpec
from skcomplex.systems.general.matching import (
    BatchMatchingSpec,
    ContinuousMatchingSpec,
    IterativeMatchingSpec,
    MatchingInteraction,
    MatchSpec,
    PartnerMatchingSpec,
    PartnerSwitchSpec,
    QueueSpec,
    RandomSearchSpec,
    StableMatchingSpec,
)

from .conftest import ParticipantElement


class RequesterElement(ParticipantElement):
    """Requester that quotes a random reserve below its value while it still requests a unit."""

    def _get_laws(self, s):
        """Return the quoting rule."""
        return [RuleLaw(self.quote, phase='quote')]

    def quote(self, agent, system):
        """Quote a random reserve below the value."""
        agent.reserve = system.rng.uniform(0, agent.value) if agent.quantity > 0 else 0.0


class ProviderElement(ParticipantElement):
    """Provider that quotes a random reserve above its value while it still provides a unit."""

    def _get_laws(self, s):
        """Return the quoting rule."""
        return [RuleLaw(self.quote, phase='quote')]

    def quote(self, agent, system):
        """Quote a random reserve above the value."""
        agent.reserve = system.rng.uniform(agent.value, 200) if agent.quantity > 0 else 200.0


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
        rule=BatchMatchingSpec(),
        cap='money',
        transfers=(('money', -1), ('goods', 1)),
        phase='clear',
    )
    system = CompositeSystem(
        elements=[requesters, providers],
        interactions=[matching],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('quote'), PhaseSpec('clear')])),
    )
    return values, costs, system


def test_match_settles_transfers_and_realises_the_surplus():
    """Test that random quotes within budgets realise most of the surplus and the transfers conserve the money."""
    values, costs, system = _auction(50, np.random.default_rng(1))
    trajectory = simulate(system, until=30, random_state=0)
    requesters = trajectory.get_agents('buyers', 30).values
    providers = trajectory.get_agents('sellers', 30).values
    bought, sold = requesters['goods'] > 0, providers['goods'] < 0
    realised = requesters['value'][bought].sum() - providers['value'][sold].sum()
    maximum = np.sum(np.maximum(np.sort(values)[::-1] - np.sort(costs), 0))
    assert realised > 0.9 * maximum
    assert requesters['money'].sum() + providers['money'].sum() == pytest.approx(values.sum())
    assert np.nansum(trajectory.get_values('auction.volume')) == bought.sum()
    assert np.all(requesters['quantity'][bought] == 0.0)


def test_match_reads_the_states_named_by_a_pair():
    """Test that a pair of names reads the requesters' reserve from its first state, the providers' from its second."""
    system = CompositeSystem(
        elements=[
            GroupElement('providers', element=ParticipantElement, size=1, values={'reserve': 5.0}),
            GroupElement('requesters', element=ParticipantElement, size=1, values={'reserve': 1.0, 'value': 9.0}),
        ],
        interactions=[
            MatchingInteraction(
                'matching',
                requesters='requesters',
                providers='providers',
                rule=BatchMatchingSpec(),
                reserve=('value', 'reserve'),
            ),
        ],
    )
    trajectory = simulate(system, until=1, random_state=0)
    assert trajectory.get_values('matching.value')[-1] == 7.0
    assert trajectory.get_values('matching.unmatched')[-1] == 0.0


def test_init_defaults_to_random_search():
    """Test that a matching without a rule matches by random search."""
    assert MatchingInteraction('matching', requesters='a', providers='b').rule == RandomSearchSpec()


def test_match_random_search_prices_at_the_midpoint_within_the_budget(clear):
    """Test that random search matches at the midpoint of the reserves, as much as the cap affords."""
    assert clear(RandomSearchSpec(), {'reserve': [10.0]}, {'reserve': [2.0]}) == [(10.0, 2.0, 0.0, 1.0, 6.0)]
    matches = clear(RandomSearchSpec(tries=2), {'reserve': [10.0], 'cap': 3.0}, {'reserve': [2.0]})
    assert matches[0][4] == 6.0
    assert matches[0][3] == pytest.approx(0.5)


def test_match_random_search_skips_limits_that_do_not_cross(clear):
    """Test that a requester whose reserve is below the provider's reserve does not match."""
    assert clear(RandomSearchSpec(tries=3), {'reserve': [1.0]}, {'reserve': [2.0]}) == []


def test_match_partner_switch_settles_on_the_lowest_partner(clear):
    """Test that a requester switches to the lower provider and then keeps it as its partner."""
    rule = PartnerSwitchSpec(sample=1, intensity=100.0)
    requesters = {'reserve': [10.0], 'quantity': 100.0, 'cap': 4.0}
    matches = clear(rule, requesters, {'reserve': [5.0, 4.0], 'quantity': 100.0}, steps=20)
    assert [provider for _, provider, _, _, _ in matches[-5:]] == [4.0] * 5


def test_match_partner_switch_prices_at_the_partner_limit(clear):
    """Test that a requester matches at the reserve of its partner."""
    matches = clear(PartnerSwitchSpec(sample=2), {'reserve': [10.0]}, {'reserve': [5.0, 4.0]})
    assert matches == [(10.0, 4.0, 0.0, 1.0, 4.0)]


def test_match_queue_serves_the_highest_limits_at_the_lowest_providers(clear):
    """Test that a queue by reserve serves the highest requesters at the reserves of the lowest providers."""
    matches = clear(QueueSpec(order='reserve'), {'reserve': [10.0, 9.0, 1.0]}, {'reserve': [4.0, 6.0]})
    assert [(requester, price) for requester, _, _, _, price in matches] == [(10.0, 4.0), (9.0, 6.0)]


def test_match_iterative_clearing_clears_at_a_price_between_the_limits(clear):
    """Test that the tâtonnement matches every crossing pair at one price between the marginal reserves."""
    matches = clear(IterativeMatchingSpec(), {'reserve': [10.0, 9.0, 1.0]}, {'reserve': [4.0, 6.0]})
    assert len(matches) == 2
    assert len({price for _, _, _, _, price in matches}) == 1
    assert all(6.0 <= price <= 9.0 for _, _, _, _, price in matches)


def test_match_iterative_clearing_matches_nothing_without_positive_limits(clear):
    """Test that offers without a positive reserve do not match."""
    assert clear(IterativeMatchingSpec(), {'reserve': [0.0]}, {'reserve': [0.0]}) == []


def test_match_continuous_clearing_gives_priority_to_the_earlier_offer(clear):
    """Test that of two providers at the same reserve, the earlier one matches first."""
    rule = ContinuousMatchingSpec(arrival='sequential')
    matches = clear(rule, {'reserve': [10.0]}, {'reserve': [10.0, 10.0], 'value': [0.0, 1.0]})
    assert matches == [(10.0, 10.0, 0.0, 1.0, 10.0)]


def test_match_continuous_clearing_trades_at_the_resting_limits(clear):
    """Test that an arriving request matches the lowest resting offers first, at their reserves."""
    rule = ContinuousMatchingSpec(arrival='sequential')
    matches = clear(rule, {'reserve': [11.5], 'quantity': 3.0}, {'reserve': [12.0, 9.0, 11.0]})
    assert [price for _, _, _, _, price in matches] == [9.0, 11.0]


def test_match_batch_clearing_pairs_the_crossing_offers_at_one_price(clear):
    """Test that the highest requesters match the lowest providers at the midpoint of the last crossing pair."""
    matches = clear(BatchMatchingSpec(), {'reserve': [10.0, 8.0, 3.0]}, {'reserve': [2.0, 5.0, 9.0]})
    assert len(matches) == 2
    assert {price for _, _, _, _, price in matches} == {6.5}


def test_match_batch_clearing_matches_nothing_without_crossing_limits(clear):
    """Test that reserves that never cross give no matches."""
    assert clear(BatchMatchingSpec(), {'reserve': [1.0]}, {'reserve': [2.0]}) == []


def _score(first, second):
    """Return the preference of a value for another, higher for a nearer and then a lower value."""
    return -abs(first - second) - 0.01 * second


def _closeness(first, second):
    """Return the preference of a participant for another by their values."""
    return _score(first.value, second.value)


def test_match_stable_gives_the_preferred_provider_to_its_preferred_requester(clear):
    """Test that the lowest provider goes to the highest requester, which it prefers by default."""
    matches = clear(StableMatchingSpec(), {'reserve': [10.0, 12.0]}, {'reserve': [5.0, 7.0]})
    assert {(requester, provider) for requester, provider, _, _, _ in matches} == {(12.0, 5.0), (10.0, 7.0)}


def test_match_stable_leaves_no_blocking_pair(clear):
    """Test that no unmatched requester and provider both prefer each other to their partners."""
    rng = np.random.default_rng(3)
    requesters, providers = rng.uniform(0, 10, 6), rng.uniform(0, 10, 6)
    rule = StableMatchingSpec(requester_preference=_closeness, provider_preference=_closeness)
    reserves = 1.0 + np.arange(6)
    matches = clear(rule, {'reserve': reserves, 'value': requesters}, {'reserve': np.zeros(6), 'value': providers})
    by_limit = dict(zip(reserves, requesters, strict=True))
    partners = {by_limit[requester]: value for requester, _, value, _, _ in matches}
    reverse = {value: requester for requester, value in partners.items()}
    assert len(partners) == 6

    for requester in requesters:
        for provider in providers:
            if partners[requester] == provider:
                continue
            requester_prefers = _score(requester, provider) > _score(requester, partners[requester])
            provider_prefers = _score(provider, requester) > _score(provider, reverse[provider])
            assert not (requester_prefers and provider_prefers)


def test_match_stable_keeps_the_pairs_of_its_reference(clear):
    """Test that the pairs and their order keep values from the previous library, without what cannot match.

    The provider of reserve 20 is above every requester, the provider of reserve 6 has nothing left, and the requester
    of reserve 3 is below every provider. The requester of reserve 8 loses out to the others.
    """
    matches = clear(
        StableMatchingSpec(),
        {'reserve': [10.0, 12.0, 8.0, 3.0, 9.0]},
        {'reserve': [5.0, 7.0, 20.0, 6.0], 'quantity': [2.0, 1.0, 1.0, 0.0]},
    )
    assert matches == [(12.0, 5.0, 0.0, 1.0, 5.0), (10.0, 5.0, 0.0, 1.0, 5.0), (9.0, 7.0, 0.0, 1.0, 7.0)]


def test_match_stable_keeps_the_pairs_of_its_reference_with_preferences(clear):
    """Test that preferences and larger capacities keep the pairs and their order from the previous library."""
    rule = StableMatchingSpec(requester_preference=_closeness, provider_preference=_closeness)
    matches = clear(
        rule,
        {'reserve': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0], 'value': [1.0, 4.0, 9.0, 2.0, 7.0, 5.0]},
        {'reserve': [0.0, 1.0, 2.0], 'value': [3.0, 6.0, 8.0], 'quantity': [1.0, 3.0, 1.0]},
    )
    assert matches == [
        (13.0, 0.0, 3.0, 1.0, 0.0),
        (15.0, 1.0, 6.0, 1.0, 1.0),
        (14.0, 1.0, 6.0, 1.0, 1.0),
        (11.0, 1.0, 6.0, 1.0, 1.0),
        (12.0, 2.0, 8.0, 1.0, 2.0),
    ]


def _lowest_limit(requester, provider):
    """Return the score of a provider, higher for a lower reserve."""
    del requester
    return -provider.reserve


def test_match_partners_takes_the_lowest_feasible_limit(clear):
    """Test that every requester takes the provider of the lowest reserve it accepts, at the provider's reserve."""
    matches = clear(
        PartnerMatchingSpec(choice='score', score=_lowest_limit, value='provider'),
        {'reserve': [10.0, 10.0]},
        {'reserve': [3.0, 5.0, 12.0]},
    )
    assert sorted((provider, value) for _, provider, _, _, value in matches) == [(3.0, 3.0), (5.0, 5.0)]


def test_match_partners_takes_the_provider_of_the_highest_score(clear):
    """Test that requesters take the provider of the highest score instead of the lowest reserve."""
    rule = PartnerMatchingSpec(choice='score', score=lambda requester, provider: provider.value, value='requester')
    matches = clear(rule, {'reserve': [10.0]}, {'reserve': [3.0, 5.0], 'value': [1.0, 2.0]})
    assert [(provider, value) for _, provider, _, _, value in matches] == [(5.0, 10.0)]


def test_match_partners_respects_the_capacity_of_providers(clear):
    """Test that a provider of capacity one takes one requester although it offers more."""
    rule = PartnerMatchingSpec(choice='score', score=_lowest_limit, capacity='value', value='provider')
    matches = clear(rule, {'reserve': [10.0, 10.0, 10.0]}, {'reserve': [1.0, 2.0], 'quantity': 5.0, 'value': 1.0})
    assert sorted(provider for _, provider, _, _, _ in matches) == [1.0, 2.0]


def test_match_partners_considers_local_cells_and_distances_only():
    """Test that requesters match only with providers in their cell or around it, or within a distance."""
    for candidates, expected in (('local', 1.0), (1.5, 1.0), ('all', 2.0)):
        requesters = GroupElement(
            'requesters',
            element=ParticipantElement,
            size=2,
            values={'reserve': 10.0},
            location=[(0, 0), (4, 4)],
        )
        providers = GroupElement(
            'providers',
            element=ParticipantElement,
            size=1,
            values={'reserve': 1.0, 'quantity': 2.0},
            location=[(1, 1)],
        )
        matching = MatchingInteraction(
            'matching',
            requesters='requesters',
            providers='providers',
            rule=PartnerSpec(candidates),
        )
        space = LatticeSpace((5, 5), metric='chebyshev', boundary='fixed')
        system = CompositeSystem(elements=[requesters, providers], interactions=[matching], space=space)
        trajectory = simulate(system, until=1, random_state=0)
        assert trajectory.get_values('matching.volume')[-1] == pytest.approx(expected)


def test_check_rejects_a_transfer_without_a_sign():
    """Test that a transfer whose sign is neither -1 nor 1 raises an error."""
    matching = MatchingInteraction('matching', requesters='a', providers='b', transfers=(('money', 2),))
    with pytest.raises(ValidationError, match='sign'):
        matching.prepare()


@pytest.mark.parametrize(
    ('rule', 'message'),
    [(MatchSpec('a', 'b', 1.0, 1.0), 'Unknown rule'), (QueueSpec(order='bid'), 'order')],
)
def test_check_rejects_a_rule_that_is_not_a_valid_matching_spec(rule, message):
    """Test that a rule that is not a matching spec, or an invalid one, raises an error."""
    matching = MatchingInteraction('matching', requesters='a', providers='b', rule=rule)
    with pytest.raises(ValidationError, match=message):
        matching.prepare()


def _reversed_pairs(requesters, providers, rng, capacity):
    """Return every requester paired with the provider at the mirrored position."""
    del rng, capacity
    return list(zip(requesters, reversed(providers), strict=True))


def test_match_custom_rule_settles_the_pairs_of_a_function_at_the_midpoint(clear):
    """Test that a pairing function as the rule matches the pairs it returns at the midpoint of their reserves."""
    matches = clear(_reversed_pairs, {'reserve': [10.0, 8.0]}, {'reserve': [2.0, 4.0]})
    assert sorted(matches) == [(8.0, 2.0, 0.0, 1.0, 5.0), (10.0, 4.0, 0.0, 1.0, 7.0)]


def test_check_rejects_an_unknown_rule_naming_the_specs_and_functions():
    """Test that a rule that is neither a matching spec nor a function raises an error naming both."""
    with pytest.raises(ValidationError, match=r'RandomSearchSpec.*StableMatchingSpec, a function'):
        MatchingInteraction('matching', requesters='a', providers='b', rule='greedy').prepare()


def test_match_rejects_a_misspelled_state_instead_of_defaulting_it():
    """Test that a cap state the requesters lack raises an error naming it instead of leaving them unbounded."""
    _, _, system = _auction(5, np.random.default_rng(1))
    system.interactions[0].set_params(cap='mony')
    with pytest.raises(ValidationError, match="no state 'mony'"):
        simulate(system, until=1, random_state=0)


def test_match_partner_spec_settles_the_pairs_at_the_midpoint(clear):
    """Test that a partner spec as the rule pairs every requester and matches at the midpoint of the reserves."""
    matches = clear(PartnerSpec('all', 'uniform'), {'reserve': [10.0]}, {'reserve': [2.0]})
    assert matches == [(10.0, 2.0, 0.0, 1.0, 6.0)]


def test_match_partners_rejects_a_provider_above_the_limit_of_the_requester(clear):
    """Test that a partner spec pairs no requester with a provider whose reserve it does not accept."""
    assert clear(PartnerSpec('all', 'uniform'), {'reserve': [1.0]}, {'reserve': [2.0]}) == []


def test_match_spec_is_a_frozen_spec():
    """Test that a match is a frozen spec of a requester, a provider, a quantity, and a price."""
    match = MatchSpec('requester', 'provider', 2.0, 5.0)
    assert (match.requester, match.provider, match.quantity * match.value) == ('requester', 'provider', 10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        match.value = 1.0


@pytest.mark.parametrize(
    ('spec', 'name', 'value'),
    [
        (RandomSearchSpec(tries=3), 'tries', 3),
        (PartnerSwitchSpec(sample=2), 'sample', 2),
        (QueueSpec(order='reserve'), 'order', 'reserve'),
        (IterativeMatchingSpec(iterations=10), 'iterations', 10),
        (ContinuousMatchingSpec(arrival='sequential'), 'arrival', 'sequential'),
        (StableMatchingSpec(value='requester'), 'value', 'requester'),
        (PartnerMatchingSpec(value='provider'), 'value', 'provider'),
    ],
)
def test_matching_rule_specs_hold_their_settings(spec, name, value):
    """Test that a matching rule is a frozen spec that keeps the setting it was given."""
    checked = spec
    assert getattr(checked, name) == value
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(checked, name, value)


def test_batch_matching_spec_has_no_settings():
    """Test that batch matching is a spec without settings."""
    assert dataclasses.fields(BatchMatchingSpec()) == ()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (RandomSearchSpec(value='bid'), 'midpoint'),
        (QueueSpec(order='bid'), 'order'),
        (ContinuousMatchingSpec(arrival='bid'), 'arrival'),
        (StableMatchingSpec(value='bid'), 'value'),
        (PartnerMatchingSpec(value='bid'), 'value'),
        (PartnerMatchingSpec(candidates=0.0), 'positive'),
    ],
)
def test_matching_rule_specs_reject_invalid_settings(spec, message):
    """Test that an unknown option, a distance that is not positive, and a bad score raise an error naming them."""
    with pytest.raises(ValidationError, match=message):
        spec._check()
