"""Define the interaction that matches requesters with providers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import heapq
import itertools
import math
from collections.abc import Callable, MutableMapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np

from ....core.components.base import BaseComponent
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, MatchLaw, RuleLaw
from ....core.components.laws.utils import transfer
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol, Pairing, SystemViewProtocol
from ....core.enums import ElementKind
from ....core.errors import ValidationError
from ....core.specs import PartnerSpec, RoleSpec
from ._batch_matching_spec import BatchMatchingSpec
from ._continuous_matching_spec import ContinuousMatchingSpec
from ._iterative_matching_spec import IterativeMatchingSpec
from ._match_spec import MatchSpec
from ._partner_matching_spec import PartnerMatchingSpec
from ._partner_switch_spec import PartnerSwitchSpec
from ._queue_spec import QueueSpec
from ._random_search_spec import RandomSearchSpec
from ._stable_matching_spec import StableMatchingSpec
from .types import MatchingRule

SIGNS = (-1, 1)
DEFAULT_STATES = ('reserve', 'quantity', 'cap')
RULES = (
    RandomSearchSpec,
    PartnerSpec,
    PartnerSwitchSpec,
    QueueSpec,
    IterativeMatchingSpec,
    ContinuousMatchingSpec,
    BatchMatchingSpec,
    StableMatchingSpec,
)


class MatchingInteraction[SystemT: SystemViewProtocol](Interaction):
    """An interaction that matches requesters with providers and settles every match.

    Requesters state in their states the highest value they accept, `reserve`, the quantity they want, `quantity`, and
    the most value they can give, `cap`. Providers state the lowest value they accept and the quantity they offer. A
    name given as a pair names the state of the requesters first and of the providers second. The `rule` is a spec of
    the algorithm that pairs the offers, such as `RandomSearchSpec`, `QueueSpec`, `BatchMatchingSpec`, or
    `StableMatchingSpec`.

    Every match lowers the quantities of both sides and moves every state of `transfers`: a negative sign moves `value *
    quantity` from the requester to the provider, and a positive sign moves `quantity` from the provider to the
    requester, so the totals stay the same. `on_match(match, system)` adds any other effect. The matching runs once per
    round of its phase and reports `<label>.value`, the clearing value weighted by quantity, `<label>.volume`, and
    `<label>.unmatched`.

    Unlike `TransferInteraction`, which moves amounts between given pairs, it first forms the pairs by a rule, through
    the `MatchLaw` of the core, and then transfers. Choose it when who trades with whom and at what value are part of
    the model.

    It appears in many fields. Buyers and sellers trade goods, workers and firms agree on wages, borrowers and banks
    agree on loans, and patients and hospitals are matched.

    Args:
        label:
            The label of the matching.

        requesters:
            The path of the group of requesters.

        providers:
            The path of the group of providers.

        rule:
            The spec of the algorithm that matches the members, such as `ContinuousMatchingSpec()`, or a function
            `rule(requesters, providers, rng, capacity)` of the agents with a positive quantity that returns their
            pairs, which match at the midpoint of both reserves. A `PartnerSpec` picks among the providers that still
            offer a quantity at a reserve the requester accepts, in random order of the requesters, and settles at
            the midpoint, or at the value of a `PartnerMatchingSpec`.

        reserve:
            The state with the reserve of both sides, or a pair of the requesters' and the providers' states. An agent
            without the default state `'reserve'` has no reserve as a requester and the reserve 0 as a provider.

        quantity:
            The state with the quantity of both sides, or a pair of the requesters' and the providers' states. An
            agent without the default state `'quantity'` offers the quantity 1.

        cap:
            The state of a requester with the most value it can give. A requester without the default state
            `'cap'` has no reserve on what it gives.

        transfers:
            The states every match moves, each with the sign `-1` for a move from the requester to the provider or
            `1` for a move from the provider to the requester.

        phase:
            The phase the matching runs in, or `None` for the default phase.

        on_match:
            A function of every match and the system view, run after the transfers, or `None`.

    Raises:
        ValidationError:
            A transfer has a sign other than `-1` or `1`, or the rule is neither a matching spec nor a function, or
            it is invalid, raised by `prepare`, or an agent lacks a state other than a default one that names its
            reserve, quantity, or cap, raised by `run`.

    Examples:
        >>> from skcomplex.systems.general.matching import MatchingInteraction
        >>> from skcomplex.systems.general.matching import ContinuousMatchingSpec
        >>> matching = MatchingInteraction(
        ...     'exchange',
        ...     requesters='traders',
        ...     providers='traders',
        ...     rule=ContinuousMatchingSpec(),
        ...     reserve=('bid', 'ask'),
        ... )
        >>> matching.reports
        ('value', 'volume', 'unmatched')
    """

    reports: ClassVar[tuple[str, ...]] = ('value', 'volume', 'unmatched')

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (MatchLaw, RuleLaw)

    def _get_draws(self) -> bool:
        """Return whether the rule draws.

        Every rule draws but a queue served in a fixed order.
        """
        return super()._get_draws() or not (isinstance(self.rule, QueueSpec) and self.rule.order != 'random')

    def __init__(
        self,
        label: str,
        *,
        requesters: Annotated[str, RoleSpec(elements=(ElementKind.GROUP, ElementKind.VARIABLE_GROUP))],
        providers: Annotated[str, RoleSpec(elements=(ElementKind.GROUP, ElementKind.VARIABLE_GROUP))],
        rule: MatchingRule | Pairing = RandomSearchSpec(),
        reserve: str | tuple[str, str] = 'reserve',
        quantity: str | tuple[str, str] = 'quantity',
        cap: str = 'cap',
        transfers: Sequence[tuple[str, int]] = (),
        phase: str | None = None,
        on_match: Callable[[MatchSpec, SystemT], None] | None = None,
    ) -> None:
        super().__init__(label)
        self.requesters = requesters
        self.providers = providers
        self.rule = rule
        self.reserve = reserve
        self.quantity = quantity
        self.cap = cap
        self.transfers = transfers
        self.phase = phase
        self.on_match = on_match

    @staticmethod
    def _get_sides(name: str | tuple[str, str]) -> tuple[str, str]:
        """Return the state of the requesters and of the providers of a name.

        The name is given once or as a pair.
        """
        return (name, name) if isinstance(name, str) else name

    def _read(self, agent: AgentViewProtocol, name: str, default: float) -> float:
        """Return a state of an agent.

        It returns a default when the agent lacks a state of a default name.
        """
        if hasattr(agent, name):
            return float(getattr(agent, name))
        if name in DEFAULT_STATES:
            return default
        error_msg = (
            f'The agents of {agent.group!r} have no state {name!r}, which the matching {self.label!r} reads. Declare '
            f'the state on their element, or name one of their states.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _get_affordable(offer: SimpleNamespace, value: float) -> float:
        """Return the quantity a participant can still match at a value.

        The value is within its cap.
        """
        if value <= 0:
            return float(offer.quantity)
        return max(0.0, min(offer.quantity, offer.cap / value))

    @staticmethod
    def _record_match(
        trades: list[MatchSpec],
        requester: SimpleNamespace,
        provider: SimpleNamespace,
        value: float,
    ) -> None:
        """Record a match of the largest quantity both sides can match at a value.

        It also lowers both offers.
        """
        quantity = min(MatchingInteraction._get_affordable(requester, value), provider.quantity)
        if quantity <= 0:
            return
        requester.quantity -= quantity
        requester.cap -= quantity * value
        provider.quantity -= quantity
        trades.append(MatchSpec(requester.agent, provider.agent, quantity, value))

    @staticmethod
    def _get_value(requester: SimpleNamespace, provider: SimpleNamespace, clearing: str) -> float:
        """Return the value of a match under a clearing rule."""
        if clearing == 'requester':
            return requester.reserve
        if clearing == 'provider':
            return provider.reserve
        return (requester.reserve + provider.reserve) / 2

    @staticmethod
    def _get_order(offers: list[SimpleNamespace], order: str, rng: np.random.Generator) -> list[SimpleNamespace]:
        """Return offers in random order, by decreasing reserve, or in order of creation."""
        if order == 'reserve':
            return sorted(offers, key=lambda item: (-item.reserve, item.agent.id))
        if order == 'sequential':
            return sorted(offers, key=lambda item: item.agent.id)
        return [offers[index] for index in rng.permutation(len(offers))]

    def _get_offers(self, system: SystemT) -> tuple[list[SimpleNamespace], list[SimpleNamespace]]:
        """Return the offers of the requesters and providers with a positive quantity.

        Matches lower the quantity.
        """
        reserves, quantities = self._get_sides(self.reserve), self._get_sides(self.quantity)
        requesters = [
            SimpleNamespace(
                agent=agent,
                reserve=self._read(agent, reserves[0], math.inf),
                quantity=self._read(agent, quantities[0], 1.0),
                cap=self._read(agent, self.cap, math.inf),
            )
            for agent in system.get_agents(self._get_role('requesters'))
        ]
        providers = [
            SimpleNamespace(
                agent=agent,
                reserve=self._read(agent, reserves[1], 0.0),
                quantity=self._read(agent, quantities[1], 1.0),
                cap=math.inf,
            )
            for agent in system.get_agents(self._get_role('providers'))
        ]
        return [offer for offer in requesters if offer.quantity > 0], [
            offer for offer in providers if offer.quantity > 0
        ]

    def _match_random_search(
        self,
        rule: RandomSearchSpec,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of requesters in random order that meet random providers.

        They match on crossing reserves.
        """
        trades: list[MatchSpec] = []
        for index in rng.permutation(len(requesters)):
            requester = requesters[index]
            for _ in range(rule.tries):
                available = [provider for provider in providers if provider.quantity > 0]
                if requester.quantity <= 0 or not available:
                    break
                provider = available[int(rng.integers(len(available)))]
                if provider.reserve <= requester.reserve:
                    self._record_match(trades, requester, provider, self._get_value(requester, provider, rule.value))
        return trades

    @staticmethod
    def _choose_partner(
        rule: PartnerSwitchSpec,
        best: SimpleNamespace,
        current: SimpleNamespace | None,
        rng: np.random.Generator,
    ) -> SimpleNamespace:
        """Return the provider a requester matches with.

        It is its current partner or the best of its sample.
        """
        if current is None or current.quantity <= 0:
            return best
        if best.reserve >= current.reserve:
            return current
        if best.reserve <= 0:
            return best
        switching = 1 - math.exp(rule.intensity * (best.reserve - current.reserve) / best.reserve)
        return best if rng.random() < switching else current

    def _match_partner_switch(
        self,
        rule: PartnerSwitchSpec,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of every requester with its kept or new partner.

        The matches are at the partner's reserve.
        """
        trades: list[MatchSpec] = []
        by_id = {provider.agent.id: provider for provider in providers}
        for index in rng.permutation(len(requesters)):
            requester = requesters[index]
            available = [provider for provider in providers if provider.quantity > 0]
            if not available:
                break
            picks = rng.choice(len(available), size=min(rule.sample, len(available)), replace=False)
            best = min((available[int(pick)] for pick in picks), key=lambda provider: provider.reserve)
            current = by_id.get(int(getattr(requester.agent, rule.partner, -1)))
            chosen = self._choose_partner(rule, best, current, rng)
            if hasattr(requester.agent, rule.partner):
                setattr(requester.agent, rule.partner, chosen.agent.id)
            if chosen.reserve <= requester.reserve:
                self._record_match(trades, requester, chosen, chosen.reserve)
        return trades

    def _match_queue(
        self,
        rule: QueueSpec,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of the queue of requesters served by the lowest providers.

        The providers serve at their reserves.
        """
        queue = self._get_order(requesters, rule.order, rng)
        stock = sorted(providers, key=lambda provider: provider.reserve)
        trades: list[MatchSpec] = []
        for requester in queue:
            for provider in stock:
                if requester.quantity <= 0 or provider.reserve > requester.reserve:
                    break
                if provider.quantity > 0:
                    self._record_match(trades, requester, provider, provider.reserve)
        return trades

    @staticmethod
    def _adjust(
        rule: IterativeMatchingSpec,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        start: float,
    ) -> float:
        """Return the value the tâtonnement reaches from a start."""
        value = start
        for _ in range(rule.iterations):
            requested = sum(
                MatchingInteraction._get_affordable(requester, value)
                for requester in requesters
                if requester.reserve >= value
            )
            provided = sum(provider.quantity for provider in providers if provider.reserve <= value)
            total = requested + provided
            if not total or abs(requested - provided) / total < rule.tolerance:
                break
            value *= 1 + rule.adjustment * (requested - provided) / total
        return value

    def _match_at(
        self,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        value: float,
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches at one value between the willing requesters and providers.

        They match in random order.
        """
        trades: list[MatchSpec] = []
        willing_requesters = self._get_order([item for item in requesters if item.reserve >= value], 'random', rng)
        willing_providers = self._get_order([item for item in providers if item.reserve <= value], 'random', rng)
        pairs = iter(willing_providers)
        provider = next(pairs, None)
        for requester in willing_requesters:
            while provider is not None and MatchingInteraction._get_affordable(requester, value) > 0:
                self._record_match(trades, requester, provider, value)
                if provider.quantity <= 0:
                    provider = next(pairs, None)
        return trades

    def _match_iterative(
        self,
        rule: IterativeMatchingSpec,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches at the value of the tâtonnement.

        The tâtonnement starts from the value of the last clearing.
        """
        reserves = [offer.reserve for offer in requesters + providers if offer.reserve > 0]
        if not reserves:
            return []
        last = storage.get('value')
        start = last if isinstance(last, float) and math.isfinite(last) and last > 0 else float(np.median(reserves))
        value = self._adjust(rule, requesters, providers, start)
        storage['value'] = value
        return self._match_at(requesters, providers, value, rng)

    @staticmethod
    def _arrange(
        rule: ContinuousMatchingSpec,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        rng: np.random.Generator,
    ) -> list[tuple[SimpleNamespace, bool]]:
        """Return the offers in their order of arrival, each with whether it is a request."""
        offers = [(offer, True) for offer in requesters] + [(offer, False) for offer in providers]
        if rule.arrival == 'sequential':
            return sorted(offers, key=lambda item: item[0].agent.id)
        return [offers[index] for index in rng.permutation(len(offers))]

    def _match_continuous(
        self,
        rule: ContinuousMatchingSpec,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of every arriving offer with the best resting offers.

        The resting offers are those of the other side.
        """
        requests: list[tuple[float, int, SimpleNamespace]] = []
        provisions: list[tuple[float, int, SimpleNamespace]] = []
        trades: list[MatchSpec] = []
        for time, (offer, is_request) in enumerate(self._arrange(rule, requesters, providers, rng)):
            if is_request:
                while (
                    provisions
                    and provisions[0][0] <= offer.reserve
                    and MatchingInteraction._get_affordable(offer, provisions[0][0]) > 0
                ):
                    resting = provisions[0][2]
                    self._record_match(trades, offer, resting, resting.reserve)
                    if resting.quantity <= 0:
                        heapq.heappop(provisions)
                if MatchingInteraction._get_affordable(offer, offer.reserve) > 0:
                    heapq.heappush(requests, (-offer.reserve, time, offer))
                continue
            while requests and -requests[0][0] >= offer.reserve and offer.quantity > 0:
                resting = requests[0][2]
                self._record_match(trades, resting, offer, resting.reserve)
                if MatchingInteraction._get_affordable(resting, resting.reserve) <= 0:
                    heapq.heappop(requests)
            if offer.quantity > 0:
                heapq.heappush(provisions, (offer.reserve, time, offer))
        return trades

    @staticmethod
    def _get_requested(offer: SimpleNamespace) -> float:
        """Return the quantity a requester can match at its own reserve."""
        return (
            offer.quantity
            if not math.isfinite(offer.cap)
            else MatchingInteraction._get_affordable(offer, offer.reserve)
        )

    def _get_pair(
        self,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
    ) -> tuple[list[tuple[SimpleNamespace, SimpleNamespace]], float]:
        """Return the pairs of crossing offers in order of their reserves.

        It also returns the value of the last pair.
        """
        ranked_requesters = sorted(requesters, key=lambda offer: (-offer.reserve, offer.agent.id))
        ranked_providers = sorted(providers, key=lambda offer: (offer.reserve, offer.agent.id))
        left = {id(offer): self._get_requested(offer) for offer in ranked_requesters}
        offered = {id(offer): offer.quantity for offer in ranked_providers}
        pairs: list[tuple[SimpleNamespace, SimpleNamespace]] = []
        marginal = math.nan
        requester_iter, provider_iter = iter(ranked_requesters), iter(ranked_providers)
        requester, provider = next(requester_iter, None), next(provider_iter, None)
        while requester is not None and provider is not None and requester.reserve >= provider.reserve:
            pairs.append((requester, provider))
            marginal = (requester.reserve + provider.reserve) / 2
            quantity = min(left[id(requester)], offered[id(provider)])
            left[id(requester)] -= quantity
            offered[id(provider)] -= quantity
            if left[id(requester)] <= 0:
                requester = next(requester_iter, None)
            if offered[id(provider)] <= 0:
                provider = next(provider_iter, None)
        return pairs, marginal

    def _match_batch(self, requesters: list[SimpleNamespace], providers: list[SimpleNamespace]) -> list[MatchSpec]:
        """Return the matches of the crossing offers at the midpoint of the last pair.

        The last pair is the last crossing pair.
        """
        pairs, marginal = self._get_pair(requesters, providers)
        trades: list[MatchSpec] = []
        for requester, provider in pairs:
            self._record_match(trades, requester, provider, marginal)
        return trades

    @staticmethod
    def _score_provider(rule: StableMatchingSpec, requester: SimpleNamespace, provider: SimpleNamespace) -> float:
        """Return the score of a provider for a requester, higher for a preferred provider."""
        if rule.requester_preference is None:
            return -provider.reserve
        return rule.requester_preference(requester.agent, provider.agent)

    @staticmethod
    def _score_requester(rule: StableMatchingSpec, provider: SimpleNamespace, requester: SimpleNamespace) -> float:
        """Return the score of a requester for a provider.

        The score is higher for a preferred requester.
        """
        if rule.provider_preference is None:
            return requester.reserve
        return rule.provider_preference(provider.agent, requester.agent)

    def _rank_providers(
        self,
        rule: StableMatchingSpec,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
    ) -> dict[int, list[int]]:
        """Return the ids of the acceptable providers of every requester.

        The preferred come first.
        """
        capacities = {provider.agent.id for provider in providers if int(provider.quantity) > 0}
        ranks = {}
        for requester in requesters:
            acceptable = [
                provider
                for provider in providers
                if provider.agent.id in capacities and provider.reserve <= requester.reserve
            ]
            ranked = sorted(
                (-self._score_provider(rule, requester, provider), provider.agent.id) for provider in acceptable
            )
            if ranked:
                ranks[requester.agent.id] = [identity for _, identity in ranked]
        return ranks

    def _rank_requesters(
        self,
        rule: StableMatchingSpec,
        ranks: dict[int, list[int]],
        providers: list[SimpleNamespace],
        by_id: dict[int, SimpleNamespace],
    ) -> dict[int, list[int]]:
        """Return the ids of the requesters that rank every provider, the preferred first."""
        provider_ranks = {}
        for provider in providers:
            suitors = sorted(
                (-self._score_requester(rule, provider, by_id[requester_id]), requester_id)
                for requester_id, ranked in ranks.items()
                if provider.agent.id in ranked
            )
            if suitors:
                provider_ranks[provider.agent.id] = [identity for _, identity in suitors]
        return provider_ranks

    @staticmethod
    def _defer_acceptance(
        requester_ranks: dict[int, list[int]],
        provider_ranks: dict[int, list[int]],
        capacities: dict[int, int],
    ) -> dict[int, list[int]]:
        """Return the requesters of every provider in the stable matching.

        The matching is the requester-optimal one, and the preferred requesters come first.

        Every free requester proposes to its next provider, which holds its preferred proposals up to its capacity and
        rejects the rest. The result does not depend on the order of the proposals.
        """
        held: dict[int, list[int]] = {provider: [] for provider in provider_ranks}
        proposals = {requester: iter(ranked) for requester, ranked in requester_ranks.items()}
        free = list(requester_ranks)
        while free:
            requester = free.pop()
            provider = next(proposals[requester], None)
            if provider is None:
                continue
            held[provider].append(requester)
            if len(held[provider]) > capacities[provider]:
                rejected = max(held[provider], key=provider_ranks[provider].index)
                held[provider].remove(rejected)
                free.append(rejected)
        return {provider: sorted(held[provider], key=ranked.index) for provider, ranked in provider_ranks.items()}

    def _match_stable(
        self,
        rule: StableMatchingSpec,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
    ) -> list[MatchSpec]:
        """Return the matches of the requester-optimal stable matching, one unit per pair."""
        by_id = {offer.agent.id: offer for offer in itertools.chain(requesters, providers)}
        requester_ranks = self._rank_providers(rule, requesters, providers)
        if not requester_ranks:
            return []
        provider_ranks = self._rank_requesters(rule, requester_ranks, providers, by_id)
        capacities = {provider_id: int(by_id[provider_id].quantity) for provider_id in provider_ranks}
        trades: list[MatchSpec] = []
        for provider_id, requester_ids in self._defer_acceptance(requester_ranks, provider_ranks, capacities).items():
            provider = by_id[provider_id]
            for requester_id in requester_ids:
                requester = by_id[requester_id]
                value = self._get_value(requester, provider, rule.value)
                quantity = min(1.0, MatchingInteraction._get_affordable(requester, value), provider.quantity)
                if quantity > 0:
                    requester.quantity -= quantity
                    requester.cap -= quantity * value
                    provider.quantity -= quantity
                    trades.append(MatchSpec(requester.agent, provider.agent, quantity, value))
        return trades

    def _match_partners(
        self,
        rule: PartnerSpec,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of every requester with the feasible partners its rule picks.

        The requesters go in random order.
        """
        wanting = {id(offer.agent): offer for offer in requesters}
        offering = {id(offer.agent): offer for offer in providers}

        def is_feasible(requester: AgentViewProtocol, provider: AgentViewProtocol) -> bool:
            """Return whether a provider still offers a quantity at an acceptable reserve.

            The requester accepts the reserve.
            """
            offer = offering[id(provider)]
            return offer.quantity > 0 and offer.reserve <= wanting[id(requester)].reserve

        trades: list[MatchSpec] = []
        left, right = [offer.agent for offer in requesters], [offer.agent for offer in providers]
        for requester, provider in MatchLaw(self.requesters, self.providers, rule).iterate_pairs(
            left,
            right,
            rng,
            is_feasible,
        ):
            first, second = wanting[id(requester)], offering[id(provider)]
            value = rule.value if isinstance(rule, PartnerMatchingSpec) else 'midpoint'
            self._record_match(trades, first, second, self._get_value(first, second, value))
        return trades

    def _match_function(
        self,
        rule: Pairing,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of the pairs of a function.

        Each is at the midpoint of the reserves.
        """
        wanting = {id(offer.agent): offer for offer in requesters}
        offering = {id(offer.agent): offer for offer in providers}
        trades: list[MatchSpec] = []
        left = [offer.agent for offer in requesters]
        right = left if self.requesters == self.providers else [offer.agent for offer in providers]
        for requester, provider in MatchLaw(self.requesters, self.providers, rule).get_pairs(left, right, rng):
            first, second = wanting[id(requester)], offering[id(provider)]
            self._record_match(trades, first, second, self._get_value(first, second, 'midpoint'))
        return trades

    def _get_matches(
        self,
        requesters: list[SimpleNamespace],
        providers: list[SimpleNamespace],
        system: SystemT,
    ) -> list[MatchSpec]:
        """Return the matches of the offers by the algorithm the rule names.

        There are none when a side has no offers.
        """
        if not requesters or not providers:
            return []
        rule, rng = self.rule, system.rng
        match rule:
            case RandomSearchSpec():
                matches = self._match_random_search(rule, requesters, providers, rng)
            case PartnerSwitchSpec():
                matches = self._match_partner_switch(rule, requesters, providers, rng)
            case QueueSpec():
                matches = self._match_queue(rule, requesters, providers, rng)
            case IterativeMatchingSpec():
                matches = self._match_iterative(rule, requesters, providers, rng, system.storage)
            case ContinuousMatchingSpec():
                matches = self._match_continuous(rule, requesters, providers, rng)
            case BatchMatchingSpec():
                matches = self._match_batch(requesters, providers)
            case StableMatchingSpec():
                matches = self._match_stable(rule, requesters, providers)
            case PartnerSpec():
                matches = self._match_partners(rule, requesters, providers, rng)
            case _:
                matches = self._match_function(rule, requesters, providers, rng)
        return matches

    def _settle(self, match: MatchSpec, system: SystemT) -> None:
        """Move the transfers of a match, lower both quantities, and run `on_match`.

        It lowers the quantities of both sides.
        """
        for state, sign in self.transfers:
            if sign < 0:
                transfer(match.requester, match.provider, match.quantity * match.value, state)
            else:
                transfer(match.provider, match.requester, match.quantity, state)
        for agent, name in zip((match.requester, match.provider), self._get_sides(self.quantity), strict=True):
            if hasattr(agent, name):
                setattr(agent, name, getattr(agent, name) - match.quantity)
        if self.on_match is not None:
            self.on_match(match, system)

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, system: SystemT) -> None:
        """Report the value, the matched volume, and the unmatched quantity of a clearing.

        The value is weighted by quantity.

        Args:
            matches:
                The matches of the clearing.

            unmatched:
                The quantity of both sides left after the clearing.

            system:
                The system view of the matching.
        """
        volume = sum(match.quantity for match in matches)
        value = sum(match.quantity * match.value for match in matches) / volume if volume else math.nan
        system.report('value', value)
        system.report('volume', volume)
        system.report('unmatched', unmatched)

    def _check_space(self, space: BaseSpace | None) -> None:
        """Raise an error for candidates in neighbouring cells or nodes.

        The error is for a continuous space.
        """
        self._check_partners(self.rule, space)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for a bad declaration, an unsigned transfer, or a bad rule.

        A bad declaration is one the matching cannot take.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                A transfer has a sign other than `-1` or `1`, the rule is neither a matching spec nor a function, or
                it is invalid.
        """
        super()._prepare(context)
        for state, sign in self.transfers:
            if sign not in SIGNS:
                error_msg = f'The transfer of {state!r} has the sign {sign!r}. Use one of {SIGNS}.'
                raise ValidationError(error_msg)
        self._check_variant('rule', self.rule, (), RULES, functions=True)
        if isinstance(self.rule, RULES):
            self.rule._check()

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the rule that matches the members in the matching's phase.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return [RuleLaw(self.match, phase=self.phase)]

    def match(self, system: SystemT) -> None:
        """Match the members, settle the matches, and report the results.

        It reports the value, the volume, and the unmatched quantity.

        Args:
            system:
                The system view of the matching.
        """
        requesters, providers = self._get_offers(system)
        matches = self._get_matches(requesters, providers, system)
        for match in matches:
            self._settle(match, system)
        unmatched = sum(offer.quantity for offer in requesters + providers)
        self._report_matches(matches, unmatched, system)
