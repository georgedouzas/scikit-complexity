"""Define matching interactions, which match requesters with providers by a rule and settle every match."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import heapq
import itertools
import math
import warnings
from collections.abc import Callable, MutableMapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from matching.games import HospitalResident

from ...errors import ValidationError
from ...specs import (
    BatchClearingSpec,
    ContinuousClearingSpec,
    GreedyMatchingSpec,
    IterativeClearingSpec,
    MatchSpec,
    PartnerSwitchSpec,
    QueueSpec,
    RandomSearchSpec,
    RoleSpec,
    StableMatchingSpec,
)
from ..base import BasePart
from ..laws import RuleLaw
from ..laws.utils import transfer
from ..types import AgentViewProtocol, MatchingRule, ModelViewProtocol
from ._interaction import Interaction

SIGNS = (-1, 1)
RULES = (
    RandomSearchSpec,
    PartnerSwitchSpec,
    QueueSpec,
    IterativeClearingSpec,
    ContinuousClearingSpec,
    BatchClearingSpec,
    StableMatchingSpec,
    GreedyMatchingSpec,
)


class _Offer:
    """Offer of one participant of a matching during one clearing, which the clearing lowers as it matches."""

    def __init__(self, agent: AgentViewProtocol, limit: float, quantity: float, budget: float = math.inf) -> None:
        self.agent = agent
        self.limit = limit
        self.quantity = quantity
        self.budget = budget

    def get_affordable(self, value: float) -> float:
        """Return the quantity the participant can still match at a value within its budget."""
        if value <= 0:
            return self.quantity
        return max(0.0, min(self.quantity, self.budget / value))


class MatchingInteraction[ModelT: ModelViewProtocol](Interaction):
    """Matching of the members of a group of requesters with those of a group of providers by a rule.

    Requesters state the highest clearing value they accept in their `limit` state, the quantity they want in their
    `quantity` state, and the most value they can give in their `budget` state, and providers the lowest clearing value
    they accept and the quantity they offer. A name given as a pair names the state of the requesters first and of the
    providers second, and a missing state counts as no limit, one unit, and no budget. The rule is a spec that names
    the algorithm that pairs the offers and holds its settings: random search, partner switching, a queue, iterative,
    continuous, or batch clearing, stable matching, or greedy matching. The `price` of a match is its clearing value
    per unit. Every match lowers the quantities of both sides and moves every state of `transfers`: a negative sign
    moves `price * quantity` from the requester to the provider, and a positive sign moves `quantity` from the provider
    to the requester, so the totals stay the same. `on_match(match, model)` adds any other effect. The matching runs
    once per round of its phase and reports the quantity-weighted clearing value `price`, the matched `volume`, and the
    `unmatched` quantity left, as the reporters `<label>.price`, `<label>.volume`, and `<label>.unmatched`.

    Args:
        label:
            The label of the matching.

        requesters:
            The path of the group of requesters.

        providers:
            The path of the group of providers.

        rule:
            The spec of the algorithm that matches the members, such as `ContinuousClearingSpec()`.

        limit:
            The state with the limit of both sides, or a pair of the requesters' and the providers' states.

        quantity:
            The state with the quantity of both sides, or a pair of the requesters' and the providers' states.

        budget:
            The state of a requester with the most value it can give.

        transfers:
            The states every match moves, each with the sign `-1` for a move from the requester to the provider or
            `1` for a move from the provider to the requester.

        phase:
            The phase the matching runs in, or `None` for the default phase.

        on_match:
            A function of every match and the model view, run after the transfers, or `None`.

    Raises:
        ValidationError:
            A transfer has a sign other than `-1` or `1`, or the rule is not a matching spec or is invalid, raised by
            `check`.

    Examples:
        >>> from skcomplex.core.interactions import MatchingInteraction
        >>> from skcomplex.specs import ContinuousClearingSpec
        >>> matching = MatchingInteraction(
        ...     'exchange',
        ...     requesters='traders',
        ...     providers='traders',
        ...     rule=ContinuousClearingSpec(),
        ...     limit=('bid', 'ask'),
        ... )
        >>> matching.check().reports
        ('price', 'volume', 'unmatched')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'volume', 'unmatched')

    def __init__(
        self,
        label: str,
        *,
        requesters: Annotated[str, RoleSpec()],
        providers: Annotated[str, RoleSpec()],
        rule: MatchingRule = RandomSearchSpec(),
        limit: str | tuple[str, str] = 'limit',
        quantity: str | tuple[str, str] = 'quantity',
        budget: str = 'budget',
        transfers: Sequence[tuple[str, int]] = (),
        phase: str | None = None,
        on_match: Callable[[MatchSpec, ModelT], None] | None = None,
    ) -> None:
        super().__init__(label)
        self.requesters = requesters
        self.providers = providers
        self.rule = rule
        self.limit = limit
        self.quantity = quantity
        self.budget = budget
        self.transfers = transfers
        self.phase = phase
        self.on_match = on_match

    @staticmethod
    def _get_sides(name: str | tuple[str, str]) -> tuple[str, str]:
        """Return the state of the requesters and of the providers of a name given once or as a pair."""
        return (name, name) if isinstance(name, str) else name

    @staticmethod
    def _read(agent: AgentViewProtocol, name: str, default: float) -> float:
        """Return a state of an agent, or a default when it has none."""
        return float(getattr(agent, name, default))

    @staticmethod
    def _book(trades: list[MatchSpec], requester: _Offer, provider: _Offer, value: float) -> None:
        """Record a match of the largest quantity both sides can match at a value and lower both offers."""
        quantity = min(requester.get_affordable(value), provider.quantity)
        if quantity <= 0:
            return
        requester.quantity -= quantity
        requester.budget -= quantity * value
        provider.quantity -= quantity
        trades.append(MatchSpec(requester.agent, provider.agent, quantity, value))

    @staticmethod
    def _get_value(requester: _Offer, provider: _Offer, clearing: str) -> float:
        """Return the value of a match under a clearing rule."""
        if clearing == 'requester':
            return requester.limit
        if clearing == 'provider':
            return provider.limit
        return (requester.limit + provider.limit) / 2

    @staticmethod
    def _order(offers: list[_Offer], order: str, rng: np.random.Generator) -> list[_Offer]:
        """Return offers in random order, by decreasing limit, or in order of creation."""
        if order == 'limit':
            return sorted(offers, key=lambda item: (-item.limit, item.agent.id))
        if order == 'sequential':
            return sorted(offers, key=lambda item: item.agent.id)
        return [offers[index] for index in rng.permutation(len(offers))]

    def _get_offers(self, model: ModelT) -> tuple[list[_Offer], list[_Offer]]:
        """Return the offers of the requesters and of the providers with a positive quantity."""
        limits, quantities = self._get_sides(self.limit), self._get_sides(self.quantity)
        requesters = [
            _Offer(
                agent,
                self._read(agent, limits[0], math.inf),
                self._read(agent, quantities[0], 1.0),
                self._read(agent, self.budget, math.inf),
            )
            for agent in model.agents('requesters')
        ]
        providers = [
            _Offer(agent, self._read(agent, limits[1], 0.0), self._read(agent, quantities[1], 1.0))
            for agent in model.agents('providers')
        ]
        return [offer for offer in requesters if offer.quantity > 0], [
            offer for offer in providers if offer.quantity > 0
        ]

    def _match_random_search(
        self,
        rule: RandomSearchSpec,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of requesters in random order that meet random providers and match on crossing limits."""
        trades: list[MatchSpec] = []
        for index in rng.permutation(len(requesters)):
            requester = requesters[index]
            for _ in range(rule.tries):
                available = [provider for provider in providers if provider.quantity > 0]
                if requester.quantity <= 0 or not available:
                    break
                provider = available[int(rng.integers(len(available)))]
                if provider.limit <= requester.limit:
                    self._book(trades, requester, provider, self._get_value(requester, provider, rule.clearing))
        return trades

    @staticmethod
    def _choose_partner(
        rule: PartnerSwitchSpec,
        best: _Offer,
        current: _Offer | None,
        rng: np.random.Generator,
    ) -> _Offer:
        """Return the provider a requester matches with, its current partner or the best of its sample."""
        if current is None or current.quantity <= 0:
            return best
        if best.limit >= current.limit:
            return current
        if best.limit <= 0:
            return best
        switching = 1 - math.exp(rule.intensity * (best.limit - current.limit) / best.limit)
        return best if rng.random() < switching else current

    def _match_partner_switch(
        self,
        rule: PartnerSwitchSpec,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of every requester with its kept or new partner, at the partner's limit."""
        trades: list[MatchSpec] = []
        by_id = {provider.agent.id: provider for provider in providers}
        for index in rng.permutation(len(requesters)):
            requester = requesters[index]
            available = [provider for provider in providers if provider.quantity > 0]
            if not available:
                break
            picks = rng.choice(len(available), size=min(rule.sample, len(available)), replace=False)
            best = min((available[int(pick)] for pick in picks), key=lambda provider: provider.limit)
            current = by_id.get(int(getattr(requester.agent, rule.partner, -1)))
            chosen = self._choose_partner(rule, best, current, rng)
            if hasattr(requester.agent, rule.partner):
                setattr(requester.agent, rule.partner, chosen.agent.id)
            if chosen.limit <= requester.limit:
                self._book(trades, requester, chosen, chosen.limit)
        return trades

    def _match_queue(
        self,
        rule: QueueSpec,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of the queue of requesters served by the lowest providers at their limits."""
        queue = self._order(requesters, rule.order, rng)
        stock = sorted(providers, key=lambda provider: provider.limit)
        trades: list[MatchSpec] = []
        for requester in queue:
            for provider in stock:
                if requester.quantity <= 0 or provider.limit > requester.limit:
                    break
                if provider.quantity > 0:
                    self._book(trades, requester, provider, provider.limit)
        return trades

    @staticmethod
    def _adjust(rule: IterativeClearingSpec, requesters: list[_Offer], providers: list[_Offer], start: float) -> float:
        """Return the value the tâtonnement reaches from a start."""
        value = start
        for _ in range(rule.iterations):
            requested = sum(requester.get_affordable(value) for requester in requesters if requester.limit >= value)
            provided = sum(provider.quantity for provider in providers if provider.limit <= value)
            total = requested + provided
            if not total or abs(requested - provided) / total < rule.tolerance:
                break
            value *= 1 + rule.adjustment * (requested - provided) / total
        return value

    def _clear_at(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        value: float,
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches at one value between the willing requesters and providers in random order."""
        trades: list[MatchSpec] = []
        willing_requesters = self._order([item for item in requesters if item.limit >= value], 'random', rng)
        willing_providers = self._order([item for item in providers if item.limit <= value], 'random', rng)
        pairs = iter(willing_providers)
        provider = next(pairs, None)
        for requester in willing_requesters:
            while provider is not None and requester.get_affordable(value) > 0:
                self._book(trades, requester, provider, value)
                if provider.quantity <= 0:
                    provider = next(pairs, None)
        return trades

    def _match_iterative_clearing(
        self,
        rule: IterativeClearingSpec,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches at the value of the tâtonnement, which starts from the value of the last clearing."""
        limits = [offer.limit for offer in requesters + providers if offer.limit > 0]
        if not limits:
            return []
        last = storage.get('value')
        start = last if isinstance(last, float) and math.isfinite(last) and last > 0 else float(np.median(limits))
        value = self._adjust(rule, requesters, providers, start)
        storage['value'] = value
        return self._clear_at(requesters, providers, value, rng)

    @staticmethod
    def _arrange(
        rule: ContinuousClearingSpec,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
    ) -> list[tuple[_Offer, bool]]:
        """Return the offers in their order of arrival, each with whether it is a request."""
        offers = [(offer, True) for offer in requesters] + [(offer, False) for offer in providers]
        if rule.arrival == 'sequential':
            return sorted(offers, key=lambda item: item[0].agent.id)
        return [offers[index] for index in rng.permutation(len(offers))]

    def _match_continuous_clearing(
        self,
        rule: ContinuousClearingSpec,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of every arriving offer with the best resting offers of the other side."""
        requests: list[tuple[float, int, _Offer]] = []
        provisions: list[tuple[float, int, _Offer]] = []
        trades: list[MatchSpec] = []
        for time, (offer, is_request) in enumerate(self._arrange(rule, requesters, providers, rng)):
            if is_request:
                while provisions and provisions[0][0] <= offer.limit and offer.get_affordable(provisions[0][0]) > 0:
                    resting = provisions[0][2]
                    self._book(trades, offer, resting, resting.limit)
                    if resting.quantity <= 0:
                        heapq.heappop(provisions)
                if offer.get_affordable(offer.limit) > 0:
                    heapq.heappush(requests, (-offer.limit, time, offer))
                continue
            while requests and -requests[0][0] >= offer.limit and offer.quantity > 0:
                resting = requests[0][2]
                self._book(trades, resting, offer, resting.limit)
                if resting.get_affordable(resting.limit) <= 0:
                    heapq.heappop(requests)
            if offer.quantity > 0:
                heapq.heappush(provisions, (offer.limit, time, offer))
        return trades

    @staticmethod
    def _get_requested(offer: _Offer) -> float:
        """Return the quantity a requester can match at its own limit."""
        return offer.quantity if not math.isfinite(offer.budget) else offer.get_affordable(offer.limit)

    def _pair(self, requesters: list[_Offer], providers: list[_Offer]) -> tuple[list[tuple[_Offer, _Offer]], float]:
        """Return the pairs of crossing offers in order of their limits and the value of the last pair."""
        ranked_requesters = sorted(requesters, key=lambda offer: (-offer.limit, offer.agent.id))
        ranked_providers = sorted(providers, key=lambda offer: (offer.limit, offer.agent.id))
        left = {id(offer): self._get_requested(offer) for offer in ranked_requesters}
        offered = {id(offer): offer.quantity for offer in ranked_providers}
        pairs: list[tuple[_Offer, _Offer]] = []
        marginal = math.nan
        requester_iter, provider_iter = iter(ranked_requesters), iter(ranked_providers)
        requester, provider = next(requester_iter, None), next(provider_iter, None)
        while requester is not None and provider is not None and requester.limit >= provider.limit:
            pairs.append((requester, provider))
            marginal = (requester.limit + provider.limit) / 2
            quantity = min(left[id(requester)], offered[id(provider)])
            left[id(requester)] -= quantity
            offered[id(provider)] -= quantity
            if left[id(requester)] <= 0:
                requester = next(requester_iter, None)
            if offered[id(provider)] <= 0:
                provider = next(provider_iter, None)
        return pairs, marginal

    def _match_batch_clearing(self, requesters: list[_Offer], providers: list[_Offer]) -> list[MatchSpec]:
        """Return the matches of the crossing offers at the midpoint of the last crossing pair."""
        pairs, marginal = self._pair(requesters, providers)
        trades: list[MatchSpec] = []
        for requester, provider in pairs:
            self._book(trades, requester, provider, marginal)
        return trades

    @staticmethod
    def _score_provider(rule: StableMatchingSpec, requester: _Offer, provider: _Offer) -> float:
        """Return the score of a provider for a requester, higher for a preferred provider."""
        if rule.requester_preference is None:
            return -provider.limit
        return rule.requester_preference(requester.agent, provider.agent)

    @staticmethod
    def _score_requester(rule: StableMatchingSpec, provider: _Offer, requester: _Offer) -> float:
        """Return the score of a requester for a provider, higher for a preferred requester."""
        if rule.provider_preference is None:
            return requester.limit
        return rule.provider_preference(provider.agent, requester.agent)

    def _rank_providers(
        self,
        rule: StableMatchingSpec,
        requesters: list[_Offer],
        providers: list[_Offer],
    ) -> dict[int, list[int]]:
        """Return the ids of the acceptable providers of every requester, the preferred first."""
        capacities = {provider.agent.id for provider in providers if int(provider.quantity) > 0}
        ranks = {}
        for requester in requesters:
            acceptable = [
                provider
                for provider in providers
                if provider.agent.id in capacities and provider.limit <= requester.limit
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
        providers: list[_Offer],
        by_id: dict[int, _Offer],
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

    def _match_stable(
        self,
        rule: StableMatchingSpec,
        requesters: list[_Offer],
        providers: list[_Offer],
    ) -> list[MatchSpec]:
        """Return the matches of the requester-optimal stable matching, one unit per pair."""
        by_id = {offer.agent.id: offer for offer in itertools.chain(requesters, providers)}
        requester_ranks = self._rank_providers(rule, requesters, providers)
        if not requester_ranks:
            return []
        provider_ranks = self._rank_requesters(rule, requester_ranks, providers, by_id)
        capacities = {provider_id: int(by_id[provider_id].quantity) for provider_id in provider_ranks}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            game = HospitalResident.create_from_dictionaries(requester_ranks, provider_ranks, capacities, clean=True)
            solution = game.solve(optimal='resident')
        trades: list[MatchSpec] = []
        for hospital, residents in solution.items():
            provider = by_id[hospital.name]
            for resident in residents:
                requester = by_id[resident.name]
                value = self._get_value(requester, provider, rule.clearing)
                quantity = min(1.0, requester.get_affordable(value), provider.quantity)
                if quantity > 0:
                    requester.quantity -= quantity
                    requester.budget -= quantity * value
                    provider.quantity -= quantity
                    trades.append(MatchSpec(requester.agent, provider.agent, quantity, value))
        return trades

    @staticmethod
    def _is_candidate(rule: GreedyMatchingSpec, requester: AgentViewProtocol, provider: AgentViewProtocol) -> bool:
        """Check whether a provider is a candidate of a requester."""
        if rule.candidates == 'all':
            return True
        if rule.candidates == 'neighbors':
            return provider.location_key in {requester.location_key, *requester.neighbor_cells()}
        return requester.distance_to(provider) <= float(rule.candidates)

    @staticmethod
    def _get_score(rule: GreedyMatchingSpec, requester: _Offer, provider: _Offer) -> float:
        """Return the score of a provider for a requester, higher for a preferred provider."""
        if rule.score is None:
            return -provider.limit
        return float(rule.score(requester.agent, provider.agent))

    def _get_feasible(
        self,
        rule: GreedyMatchingSpec,
        requester: _Offer,
        providers: list[_Offer],
        taken: dict[int, int],
    ) -> list[_Offer]:
        """Return the providers a requester can still match with."""
        feasible = []
        for provider in providers:
            room = np.inf if rule.capacity is None else float(getattr(provider.agent, rule.capacity))
            if provider.quantity <= 0 or taken.get(id(provider), 0) >= room or provider.limit > requester.limit:
                continue
            if self._is_candidate(rule, requester.agent, provider.agent):
                feasible.append(provider)
        return feasible

    def _match_greedy(
        self,
        rule: GreedyMatchingSpec,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches of every requester, in random order, with its best feasible provider."""
        trades: list[MatchSpec] = []
        taken: dict[int, int] = {}
        for requester in self._order(requesters, 'random', rng):
            feasible = self._get_feasible(rule, requester, providers, taken)
            if not feasible:
                continue
            scores = np.asarray([self._get_score(rule, requester, provider) for provider in feasible], dtype=float)
            best = np.flatnonzero(scores == scores.max())
            provider = feasible[int(best[int(rng.integers(len(best)))])]
            matched = len(trades)
            self._book(trades, requester, provider, self._get_value(requester, provider, rule.clearing))
            if len(trades) > matched:
                taken[id(provider)] = taken.get(id(provider), 0) + 1
        return trades

    def _get_matches(self, requesters: list[_Offer], providers: list[_Offer], model: ModelT) -> list[MatchSpec]:
        """Return the matches of the offers by the algorithm the rule names, none when a side has no offers."""
        if not requesters or not providers:
            return []
        rule, rng = self.rule, model.rng
        match rule:
            case RandomSearchSpec():
                matches = self._match_random_search(rule, requesters, providers, rng)
            case PartnerSwitchSpec():
                matches = self._match_partner_switch(rule, requesters, providers, rng)
            case QueueSpec():
                matches = self._match_queue(rule, requesters, providers, rng)
            case IterativeClearingSpec():
                matches = self._match_iterative_clearing(rule, requesters, providers, rng, model.storage)
            case ContinuousClearingSpec():
                matches = self._match_continuous_clearing(rule, requesters, providers, rng)
            case BatchClearingSpec():
                matches = self._match_batch_clearing(requesters, providers)
            case StableMatchingSpec():
                matches = self._match_stable(rule, requesters, providers)
            case GreedyMatchingSpec():
                matches = self._match_greedy(rule, requesters, providers, rng)
        return matches

    def _settle(self, match: MatchSpec, model: ModelT) -> None:
        """Move the transfers of a match, lower the quantities of both sides, and run `on_match`."""
        for state, sign in self.transfers:
            if sign < 0:
                transfer(match.requester, match.provider, match.quantity * match.price, state)
            else:
                transfer(match.provider, match.requester, match.quantity, state)
        for agent, name in zip((match.requester, match.provider), self._get_sides(self.quantity), strict=True):
            if hasattr(agent, name):
                setattr(agent, name, getattr(agent, name) - match.quantity)
        if self.on_match is not None:
            self.on_match(match, model)

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, model: ModelT) -> None:
        """Report the quantity-weighted price, the matched volume, and the unmatched quantity of one clearing.

        Args:
            matches:
                The matches of the clearing.

            unmatched:
                The quantity of both sides left after the clearing.

            model:
                The model view of the matching.
        """
        volume = sum(match.quantity for match in matches)
        price = sum(match.quantity * match.price for match in matches) / volume if volume else math.nan
        model.report('price', price)
        model.report('volume', volume)
        model.report('unmatched', unmatched)

    def _check(self) -> None:
        """Raise an error for a declaration the matching cannot take, a transfer without a sign, or a bad rule.

        Raises:
            ValidationError:
                A transfer has a sign other than `-1` or `1`, the rule is not a matching spec, or it is invalid.
        """
        super()._check()
        for state, sign in self.transfers:
            if sign not in SIGNS:
                error_msg = f'The transfer of {state!r} has the sign {sign!r}. Use one of {SIGNS}.'
                raise ValidationError(error_msg)
        if not isinstance(self.rule, RULES):
            names = ', '.join(rule.__name__ for rule in RULES)
            error_msg = f'The rule of {self.label!r} is {self.rule!r}. Give one of the matching specs {names}.'
            raise ValidationError(error_msg)
        self.rule.check()

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the rule that matches the members in the matching's phase.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return [RuleLaw(self.match, phase=self.phase)]

    def match(self, model: ModelT) -> None:
        """Match the members, settle the matches, and report the price, the volume, and the unmatched quantity.

        Args:
            model:
                The model view of the matching.
        """
        requesters, providers = self._get_offers(model)
        matches = self._get_matches(requesters, providers, model)
        for match in matches:
            self._settle(match, model)
        unmatched = sum(offer.quantity for offer in requesters + providers)
        self._report_matches(matches, unmatched, model)
