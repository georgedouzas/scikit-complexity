"""Define stable matching, the deferred acceptance of Gale and Shapley with providers of several units."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import warnings
from collections.abc import MutableMapping

import numpy as np
from matching.games import HospitalResident

from ...specs import MatchSpec
from ..types import Preference
from ._base_matching_protocol import CLEARINGS, BaseMatchingProtocol
from ._offer import _Offer


class StableMatchingProtocol(BaseMatchingProtocol):
    """Stable matching of Gale and Shapley, requesters proposing to providers that take as many as they offer units.

    Requesters rank the providers whose limit they accept by `requester_preference(requester, provider)`, by default
    the lower limit, and providers rank the requesters by `provider_preference(provider, requester)`, by default the
    higher limit. The requester-optimal stable matching, from the hospital-resident solver of the `matching` library,
    pairs every requester with at most one provider, and each pair matches one unit at the value of the clearing rule.

    Args:
        requester_preference:
            The score of a provider for a requester, a function of their agent views, higher for preferred providers,
            or `None` for the lower limit.

        provider_preference:
            The score of a requester for a provider, a function of their agent views, higher for preferred
            requesters, or `None` for the higher limit.

        clearing:
            The value of a match, `'provider'`, `'requester'`, or `'midpoint'`.

    Raises:
        ValidationError:
            The clearing rule is unknown, raised by `check`.

    Examples:
        >>> from skcomplex.core.interactions import StableMatchingProtocol
        >>> StableMatchingProtocol(clearing='requester').check().clearing
        'requester'
    """

    def __init__(
        self,
        requester_preference: Preference | None = None,
        provider_preference: Preference | None = None,
        clearing: str = 'provider',
    ) -> None:
        self.requester_preference = requester_preference
        self.provider_preference = provider_preference
        self.clearing = clearing

    def _check(self) -> None:
        """Check the clearing rule."""
        self._check_option(self.clearing, CLEARINGS, 'clearing')

    def _score_provider(self, requester: _Offer, provider: _Offer) -> float:
        """Return the score of a provider for a requester, higher for a preferred provider."""
        if self.requester_preference is None:
            return -provider.limit
        return self.requester_preference(requester.agent, provider.agent)

    def _score_requester(self, provider: _Offer, requester: _Offer) -> float:
        """Return the score of a requester for a provider, higher for a preferred requester."""
        if self.provider_preference is None:
            return requester.limit
        return self.provider_preference(provider.agent, requester.agent)

    def _rank_providers(self, requesters: list[_Offer], providers: list[_Offer]) -> dict[int, list[int]]:
        """Return the ids of the acceptable providers of every requester, the preferred first."""
        capacities = {provider.agent.id for provider in providers if int(provider.quantity) > 0}
        ranks = {}
        for requester in requesters:
            acceptable = [
                provider
                for provider in providers
                if provider.agent.id in capacities and provider.limit <= requester.limit
            ]
            ranked = sorted((-self._score_provider(requester, provider), provider.agent.id) for provider in acceptable)
            if ranked:
                ranks[requester.agent.id] = [identity for _, identity in ranked]
        return ranks

    def _rank_requesters(
        self,
        ranks: dict[int, list[int]],
        providers: list[_Offer],
        by_id: dict[int, _Offer],
    ) -> dict[int, list[int]]:
        """Return the ids of the requesters that rank every provider, the preferred first."""
        provider_ranks = {}
        for provider in providers:
            suitors = sorted(
                (-self._score_requester(provider, by_id[requester_id]), requester_id)
                for requester_id, ranked in ranks.items()
                if provider.agent.id in ranked
            )
            if suitors:
                provider_ranks[provider.agent.id] = [identity for _, identity in suitors]
        return provider_ranks

    def _match(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches of the stable matching.

        Args:
            requesters:
                The offers of the requesters.

            providers:
                The offers of the providers.

            rng:
                The random generator.

            storage:
                The storage of the matching that persists across clearings.

        Returns:
            trades:
                The matches.
        """
        by_id = {offer.agent.id: offer for offer in itertools.chain(requesters, providers)}
        requester_ranks = self._rank_providers(requesters, providers)
        if not requester_ranks:
            return []
        provider_ranks = self._rank_requesters(requester_ranks, providers, by_id)
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
                value = self._get_value(requester, provider, self.clearing)
                quantity = min(1.0, requester.get_affordable(value), provider.quantity)
                if quantity > 0:
                    requester.quantity -= quantity
                    requester.budget -= quantity * value
                    provider.quantity -= quantity
                    trades.append(MatchSpec(requester.agent, provider.agent, quantity, value))
        return trades
