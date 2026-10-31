"""Define partner switching, in which requesters sample providers and switch to a lower limit with a probability."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import MutableMapping

import numpy as np

from ...specs import MatchSpec
from ._base_matching_protocol import BaseMatchingProtocol
from ._offer import _Offer


class PartnerSwitchProtocol(BaseMatchingProtocol):
    """Partner choice that samples providers and switches to one of a lower limit with some probability.

    The protocol follows chapter 2 of Caiani et al. (2016). Every requester, in random order, samples `sample`
    providers that still offer something. When the lowest limit among them is below the limit of its current partner,
    the requester switches with probability `1 - exp(intensity * (p_new - p_old) / p_new)`, and otherwise stays. It
    matches with its partner at the partner's limit, and the requester's state `partner`, when it has one, holds the
    id of the partner across steps.

    Args:
        sample:
            The number of providers every requester samples.

        intensity:
            The intensity of switching.

        partner:
            The name of the requester state that holds the id of its partner.

    Examples:
        >>> from skcomplex.core.interactions import PartnerSwitchProtocol
        >>> PartnerSwitchProtocol(sample=2, intensity=4.0).check().intensity
        4.0
    """

    def __init__(
        self,
        sample: int = 3,
        intensity: float = 1.0,
        partner: str = 'partner',
    ) -> None:
        self.sample = sample
        self.intensity = intensity
        self.partner = partner

    def _choose(self, best: _Offer, current: _Offer | None, rng: np.random.Generator) -> _Offer:
        """Return the provider a requester matches with, its current partner or the best of its sample."""
        if current is None or current.quantity <= 0:
            return best
        if best.limit >= current.limit:
            return current
        if best.limit <= 0:
            return best
        switching = 1 - math.exp(self.intensity * (best.limit - current.limit) / best.limit)
        return best if rng.random() < switching else current

    def _match(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches of every requester with its kept or new partner.

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
        trades: list[MatchSpec] = []
        by_id = {provider.agent.id: provider for provider in providers}
        for index in rng.permutation(len(requesters)):
            requester = requesters[index]
            available = [provider for provider in providers if provider.quantity > 0]
            if not available:
                break
            picks = rng.choice(len(available), size=min(self.sample, len(available)), replace=False)
            best = min((available[int(pick)] for pick in picks), key=lambda provider: provider.limit)
            current = by_id.get(int(getattr(requester.agent, self.partner, -1)))
            chosen = self._choose(best, current, rng)
            if hasattr(requester.agent, self.partner):
                setattr(requester.agent, self.partner, chosen.agent.id)
            if chosen.limit <= requester.limit:
                self._book(trades, requester, chosen, chosen.limit)
        return trades
