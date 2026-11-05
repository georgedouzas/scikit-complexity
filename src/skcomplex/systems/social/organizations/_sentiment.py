"""Define sentiments, which follow events on topics and the partner each resident chooses, after Zanbaka et al."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated, cast

import numpy as np

from ....core.components.interactions import MapInteraction
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.specs import RoleSpec
from ._resident_view import ResidentView

AGE_RANGE = 100.0
SENTIMENT_RANGE = 2.0
DEMOGRAPHIC_TERMS = 5


class SentimentInteraction(MapInteraction):
    """Sentiment of every resident, which responds to events and to the partner the resident chooses at each step.

    The rules are those of Zanbaka et al. in Carmichael et al. (2019 ch. 3 §4.3). An event is one or more entries
    `(time, topic, weight)`, and it changes the sentiment `S` to `S + sum(mu A_T (E_T - S))` over its topics, with `mu`
    the parameter `event_rate` of the resident, `A_T` its weight of the topic, and `E_T` the weight of the event
    (Eq. 5).
    The partner is the most likely of at most `partners` residents drawn from the grid cell of the resident, by
    `P = DSS / 2 + STI / 2`, with `DSS` the mean of five similarities and `STI` the share of topics both weigh (Eqs. 1
    and 2). The two sentiments then change at once to `a1 + mu1 (a2 - a1)` and `a2 + mu2 (a1 - a2)`, with
    `mu = 1 - |a| CR` of each and `CR` its parameter `convergence` (Eq. 4). The chapter prints no definition of the
    terms of `DSS`, so here race, gender, and occupation count 1 when equal and 0 otherwise, and age and sentiment count
    one minus their distance over their range. A sentiment stays between -1 and 1.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of residents.

        events:
            The entries of the events as `(time, topic, weight)`.

    Examples:
        >>> from skcomplex.systems.social.organizations import SentimentInteraction
        >>> SentimentInteraction('sentiment', target='residents', events=[(1, 0, 0.5)]).events
        [(1, 0, 0.5)]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        events: Sequence[tuple[int, int, float]] = (),
    ) -> None:
        super().__init__(
            label,
            target=target,
            state='sentiment',
            function=self._update,
            schedule='member',
            form='agent',
        )
        self.events = events

    def _respond_to_events(self, resident: ResidentView, time: int) -> float:
        """Return the sentiment of a resident after the events of a step, by Eq. 5."""
        sentiment = resident.sentiment
        changes = [
            resident.event_rate * resident.weights[topic] * (weight - sentiment)
            for when, topic, weight in self.events
            if when == time
        ]
        return float(np.clip(sentiment + sum(changes), -1.0, 1.0))

    @staticmethod
    def _read_probability(first: ResidentView, second: ResidentView) -> float:
        """Return the probability of interaction of two residents, by Eqs. 1 and 2."""
        similarities = (
            float(first.race == second.race),
            float(first.gender == second.gender),
            float(first.occupation == second.occupation),
            1 - abs(first.age - second.age) / AGE_RANGE,
            1 - abs(first.sentiment - second.sentiment) / SENTIMENT_RANGE,
        )
        shared = np.mean((np.asarray(first.weights) > 0) & (np.asarray(second.weights) > 0))
        return float(sum(similarities) / DEMOGRAPHIC_TERMS / 2 + shared / 2)

    @staticmethod
    def _read_grid_cell(resident: ResidentView) -> tuple[int, ...]:
        """Return the grid cell of the location of a resident."""
        return tuple(np.floor(np.asarray(resident.location, dtype=float) / resident.cell_size).astype(int).tolist())

    def _choose_partner(self, resident: ResidentView, system: SystemViewProtocol) -> ResidentView | None:
        """Return the most likely of the residents drawn from the grid cell of a resident, or `None` without any."""
        diagonal = resident.cell_size * np.sqrt(2.0)
        cell = self._read_grid_cell(resident)
        around = [cast(ResidentView, other) for other in resident.neighbors(diagonal)]
        here = [other for other in around if self._read_grid_cell(other) == cell]
        count = min(int(resident.partners), len(here))
        if count == 0:
            return None
        drawn = [here[int(index)] for index in system.rng.choice(len(here), size=count, replace=False)]
        return max(drawn, key=lambda other: self._read_probability(resident, other))

    def _update(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the sentiment of a resident after the step, having also set the sentiment of its partner."""
        resident = cast(ResidentView, agent)
        own = self._respond_to_events(resident, int(system.time))
        partner = self._choose_partner(resident, system)
        if partner is None:
            return own
        other = partner.sentiment
        partner.sentiment = other + (1 - abs(other) * partner.convergence) * (own - other)
        return own + (1 - abs(own) * resident.convergence) * (other - own)
