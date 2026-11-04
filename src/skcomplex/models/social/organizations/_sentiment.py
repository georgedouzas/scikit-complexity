"""Define sentiments, which follow the events on topics and the residents around."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated, cast

import numpy as np

from ....core.interactions import MapInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import RoleSpec
from ._resident_view import ResidentView


class SentimentInteraction(MapInteraction):
    """Sentiment of every resident, which follows the events on the topics it weighs and the residents around it.

    An event `(time, topic, strength)` adds its strength times the weight of its topic to the sentiment of a resident
    at that step. Every step the sentiment also closes the share `influence` of its difference from the mean sentiment
    of the residents within `radius`, and it stays between -1 and 1.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of residents.

        events:
            The events as `(time, topic, strength)`.

    Examples:
        >>> from skcomplex.models.social.organizations import SentimentInteraction
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

    def _update(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> float:
        """Return the sentiment of a resident after the events of the step and the influence of those around it."""
        resident = cast(ResidentView, agent)
        time = int(model.time)
        value = resident.sentiment + sum(
            strength * resident.weights[topic] for when, topic, strength in self.events if when == time
        )
        others = [cast(ResidentView, other).sentiment for other in resident.neighbors(resident.radius)]
        if others:
            value += resident.influence * (float(np.mean(others)) - resident.sentiment)
        return float(np.clip(value, -1.0, 1.0))
