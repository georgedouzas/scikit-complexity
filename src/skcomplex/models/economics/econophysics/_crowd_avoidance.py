"""Stochastic crowd avoidance of diners, who leave crowded restaurants."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

from ....core.interactions import ChoiceInteraction
from ....core.types import ModelViewProtocol
from ._diner_view import DinerView


class CrowdAvoidanceInteraction(ChoiceInteraction):
    """Stochastic crowd avoidance of diners, who leave crowded restaurants.

    A diner returns to last evening's restaurant with probability `1 / n`, with `n` the crowd it found there, and
    otherwise moves to one of the other restaurants at random, which raises the utilization to about 0.8. It is a
    Metropolis choice of intensity 1 where every other restaurant has utility 0 and staying has the utility
    `-log(1 - 1 / n)`, so a proposed move is accepted with probability `1 - 1 / n`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of diners.

        options:
            The restaurants, from 0 to their number minus one.

        state:
            The state of the diners that holds their restaurant.

        intensity:
            The intensity of choice, 1 for the stay probability `1 / n`.

        activity:
            The probability that a diner revises its choice in an evening.

        phase:
            The phase the choice runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.economics.econophysics import CrowdAvoidanceInteraction
        >>> CrowdAvoidanceInteraction('choice', target='diners', options=(0.0, 1.0)).check().rule
        'metropolis'
    """

    def __init__(
        self,
        label: str,
        *,
        target: str,
        options: Sequence[float],
        state: str = 'restaurant',
        intensity: float = 1.0,
        activity: float = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            options=options,
            utility=self._get_utility,
            rule='metropolis',
            intensity=intensity,
            activity=activity,
            form='agent',
            phase=phase,
        )

    def _get_utility(self, agent: DinerView, model: ModelViewProtocol, option: float) -> float:
        """Return 0 for another restaurant and `-log(1 - 1 / n)` for staying at a crowd of `n`."""
        if option != getattr(agent, self.state) or agent.crowd < 1:
            return 0.0
        if agent.crowd == 1:
            return math.inf
        return -math.log1p(-1 / agent.crowd)
