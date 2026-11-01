"""Regulator that halves the quota of cows per herder when the grass runs short and raises it when it recovers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import FeedbackInteraction
from ....engine.solvers import ModelView
from ....specs import (
    ParameterSpec,
)
from ._herder_view import HerderView
from ._pasture_view import PastureView


class QuotaRegulatorInteraction(FeedbackInteraction[ModelView]):
    """Regulator that halves the quota of cows per herder when the grass runs short and raises it when it recovers.

    The quota is the adaptive parameter `quota` of every herder. It halves, down to one cow, when the grass falls below
    the share `low` of the capacity of the meadow, and grows by one cow when the grass is above the share `high`.
    Without regulation it stays as it is.

    Args:
        label:
            The label of the regulator.

        rule:
            Unused, since the regulator overrides `act`.

        phase:
            The phase the regulator acts in.

        regulated:
            One when the regulator changes the quota, zero otherwise.

        low:
            The share of the capacity of the meadow below which the quota halves.

        high:
            The share of the capacity of the meadow above which the quota rises by one cow.

    Examples:
        >>> from skcomplex.models.economics.resources import QuotaRegulatorInteraction
        >>> QuotaRegulatorInteraction('regulator', regulated=1.0).regulated
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[ModelView], None] | None = None,
        phase: str | None = None,
        regulated: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        low: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        high: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.regulated = regulated
        self.low = low
        self.high = high

    def act(self, model: ModelView) -> None:
        """Change the quota with the share of grass left on the meadow.

        Args:
            model:
                The model view of the regulator.
        """
        if not model.parameter('regulated'):
            return
        pastures = model.agents('meadow', PastureView)
        share = sum(pasture.grass for pasture in pastures) / sum(pasture.capacity for pasture in pastures)
        for herder in model.agents('herders', HerderView):
            if share < model.parameter('low'):
                herder.quota = max(1.0, float(np.floor(herder.quota / 2)))
            elif share > model.parameter('high'):
                herder.quota += 1
