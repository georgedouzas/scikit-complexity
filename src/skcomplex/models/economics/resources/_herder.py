"""Herder who adds a cow whenever a cow earned more than its upkeep, and sells one when the herd exceeds its quota."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._herder_view import HerderView

UNLIMITED = 1e9


class HerderElement(Element):
    """Herder who adds a cow whenever a cow earned more than its upkeep, and sells one when the herd exceeds its quota.

    The life cycle `investment` of the model buys a cow, at a random cell, or sells the first cow the herder owns,
    and two maps clear the income and the herd the grazing tallies, all member interactions in the phase of herding.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        income:
            The milk income of the herder in the last step.

        herd:
            The number of cows of the herder in the last step.

        upkeep:
            The cost of keeping a cow for a step.

        newborn:
            The energy of a new cow.

        quota:
            The largest herd of a herder, an adaptive parameter that a regulator may change.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.resources import HerderElement
        >>> GroupElement('herders', element=HerderElement, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        income: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        herd: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        upkeep: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        newborn: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        quota: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo), adaptive=True)] = UNLIMITED,
    ) -> None:
        super().__init__(label, visual=visual)
        self.income = income
        self.herd = herd
        self.upkeep = upkeep
        self.newborn = newborn
        self.quota = quota

    @classmethod
    def _get_view(cls) -> type[HerderView]:
        """Return the view of the agents of the element."""
        return HerderView
