"""Site of the Drossel-Schwabl forest fire: empty at 0, a tree at 1, or burning at 2."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ForestElement(Element):
    """Site of the Drossel-Schwabl forest fire: empty at 0, a tree at 1, or burning at 2.

    A `FireInteraction` grows trees on empty sites with probability `growth`, sets them on fire from a burning
    neighbour or from lightning with probability `lightning`, and empties burning sites.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        cover:
            The initial cover, 0, 1, or 2.

        growth:
            The probability that a tree grows on an empty site per step.

        lightning:
            The probability that lightning strikes a tree per step.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ForestElement
        >>> ForestElement('site').growth
        0.05
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        cover: Annotated[float, StateSpec(domain=NN)] = 0,
        growth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        lightning: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
    ) -> None:
        super().__init__(label, visual=visual)
        self.cover = cover
        self.growth = growth
        self.lightning = lightning
