"""Marks of one cell, the happiness and fear of prey and the happiness and anger of predators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class EmotionMarksElement(Element):
    """Marks of one cell, the happiness and fear of prey and the happiness and anger of predators.

    Animals with emotions leave their feelings in the cell they stand on, and the marks evaporate at the rate `decay`,
    which the system adds as an evaporation of every mark.

    Args:
        label:
            The label of the element or the lattice field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        prey_happiness:
            The initial happiness left by prey.

        prey_fear:
            The initial fear left by prey.

        predator_happiness:
            The initial happiness left by predators.

        predator_anger:
            The initial anger left by predators.

        decay:
            The rate at which every mark evaporates.

    Examples:
        >>> from skcomplex.systems.biology.ecology import EmotionMarksElement
        >>> EmotionMarksElement('marks').decay
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        prey_happiness: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        prey_fear: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        predator_happiness: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        predator_anger: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        decay: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.prey_happiness = prey_happiness
        self.prey_fear = prey_fear
        self.predator_happiness = predator_happiness
        self.predator_anger = predator_anger
        self.decay = decay
