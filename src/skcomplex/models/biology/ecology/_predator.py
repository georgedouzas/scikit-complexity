"""Predator that walks at random, eats a prey of its cell, breeds after a meal, and dies."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._predator_view import PredatorView


class PredatorElement(Element):
    """Predator that walks at random, eats a prey of its cell, breeds after a meal, and dies.

    In its own activation a predator walks to a random neighbouring cell, eats and removes one random prey of the new
    cell, counted in `meals`, breeds with the probability `conversion` after a meal, dies with the probability
    `death`, and digests its meals, by member interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        meals:
            The prey the predator ate in the current step.

        conversion:
            The probability of breeding after eating a prey.

        death:
            The probability of dying at every step.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.ecology import PredatorElement
        >>> GroupElement('foxes', element=PredatorElement, size=50).size
        50
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.4}),
        *,
        meals: Annotated[float, StateSpec(domain=NN)] = 0.0,
        conversion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        death: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.meals = meals
        self.conversion = conversion
        self.death = death

    @classmethod
    def _get_view(cls) -> type[PredatorView]:
        """Return the view of a predator, `PredatorView`."""
        return PredatorView
