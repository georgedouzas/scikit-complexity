"""Lotka-Volterra predation, where encounters remove prey and feed predators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, RateLaw
from .....core.specs import ParameterSpec, RoleSpec


class PredationInteraction(Interaction):
    """Lotka-Volterra predation, where encounters remove prey and feed predators.

    The prey loses `a * predator * prey` per unit time and the predator gains `b * predator * prey`.

    Args:
        label:
            The label of the interaction.

        predator:
            The path of the predator population.

        prey:
            The path of the prey population.

        a:
            The attack rate.

        b:
            The conversion rate of eaten prey into predators.

    Examples:
        >>> from skcomplex.systems.biology.ecology import PredationInteraction
        >>> PredationInteraction('hunt', predator='foxes', prey='rabbits', a=0.4, b=0.1).prey
        'rabbits'
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        predator: Annotated[str, RoleSpec()],
        prey: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label)
        self.predator = predator
        self.prey = prey
        self.a = a
        self.b = b

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates `-a * predator * prey` and `b * predator * prey`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        encounters = s.predator.x * s.prey.x
        return [RateLaw(s.prey.x, -s.a * encounters), RateLaw(s.predator.x, s.b * encounters)]
