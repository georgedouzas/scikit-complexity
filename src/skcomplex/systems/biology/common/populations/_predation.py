"""Lotka-Volterra predation, where encounters remove prey and feed predators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.spaces import BaseSpace
from .....core.types import Role


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
        predator: Role,
        prey: Role,
        a: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        b: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
    ) -> None:
        super().__init__(label)
        self.predator = predator
        self.prey = prey
        self.a = a
        self.b = b

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates `-a * predator * prey` and `b * predator * prey`.

        Args:
            context:
                The namespace of the bound elements, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            encounters = s.predator.x * s.prey.x
            self.laws_ = [RateLaw(s.prey.x, -s.a * encounters), RateLaw(s.predator.x, s.b * encounters)]
