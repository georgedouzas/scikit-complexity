"""Pheromone of one cell of the ground, a level that diffuses to the neighbouring cells and evaporates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class PheromoneElement(Element):
    """Pheromone of one cell of the ground, a level that diffuses to the neighbouring cells and evaporates.

    The element holds the level and its parameters. The system adds the diffusion `D laplacian(level)` and the
    evaporation `-evaporation level` as interactions.

    Args:
        label:
            The label of the element or the lattice field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial level of pheromone.

        D:
            The diffusion coefficient of the pheromone.

        evaporation:
            The rate at which the pheromone evaporates.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import PheromoneElement
        >>> PheromoneElement('pheromone').evaporation
        0.05
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        D: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        evaporation: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.D = D
        self.evaporation = evaporation
