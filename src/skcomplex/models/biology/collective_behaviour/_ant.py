"""Ant that searches for food guided by pheromone and carries food home, marking the way with pheromone."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._ant_view import AntView


class AntElement(Element):
    """Ant that searches for food guided by pheromone and carries food home, marking the way with pheromone.

    A searching ant moves to a neighbouring cell with a probability that grows with its pheromone and, by `outward`,
    with its distance from the nest, as a scent of the nest would, and picks up food within a radius of the food. A
    carrying ant walks straight home, deposits pheromone on every cell it enters, and drops the food at the nest. The
    model adds the walk home, the deposit, the search, the pick-up and drop, and the arrival as member interactions,
    and an ant picks up or drops food the step after it arrives.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        carrying:
            Whether the ant carries food.

        arrived:
            Whether the ant stood at the nest with food, or near the food without it, at the end of the previous step.

        nest_x:
            The column of the nest.

        nest_y:
            The row of the nest.

        food_x:
            The column of the food.

        food_y:
            The row of the food.

        food_radius:
            The distance from the food within which ants pick it up.

        attraction:
            How strongly pheromone draws searching ants.

        outward:
            How strongly searching ants move away from the nest.

        deposit:
            The pheromone an ant deposits per step home.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.collective_behaviour import AntElement
        >>> GroupElement('ants', element=AntElement, size=5, location=(0, 0)).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'carrying', 'color_range': (0.0, 1.0)}),
        *,
        carrying: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        arrived: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        nest_x: Annotated[float, ParameterSpec()] = 5.0,
        nest_y: Annotated[float, ParameterSpec()] = 10.0,
        food_x: Annotated[float, ParameterSpec()] = 34.0,
        food_y: Annotated[float, ParameterSpec()] = 10.0,
        food_radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        attraction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        outward: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        deposit: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.carrying = carrying
        self.arrived = arrived
        self.nest_x = nest_x
        self.nest_y = nest_y
        self.food_x = food_x
        self.food_y = food_y
        self.food_radius = food_radius
        self.attraction = attraction
        self.outward = outward
        self.deposit = deposit

    @classmethod
    def _get_view(cls) -> type[AntView]:
        """Return the view of an ant, `AntView`."""
        return AntView
