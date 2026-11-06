"""Pedestrian who walks toward a goal, drawn to existing trails, and leaves footprints that build them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._walker_view import WalkerView


class WalkerElement(Element):
    """Pedestrian who walks toward a goal, drawn to existing trails, and leaves footprints that build them.

    The walker heads along the unit vector toward its goal plus `attraction` times the gradient of the trail potential
    `V`, after Helbing, Keltsch, and Molnár. Every step it moves to a neighbouring cell with a probability that grows
    with `exp(directedness * cos angle)`, where the angle is between the step and the heading, and adds the footprint
    `intensity (1 - G / G_max)` to the ground. The step after it arrives at its goal it leaves the green, and a new
    walker enters there with another entry point as goal, drawn at random, a renewal in place. The system adds the
    step, the footprint, the renewal, and the arrival as member interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        goal_x:
            The column of the goal.

        goal_y:
            The row of the goal.

        arrived:
            Whether the walker stood at its goal at the end of the previous step, 1 or 0.

        directedness:
            How strongly the walker heads for its goal.

        attraction:
            How strongly trails draw the walker.

        intensity:
            The footprint on bare ground.

        G_max:
            The largest ground structure.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.collective_behaviour import WalkerElement
        >>> GroupElement('walkers', element=WalkerElement, size=4, location=(0, 0)).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'goal_x'}),
        *,
        goal_x: Annotated[float, StateSpec()] = 0.0,
        goal_y: Annotated[float, StateSpec()] = 0.0,
        arrived: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        directedness: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        attraction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        intensity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        G_max: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
        height: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.arrived = arrived
        self.directedness = directedness
        self.attraction = attraction
        self.intensity = intensity
        self.G_max = G_max
        self.width = width
        self.height = height

    @classmethod
    def _get_view(cls) -> type[WalkerView]:
        """Return the view of a walker, `WalkerView`."""
        return WalkerView
