"""Random walker that carries a deadly disease, of Charbonneau's epidemic model."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._walker_view import WalkerView

HEALTHY, SICK, DECEASED = 0.0, 1.0, 2.0


class WalkerElement(Element):
    """Random walker that carries a deadly disease, of Charbonneau's epidemic model.

    In its own activation, a living walker moves to a neighbouring cell or node with probability `travel` by a
    `MovementInteraction`, and a `TransitionInteraction` whose members act on their cell makes a sick walker infect
    every healthy walker of its cell with probability `infection`. Another transition kills a sick walker after a
    residence of a number of steps. Health is 0 healthy, 1 sick, or 2 dead.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        health:
            The health state.

        infection:
            The probability that a sick walker infects a healthy walker of its cell at every step.

        travel:
            The probability of moving at every step.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.epidemiology import WalkerElement
        >>> GroupElement('walkers', element=WalkerElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 2.0)}),
        *,
        health: Annotated[float, StateSpec(domain=NN)] = HEALTHY,
        infection: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        travel: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health
        self.infection = infection
        self.travel = travel

    @classmethod
    def _get_view(cls) -> type[WalkerView]:
        """Return the view of a walker, `WalkerView`."""
        return WalkerView
