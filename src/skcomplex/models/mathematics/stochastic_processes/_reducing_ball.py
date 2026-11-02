"""Ball of the sample-space-reducing process, whose state the reducing jump moves to a random lower state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._reducing_ball_view import ReducingBallView


class ReducingBallElement(Element):
    """Ball of the sample-space-reducing process, whose state the reducing jump moves to a random lower state.

    With probability `lam` the jump takes the ball uniformly to a state below its own, and otherwise uniformly to any
    state, and the ball restarts from the top state after it reaches state 1.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial state, from 1 to `n_states`.

        n_states:
            The number of states.

        lam:
            The probability of a reducing jump.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.mathematics.stochastic_processes import ReducingBallElement
        >>> group = GroupElement('balls', element=ReducingBallElement, size=3, values={'n_states': 10.0})
        >>> group.label, group.element.__name__, group.size
        ('balls', 'ReducingBallElement', 3)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'level'}),
        *,
        level: Annotated[float, StateSpec(domain=NN)] = 10.0,
        n_states: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.n_states = n_states
        self.lam = lam

    @classmethod
    def _get_view(cls) -> type[ReducingBallView]:
        """Return the view of the agents, which types their states and parameters."""
        return ReducingBallView
