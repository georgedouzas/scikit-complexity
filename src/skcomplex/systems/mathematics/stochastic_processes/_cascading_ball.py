"""Ball of a cascading sample-space-reducing process, which multiplies into `mu` balls at every state it hits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._cascading_ball_view import CascadingBallView


class CascadingBallElement(Element):
    """Ball of a cascading sample-space-reducing process, which multiplies into `mu` balls at every state it hits.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial state.

        mu:
            The mean number of balls a ball becomes at every jump, the multiplication factor.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.stochastic_processes import CascadingBallElement
        >>> group = GroupElement('balls', element=CascadingBallElement, size=5, values={'mu': 2.0})
        >>> group.label, group.element.__name__, group.size
        ('balls', 'CascadingBallElement', 5)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'level'}),
        *,
        level: Annotated[float, StateSpec(domain=NN)] = 100.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.mu = mu

    @classmethod
    def _get_view(cls) -> type[CascadingBallView]:
        """Return the view of the agents, which types their states and parameters."""
        return CascadingBallView

    @staticmethod
    def stops(agent: CascadingBallView, system: SystemView) -> bool:
        """Return whether a ball has reached the ground state 1, where its cascade ends.

        Args:
            agent:
                The ball.

            system:
                The system.

        Returns:
            stopped:
                Whether the ball is at state 1.
        """
        return agent.level <= 1

    @staticmethod
    def multiplies(agent: CascadingBallView, system: SystemView) -> int:
        """Return the number of balls a ball becomes, `floor(mu)` and one more with the probability of the rest.

        Args:
            agent:
                The ball.

            system:
                The system.

        Returns:
            count:
                The number of new balls, of mean `mu`.
        """
        whole = math.floor(agent.mu)
        return whole + int(system.rng.random() < agent.mu - whole)

    @staticmethod
    def jumps(agent: CascadingBallView, system: SystemView) -> dict[str, float]:
        """Return the state of a new ball, drawn uniformly below the state of its parent.

        Args:
            agent:
                The parent ball.

            system:
                The system.

        Returns:
            values:
                The state of the new ball.
        """
        return {'level': float(system.rng.integers(1, round(agent.level)))}
