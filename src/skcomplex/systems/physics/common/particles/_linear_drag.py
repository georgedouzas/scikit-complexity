"""Define linear drag.

It slows a moving mass in proportion to its velocity.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.spaces import BaseSpace
from .....core.types import Role


class LinearDragInteraction(Interaction):
    """Linear drag, a force `-gamma v` against the velocity `v` of a mass `m`.

    Every coordinate `x` of the target follows `x'' = -gamma x' / m`. It is air resistance on a slow ball, the
    friction of a pendulum bob, or the dashpot of a car suspension.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the mass.

        gamma:
            The drag coefficient.

        coordinates:
            The names of the coordinates of the target that the drag slows, such as `('x', 'y')` in a plane.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.particles import LinearDragInteraction
        >>> ball = Element('ball', values={'x': 0.0, 'm': 1.0})
        >>> system = System(elements=[ball], interactions=[LinearDragInteraction('air', target='ball', gamma=0.5)])
        >>> trajectory = system.prepare().simulate(until=2.0, dt=1.0, initial_state={"ball.x'": 1.0})
        >>> round(float(trajectory.get_values("ball.x'")[-1]), 3)
        0.368
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        gamma: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        coordinates: Sequence[str] = ('x',),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma
        self.coordinates = coordinates

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of every coordinate of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            t, target = context.t, context.target
            positions = [getattr(target, name) for name in self.coordinates]
            self.laws_ = [RateLaw(x.diff(t, 2), -context.gamma * x.diff(t) / target.m) for x in positions]
