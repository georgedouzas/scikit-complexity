"""Define the restoring force.

It pulls a mass back to a center in proportion to its distance.
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
from .....core.specs import Role


class RestoringForceInteraction(Interaction):
    """Restoring force `-k (x - center)` on a mass `m`, from the potential `k |x - center|² / 2`.

    Every coordinate `x` of the target follows `x'' = -k (x - center) / m`. It is a mass on a spring tied to a wall, a
    pendulum swinging through small angles, or an atom held in a crystal.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the mass.

        k:
            The stiffness.

        center:
            The center, the same for every coordinate.

        coordinates:
            The names of the coordinates of the target that the force pulls.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.particles import RestoringForceInteraction
        >>> ball = Element('ball', values={'x': 1.0, 'm': 1.0})
        >>> spring = RestoringForceInteraction('spring', target='ball', k=1.0)
        >>> trajectory = System(elements=[ball], interactions=[spring]).prepare().simulate(until=3.0, dt=3.0)
        >>> round(float(trajectory.get_values('ball.x')[-1]), 3)
        -0.99
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        k: Annotated[float, RealSet.open(0, oo)] = 1.0,
        center: float = 0.0,
        coordinates: Sequence[str] = ('x',),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k
        self.center = center
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
            self.laws_ = [RateLaw(x.diff(t, 2), -context.k * (x - context.center) / target.m) for x in positions]
