"""Define the periodic force.

It shakes a mass with a force that oscillates in time.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.spaces import BaseSpace
from .....core.types import Role


class PeriodicForceInteraction(Interaction):
    """Periodic force `F cos(omega t)` on a mass `m`.

    Every coordinate `x` of the target follows `x'' = F cos(omega t) / m`. It is the motor that shakes a driven
    oscillator, an alternating voltage on a circuit, or the push of a parent on a swing.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the mass.

        F:
            The amplitude of the force.

        omega:
            The angular frequency of the force.

        coordinates:
            The names of the coordinates of the target that the force drives.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.particles import PeriodicForceInteraction
        >>> ball = Element('ball', values={'x': 0.0, 'm': 2.0})
        >>> force = PeriodicForceInteraction('drive', target='ball', F=2.0, omega=1.0)
        >>> trajectory = System(elements=[ball], interactions=[force]).prepare().simulate(until=1.0, dt=1.0)
        >>> round(float(trajectory.get_values('ball.x')[-1]), 3)
        0.46
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        F: float = 0.0,
        omega: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        coordinates: Sequence[str] = ('x',),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.F = F
        self.omega = omega
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
            force = context.F * cos(context.omega * t)
            self.laws_ = [RateLaw(getattr(target, name).diff(t, 2), force / target.m) for name in self.coordinates]
