"""Define damping interactions.

They slow a velocity in proportion to itself.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.types import Role


class DampingInteraction(Interaction):
    """Linear damping, a force `-gamma x'` against the velocity `x'` of a particle of mass `m`.

    The position follows `x'' = -gamma x' / m`, component by component for a vector. It is the viscous drag on a bead
    in water, the friction of a sliding block, or the loss of a damped spring. The LAMMPS solver runs it as
    `fix viscous`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group of particles.

        gamma:
            The damping coefficient.

        state:
            The name of the position of the target.


    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.particles import DampingInteraction
        >>> ball = Element('ball', values={'x': 0.0, "x'": 1.0, 'm': 1.0})
        >>> system = System(elements=[ball], interactions=[DampingInteraction('air', target='ball', gamma=0.5)])
        >>> round(float(system.prepare().simulate(until=2.0, dt=1.0).get_values("ball.x'")[-1]), 3)
        0.368
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        gamma: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        state: str = 'x',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma
        self.state = state

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the acceleration of the position of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        t, target = context.t, context.target
        x = getattr(target, self.state)
        self.laws_ = [RateLaw(x.diff(t, 2), -context.gamma * x.diff(t) / target.m)]
