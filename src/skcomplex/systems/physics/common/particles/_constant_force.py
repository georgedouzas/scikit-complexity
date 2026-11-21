"""Define constant force interactions.

They push a target with a fixed force.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, vector
from sage.modules.free_module_element import FreeModuleElement

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.types import Role


class ConstantForceInteraction(Interaction):
    """Constant force `F` on a particle of mass `m`.

    The position follows `x'' = F / m`. It is gravity near the ground, an electric field on a charge, or the pull of
    optical tweezers far from their center. A number applies to every component of a vector position. The LAMMPS
    solver runs it as `fix addforce`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group of particles.

        F:
            The force vector, or a number for every component.

        state:
            The name of the position of the target.


    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.particles import ConstantForceInteraction
        >>> ball = Element('ball', values={'x': np.array([0.0, 10.0]), 'm': 1.0})
        >>> gravity = ConstantForceInteraction('gravity', target='ball', F=[0.0, -9.81])
        >>> system = System(elements=[ball], interactions=[gravity]).prepare()
        >>> system.simulate(until=1.0, dt=1.0).get_values('ball.x')[-1].round(3).tolist()
        [0.0, 5.095]
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        F: npt.ArrayLike = 0.0,
        state: str = 'x',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.F = F
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
        force = context.F
        if isinstance(x, FreeModuleElement) and not isinstance(force, FreeModuleElement):
            force = vector(SR, [force] * len(x))
        self.laws_ = [RateLaw(x.diff(t, 2), force / target.m)]
