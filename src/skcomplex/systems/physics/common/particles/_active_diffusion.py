"""Define the diffusion of active particles.

The position and the orientation of a particle diffuse, and the orientation turns.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import NoiseLaw, RateLaw
from .....core.components.spaces import BaseSpace
from .....core.types import Role


class ActiveDiffusionInteraction(Interaction):
    """Diffusion of the position and the orientation of active particles.

    The position follows `dx = sqrt(2 D_t) dW` and the orientation `dtheta = omega dt + sqrt(2 D_r) dW`. With `omega =
    0` it is an active Brownian particle, and otherwise a chiral one. It is a bacterium, a Janus colloid, or a robot
    that loses its heading.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, whose position is `x` and orientation `theta`.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

        omega:
            The angular velocity of a chiral particle.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.spaces import ContinuousSpace
        >>> from skcomplex.core.engine.solvers import EulerMaruyamaSolver
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.particles import ActiveDiffusionInteraction, ActiveParticleElement
        >>> plane = ContinuousSpace(n_dim=2)
        >>> swimmers = ActiveParticleElement('swimmers', x=np.zeros((4000, 2)), size=4000, space=plane)
        >>> diffusion = ActiveDiffusionInteraction('diffusion', target='swimmers', D_r=0.5)
        >>> system = System(elements=[swimmers], interactions=[diffusion], solver=EulerMaruyamaSolver(dt=0.1))
        >>> angles = system.prepare().simulate(until=2.0, dt=2.0, random_state=0).get_values('swimmers.theta')[-1]
        >>> round(float((angles**2).mean()), 1)
        2.0
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        D_t: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        D_r: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        omega: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.D_t = D_t
        self.D_r = D_r
        self.omega = omega

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noise of the position, and the turning and the noise of the orientation of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            theta = s.target.theta
            self.laws_ = [
                NoiseLaw(s.target.x, sqrt(2 * s.D_t)),
                RateLaw(theta.diff(s.t), s.omega),
                NoiseLaw(theta, sqrt(2 * s.D_r)),
            ]
