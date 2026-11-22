"""Define self-propulsion.

Self-propulsion is the drift of a position along a heading at a constant speed.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, cos, oo, sin, vector
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.types import Role


class SelfPropulsionInteraction(Interaction):
    """Self-propulsion `x' += speed (cos theta, sin theta)` of a position in the plane.

    The position drifts along a heading `theta`, which other laws turn and diffuse. It is a swimming bacterium, a
    Janus colloid, or a walking robot.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group whose positions move.

        speed:
            The self-propulsion speed.

        state:
            The name of the vector value that holds the position.

        heading:
            The name of the value that holds the heading.

    Examples:
        >>> import math
        >>> from skcomplex.core.components.spaces import ContinuousSpace
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.particles import ActiveParticleElement, SelfPropulsionInteraction
        >>> swimmer = ActiveParticleElement('swimmer', theta=math.pi / 2)
        >>> propulsion = SelfPropulsionInteraction('propulsion', target='swimmer', speed=2.0)
        >>> system = System(elements=[swimmer], interactions=[propulsion], space=ContinuousSpace(n_dim=2))
        >>> trajectory = system.prepare().simulate(until=1.0, dt=1.0)
        >>> trajectory.get_values('swimmer.x')[-1].round(6).tolist()
        [0.0, 2.0]
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        speed: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        state: str = 'x',
        heading: str = 'theta',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.speed = speed
        self.state = state
        self.heading = heading

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the drift of the position of the target along its heading.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        heading = getattr(context.target, self.heading)
        velocity = context.speed * vector(SR, [cos(heading), sin(heading)])
        self.laws_ = [RateLaw(getattr(context.target, self.state), velocity)]
