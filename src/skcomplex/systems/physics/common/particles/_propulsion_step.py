"""Define propulsion steps.

A step moves a position along a heading by a constant distance every step.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, cos, oo, sin, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import UpdateLaw
from .....core.components.spaces import BaseSpace
from .....core.types import Role


class PropulsionStepInteraction(Interaction):
    """Step `x(t + 1) = x + speed (cos theta, sin theta)` of a position in the plane.

    The step is along a heading, in discrete time.

    The step reads the heading at the start of the step, as every map of the step does.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the particle or the group whose positions move.

        speed:
            The distance moved in one step.

        state:
            The name of the vector state that holds the position.

        heading:
            The name of the state that holds the heading.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var, vector
        >>> from skcomplex.systems.physics.common.particles import PropulsionStepInteraction
        >>> target = SimpleNamespace(x=vector([var('x1'), var('x2')]), theta=var('theta'))
        >>> step = PropulsionStepInteraction('propulsion', target='flock')
        >>> [law.value for law in step.prepare(SimpleNamespace(target=target, speed=var('v'), t=var('t'))).laws_]
        [(v*cos(theta) + x1, v*sin(theta) + x2)]
    """

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

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the step of the position along the heading.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the target, the time `t`,
                and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            position = getattr(context.target, self.state)
            self.laws_ = [UpdateLaw(position, position + self._step(context), increment=True)]

    def _step(self, namespace: SimpleNamespace) -> FreeModuleElement:
        """Return the step `speed (cos theta, sin theta)`."""
        heading = getattr(namespace.target, self.heading)
        return namespace.speed * vector(SR, [cos(heading), sin(heading)])
