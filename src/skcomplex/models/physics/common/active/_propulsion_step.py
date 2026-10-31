"""Define propulsion steps, the moves of a position along a heading by a constant distance every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, cos, oo, sin, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from .....core.interactions import MapInteraction
from .....specs import ParameterSpec, RoleSpec


class PropulsionStepInteraction(MapInteraction):
    """Step `x(t + 1) = x + speed (cos theta, sin theta)` of a position in the plane along a heading, in discrete time.

    The step reads the heading at the start of the step, as every map of the step does.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

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
        >>> from skcomplex.models.physics.common.active import PropulsionStepInteraction
        >>> step = PropulsionStepInteraction('propulsion', target='flock').check().prepare()
        >>> target = SimpleNamespace(x=vector([var('x1'), var('x2')]), theta=var('theta'))
        >>> [law.value for law in step.run(SimpleNamespace(target=target, speed=var('v'))).laws_]
        [(v*cos(theta) + x1, v*sin(theta) + x2)]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        state: str = 'x',
        heading: str = 'theta',
    ) -> None:
        super().__init__(label, target=target, state=state, function=self._step, form='increment')
        self.speed = speed
        self.heading = heading

    def _step(self, namespace: SimpleNamespace) -> FreeModuleElement:
        """Return the step `speed (cos theta, sin theta)`."""
        heading = getattr(namespace.target, self.heading)
        return namespace.speed * vector(SR, [cos(heading), sin(heading)])
