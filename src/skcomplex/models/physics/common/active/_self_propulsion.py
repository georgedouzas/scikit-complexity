"""Define self-propulsion, the drift of a position along a heading at a constant speed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, cos, oo, sin, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from .....core.interactions import RateInteraction
from .....specs import ParameterSpec, RoleSpec


class SelfPropulsionInteraction(RateInteraction):
    """Self-propulsion `x' += speed (cos theta, sin theta)` of a position in the plane along a heading `theta`.

    The heading is a state of the same target, which other laws turn and diffuse.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the particle or the group whose positions move.

        speed:
            The self-propulsion speed.

        state:
            The name of the vector state that holds the position.

        heading:
            The name of the state that holds the heading.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var, vector
        >>> from skcomplex.models.physics.common.active import SelfPropulsionInteraction
        >>> propulsion = SelfPropulsionInteraction('propulsion', target='swimmers').check().prepare()
        >>> target = SimpleNamespace(x=vector([var('x1'), var('x2')]), theta=var('theta'))
        >>> [law.value for law in propulsion.run(SimpleNamespace(target=target, speed=var('v'))).laws_]
        [(v*cos(theta), v*sin(theta))]
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
        super().__init__(label, target=target, state=state)
        self.speed = speed
        self.heading = heading

    def _get_rate(self, namespace: SimpleNamespace) -> FreeModuleElement:
        """Return the velocity `speed (cos theta, sin theta)`.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The rate of the position.
        """
        heading = getattr(namespace.target, self.heading)
        return namespace.speed * vector(SR, [cos(heading), sin(heading)])
