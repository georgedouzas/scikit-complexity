"""Define damping interactions, which slow a velocity in proportion to itself."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, RateLaw
from ._interaction import Interaction


class DampingInteraction(Interaction):
    """Linear damping, a force `-gamma * v` proportional and opposite to the velocity of a target.

    The velocity changes by `-gamma * v / m`, with `v` and `m` the velocity and mass states the interaction names.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose velocity is damped.

        gamma:
            The damping coefficient.

        velocity:
            The name of the state that holds the velocity of the target.

        mass:
            The name of the state or parameter that holds the mass of the target.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.interactions import DampingInteraction
        >>> target = SimpleNamespace(v=var('v'), m=2)
        >>> damping = DampingInteraction('air', target='ball', gamma=0.1).check().prepare()
        >>> [law.value for law in damping.run(SimpleNamespace(target=target, gamma=var('gamma'))).laws_]
        [-1/2*gamma*v]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        velocity: str = 'v',
        mass: str = 'm',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma
        self.velocity = velocity
        self.mass = mass

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the velocity change `-gamma * v / m`.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law of the velocity of the target.
        """
        velocity = getattr(namespace.target, self.velocity)
        return [RateLaw(velocity, -namespace.gamma * velocity / getattr(namespace.target, self.mass))]
