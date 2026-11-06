"""Define damping interactions, which slow a velocity in proportion to itself."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import RateInteraction
from .....core.specs import ParameterSpec, RoleSpec
from ._utils import velocity_of


class DampingInteraction(RateInteraction):
    """Linear damping, a force `-gamma * v` proportional and opposite to the velocity `v` of a target.

    The velocity is the first derivative of the position, a state of order 2, whose second derivative changes by
    `-gamma * v / m` with the mass as divisor. The LAMMPS backend runs it as `fix viscous`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group whose velocity is damped.

        gamma:
            The damping coefficient.

        state:
            The name of the state of order 2 that holds the position of the target.

        divisor:
            The name of the state or parameter that holds the mass of the target.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.physics.common.particles import DampingInteraction
        >>> target = SimpleNamespace(x=var('x'), m=2)
        >>> damping = DampingInteraction('air', target='ball', gamma=0.1).prepare()
        >>> [law.value for law in damping.run(SimpleNamespace(target=target, gamma=var('gamma'))).laws_]
        [-1/2*gamma*derivative(x, 1)]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        state: str = 'x',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=target, state=state, divisor=divisor)
        self.gamma = gamma

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the drag `-gamma * v`.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The drag, before the division by the mass.
        """
        return -namespace.gamma * velocity_of(getattr(namespace.target, self.state))
