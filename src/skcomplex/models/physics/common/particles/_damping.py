"""Define damping interactions, which slow a velocity in proportion to itself."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.interactions import RateInteraction
from .....specs import ParameterSpec, RoleSpec


class DampingInteraction(RateInteraction):
    """Linear damping, a force `-gamma * v` proportional and opposite to the velocity of a target.

    It is the rate `-gamma * v` of the velocity with the mass as divisor, so the velocity changes by `-gamma * v / m`,
    and the LAMMPS backend runs it as `fix viscous`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group whose velocity is damped.

        gamma:
            The damping coefficient.

        state:
            The name of the state that holds the velocity of the target.

        divisor:
            The name of the state or parameter that holds the mass of the target.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.models.physics.common.particles import DampingInteraction
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
        state: str = 'v',
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
        return -namespace.gamma * getattr(namespace.target, self.state)
