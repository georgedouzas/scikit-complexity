"""Define harmonic force interactions, which push a target with a force oscillating in time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import RateInteraction
from .....core.specs import ParameterSpec, RoleSpec


class HarmonicForceInteraction(RateInteraction):
    """Periodic force `F cos(omega t)` on a target, the second derivative `F cos(omega t) / m` of its position.

    It is the rate `F cos(omega t)` with the mass as divisor, applied to every component of the position.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or group the force acts on.

        F:
            The amplitude of the force.

        omega:
            The angular frequency of the force.

        state:
            The name of the state of order 2 that holds the position of the target.

        divisor:
            The name of the state or parameter that holds the mass of the target.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import HarmonicForceInteraction
        >>> HarmonicForceInteraction('drive', target='mass', F=0.5, omega=2.0).omega
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        F: Annotated[float, ParameterSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        state: str = 'x',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=target, state=state, divisor=divisor)
        self.F = F
        self.omega = omega

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the force `F cos(omega t)`.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The force, before the division by the mass.
        """
        return namespace.F * cos(namespace.omega * namespace.t)
