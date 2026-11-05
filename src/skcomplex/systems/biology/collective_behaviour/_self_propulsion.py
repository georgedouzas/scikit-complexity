"""Rayleigh self-propulsion, `v' = (alpha - beta |v|²) v`, which drives points to the speed `sqrt(alpha / beta)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import RateInteraction
from ....core.specs import ParameterSpec, RoleSpec


class SelfPropulsionInteraction(RateInteraction):
    """Rayleigh self-propulsion, `v' = (alpha - beta |v|²) v`, which drives points to the speed `sqrt(alpha / beta)`.

    It is a rate on the position `x`, a state of order 2, whose velocity `v` is the first derivative `x_d1`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or group.

        alpha:
            The propulsion.

        beta:
            The friction at high speed.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import SelfPropulsionInteraction
        >>> SelfPropulsionInteraction('flight', target='birds', alpha=1.5).alpha
        1.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, target=target, state='x')
        self.alpha = alpha
        self.beta = beta

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the velocity change `(alpha - beta |v|²) v`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespace of the target.

        Returns:
            rate:
                The change of the velocity.
        """
        velocity = namespace.target.x_d1
        return (namespace.alpha - namespace.beta * velocity.dot_product(velocity)) * velocity
