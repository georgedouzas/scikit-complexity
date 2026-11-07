"""Thermal noise of overdamped particles, the random force `sqrt(2 gamma temperature) dW` divided by `gamma`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import sqrt

from .....core.components.interactions import RateInteraction
from .....core.specs import RoleSpec


class ThermalNoiseInteraction(RateInteraction):
    """Thermal noise of overdamped particles, the random force `sqrt(2 gamma temperature) dW` divided by `gamma`.

    The friction `gamma` and the thermal energy `temperature` are parameters of the target, so every component of the
    position diffuses with the coefficient `temperature / gamma`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group.

        state:
            The name of the state that holds the positions.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import ThermalNoiseInteraction
        >>> ThermalNoiseInteraction('bath', target='beads').divisor
        'gamma'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], state: str = 'x') -> None:
        super().__init__(label, target=target, state=state, divisor='gamma')

    def _get_noise(self, namespace: SimpleNamespace) -> object:
        """Return the amplitude `sqrt(2 gamma temperature)` of the random force.

        Args:
            namespace:
                The namespace of the symbols the amplitude can use.

        Returns:
            amplitude:
                The noise amplitude, before the division by the friction.
        """
        target = namespace.target
        return sqrt(2 * target.gamma * target.temperature)
