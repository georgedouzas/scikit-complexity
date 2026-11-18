"""Thermal noise of overdamped particles.

The noise is the random force `sqrt(2 gamma temperature) dW` divided by `gamma`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import sqrt

from .....core.components.interactions import Interaction
from .....core.components.laws import NoiseLaw
from .....core.components.spaces import BaseSpace
from .....core.specs import Role


class ThermalNoiseInteraction(Interaction):
    """Thermal noise of overdamped particles, the random force `sqrt(2 gamma temperature) dW` over `gamma`.

    The friction `gamma` and the thermal energy `temperature` are values of the target, so every component of the
    position diffuses with the coefficient `temperature / gamma`. It is a bead in water, a protein in a cell, or a
    pollen grain under a microscope.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group.

        state:
            The name of the position of the target.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import ThermalNoiseInteraction
        >>> ThermalNoiseInteraction('bath', target='beads').state
        'x'
    """

    law_classes = (NoiseLaw,)

    def __init__(self, label: str, *, target: Role, state: str = 'x') -> None:
        super().__init__(label)
        self.target = target
        self.state = state

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noise of the position of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            target = context.target
            amplitude = sqrt(2 * target.gamma * target.temperature) / target.gamma
            self.laws_ = [NoiseLaw(getattr(target, self.state), amplitude)]
