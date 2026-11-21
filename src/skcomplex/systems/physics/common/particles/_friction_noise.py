"""Define the thermal noise of overdamped particles.

The noise is the random force of a bath divided by the friction of the particles.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import NoiseLaw
from .....core.types import Role


class FrictionNoiseInteraction(Interaction):
    """Thermal noise of overdamped particles with the friction `gamma`.

    The random force `sqrt(2 gamma temperature) dW` divides by the friction `gamma` of the target, so every component
    of the position `x` diffuses with the coefficient `temperature / gamma`. It is a colloid in water, a bead in an
    optical trap, or a protein in the cytoplasm.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, whose friction is `gamma`.

        temperature:
            The thermal energy of the bath.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.engine.solvers import EulerMaruyamaSolver
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.particles import FrictionNoiseInteraction
        >>> beads = Element('beads', values={'x': 0.0, 'gamma': 2.0}, size=4000)
        >>> bath = FrictionNoiseInteraction('bath', target='beads', temperature=1.0)
        >>> system = System(elements=[beads], interactions=[bath], solver=EulerMaruyamaSolver(dt=0.1)).prepare()
        >>> positions = system.simulate(until=4.0, dt=4.0, random_state=0).get_values('beads.x')[-1]
        >>> round(float((positions**2).mean()), 1)
        4.0
    """

    law_classes = (NoiseLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        temperature: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.temperature = temperature

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the noise of the position of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        gamma = context.target.gamma
        self.laws_ = [NoiseLaw(context.target.x, sqrt(2 * gamma * context.temperature) / gamma)]
