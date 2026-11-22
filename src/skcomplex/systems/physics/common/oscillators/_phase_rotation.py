"""Define the rotation of phases at their natural frequencies."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.types import Role


class PhaseRotationInteraction(Interaction):
    """Rotation of the phase `theta` of an oscillator at its natural frequency `omega`, `theta' = omega`.

    The target holds the phase and the frequency, one of each per member of a group. Couplings to other oscillators add
    to the rate of the phase, as for a clock, a firefly, or a generator of a power grid.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the oscillator or the group of oscillators.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.oscillators import PhaseRotationInteraction
        >>> clocks = Element('clocks', values={'theta': 0.0, 'omega': [1.0, 2.0]}, size=2)
        >>> system = System(elements=[clocks], interactions=[PhaseRotationInteraction('turn', target='clocks')])
        >>> system.prepare().simulate(until=1.0, dt=0.5).get_values('clocks.theta')[-1].tolist()
        [1.0, 2.0]
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate of the phase of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        oscillator = context.target
        self.laws_ = [RateLaw(oscillator.theta.diff(context.t), oscillator.omega)]
