"""Define phase oscillators, phases that turn at a natural frequency."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from .....core.components.elements import Element
from .....core.components.laws import BaseLaw, RateLaw
from .....core.specs import ParameterSpec, StateSpec


class PhaseOscillatorElement(Element):
    """Phase oscillator, a phase `theta` that turns at its natural frequency.

    The phase follows `theta' = omega`.

    Couplings to other oscillators add to the rate of the phase.

    Args:
        label:
            The label of the oscillator or the group.

        theta:
            The initial phase.

        omega:
            The natural frequency.

    Attributes:
        default_visual_:
            How the oscillator is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.physics.common.oscillators import PhaseOscillatorElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = CompositeSystem(elements=[PhaseOscillatorElement('clock', omega=2.0)])
        >>> float(simulate(system, until=1.0, dt=0.5).get_values('clock.theta')[-1])
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[float, StateSpec(role='phase')] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the turning of the phase at the natural frequency.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of the phase.
        """
        return [RateLaw(namespace.theta, namespace.omega)]
