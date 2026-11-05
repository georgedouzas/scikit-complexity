"""Define phase oscillators, phases that turn at a natural frequency."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from .....core.components.elements import Element
from .....core.components.laws import BaseLaw, RateLaw
from .....core.specs import ParameterSpec, StateSpec, VisualSpec


class PhaseOscillatorElement(Element):
    """Phase oscillator, a phase `theta` that turns at its natural frequency, `theta' = omega`.

    Couplings to other oscillators add to the rate of the phase.

    Args:
        label:
            The label of the oscillator or the group.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        theta:
            The initial phase.

        omega:
            The natural frequency.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.physics.common.oscillators import PhaseOscillatorElement
        >>> system = System(elements=[PhaseOscillatorElement('clock', omega=2.0)])
        >>> float(Simulation(system, dt=0.5).prepare().run(until=1.0).trajectory_.values('clock.theta')[-1])
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(role='phase')] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
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
