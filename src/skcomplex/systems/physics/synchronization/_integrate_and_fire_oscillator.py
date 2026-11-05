"""Integrate-and-fire oscillator that charges as `x' = S - gamma x`, and sends pulses of strength `epsilon`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class IntegrateAndFireOscillatorElement(Element):
    """Integrate-and-fire oscillator that charges as `x' = S - gamma x`, and sends pulses of strength `epsilon`.

    The oscillator charges towards `S / gamma`. Its firing at 1 is a crossing of the group it belongs to, which resets
    the most charged oscillators to 0 and lifts every other one by `epsilon`.

    Args:
        label:
            The label of the oscillator or the group.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial charge.

        S:
            The charging input.

        gamma:
            The leak rate.

        epsilon:
            The strength of a pulse.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import IntegrateAndFireOscillatorElement
        >>> IntegrateAndFireOscillatorElement('oscillator').epsilon
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        S: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.S = S
        self.gamma = gamma
        self.epsilon = epsilon

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the charging of the oscillator.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of the charge.
        """
        s = namespace
        return [RateLaw(s.x, s.S - s.gamma * s.x)]
