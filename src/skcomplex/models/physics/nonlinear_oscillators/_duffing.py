"""Displacement and parameters of the forced Duffing oscillator, whose acceleration a `RateInteraction` sets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class DuffingElement(Element):
    """Displacement and parameters of the forced Duffing oscillator, whose acceleration a `RateInteraction` sets.

    The acceleration is `x'' = -delta x' - alpha x - beta x^3 + gamma cos(omega t)`.

    For `alpha < 0 < beta` the potential has two wells, and a strong enough forcing drives the oscillator chaotically
    between them.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        delta:
            The damping.

        alpha:
            The linear stiffness.

        beta:
            The cubic stiffness.

        gamma:
            The amplitude of the forcing.

        omega:
            The frequency of the forcing.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import DuffingElement
        >>> DuffingElement('duffing').alpha
        -1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 1.0,
        x_d1: float = 0.0,
        delta: Annotated[float, ParameterSpec()] = 0.3,
        alpha: Annotated[float, ParameterSpec()] = -1.0,
        beta: Annotated[float, ParameterSpec()] = 1.0,
        gamma: Annotated[float, ParameterSpec()] = 0.5,
        omega: Annotated[float, ParameterSpec()] = 1.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
