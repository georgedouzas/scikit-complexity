"""Displacement of an oscillator that rings freely between periodic impulses that kick its velocity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ImpulsiveOscillatorElement(Element):
    """Displacement of an oscillator that rings freely between periodic impulses that kick its velocity.

    A rate interaction gives the restoring force, and impulse interactions kick the velocity `x_d1` by multiples of the
    amplitude `H` at their times, so the element holds the states and the parameters they read.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        H:
            The amplitude of the impulses, half the jump of the velocity that one impulse of the symmetric train
            causes.

        omega0:
            The natural frequency of a linear spring.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ImpulsiveOscillatorElement
        >>> ImpulsiveOscillatorElement('oscillator').H
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        H: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        omega0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.H = H
        self.omega0 = omega0
