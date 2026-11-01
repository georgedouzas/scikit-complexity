"""Gyroscope in a gimbal: a rotor in an inner ring that turns in an outer ring about a fixed axis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class GyroscopeElement(Element):
    """Gyroscope in a gimbal: a rotor in an inner ring that turns in an outer ring about a fixed axis.

    The angle `alpha` of the outer ring, `beta` of the inner ring, and `gamma` of the rotor have order 2. The motion
    comes from `GyroscopeInteraction`.

    Args:
        label:
            The label of the gyroscope.

        visual:
            How the gyroscope is drawn, or `None` for none.

        alpha:
            The initial angle of the outer ring.

        alpha_d1:
            The initial angular velocity of the outer ring.

        beta:
            The initial angle of the inner ring.

        beta_d1:
            The initial angular velocity of the inner ring.

        gamma:
            The initial angle of the rotor.

        gamma_d1:
            The initial angular velocity of the rotor.

        A:
            The equatorial moment of inertia of the rotor.

        C:
            The polar moment of inertia of the rotor.

        A1:
            The first principal moment of inertia of the inner ring.

        B1:
            The second principal moment of inertia of the inner ring.

        C1:
            The third principal moment of inertia of the inner ring.

        A2:
            The moment of inertia of the outer ring about its axis.

        torque:
            The torque on the axis of the inner ring.

    Examples:
        >>> from skcomplex.models.physics.mechanics import GyroscopeElement
        >>> GyroscopeElement('gyroscope', gamma_d1=10.0).gamma_d1
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alpha: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.0,
        alpha_d1: float = 0.0,
        beta: Annotated[float, StateSpec(order=2)] = 0.3,
        beta_d1: float = 0.0,
        gamma: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.0,
        gamma_d1: float = 0.0,
        A: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        C: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        A1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        B1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        C1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        A2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        torque: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.alpha = alpha
        self.alpha_d1 = alpha_d1
        self.beta = beta
        self.beta_d1 = beta_d1
        self.gamma = gamma
        self.gamma_d1 = gamma_d1
        self.A = A
        self.C = C
        self.A1 = A1
        self.B1 = B1
        self.C1 = C1
        self.A2 = A2
        self.torque = torque
