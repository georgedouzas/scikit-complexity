"""Control: feedback loops that keep or drive a system towards a goal, and the controllers that close them."""

from ._bang_bang_plant import BangBangPlantElement
from ._cart_pendulum import CartPendulumElement
from ._control import (
    heated_room,
    minimum_time_control,
    pendulum_swing_up,
    pid_loop,
    satellite_transfer,
    thermostat,
)
from ._driven_oscillator import DrivenOscillatorElement
from ._integrator import IntegratorElement
from ._koopman import ControlInputElement, PlantStateElement, koopman_model_predictive_control
from ._pid import PIDInteraction
from ._regulation import lqr_oscillator, pid_inverted_pendulum
from ._state_feedback import StateFeedbackInteraction
from ._thermal import (
    THERMAL,
    ConvectionElement,
    HeatCapacityElement,
    HeatingResistorElement,
    ThermalConductorElement,
)

__all__: list[str] = [
    'THERMAL',
    'BangBangPlantElement',
    'CartPendulumElement',
    'ControlInputElement',
    'ConvectionElement',
    'DrivenOscillatorElement',
    'HeatCapacityElement',
    'HeatingResistorElement',
    'IntegratorElement',
    'PIDInteraction',
    'PlantStateElement',
    'StateFeedbackInteraction',
    'ThermalConductorElement',
    'heated_room',
    'koopman_model_predictive_control',
    'lqr_oscillator',
    'minimum_time_control',
    'pendulum_swing_up',
    'pid_inverted_pendulum',
    'pid_loop',
    'satellite_transfer',
    'thermostat',
]
