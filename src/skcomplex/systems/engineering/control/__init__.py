"""Control: feedback loops that keep or drive a system towards a goal, and the controllers that close them."""

from ._control import (
    THERMAL,
    ConvectionElement,
    HeatCapacityElement,
    HeatingResistorElement,
    ThermalConductorElement,
    heated_room,
    pendulum_swing_up,
    pid_loop,
    satellite_transfer,
    thermostat,
)
from ._koopman import ControlInputElement, PlantStateElement, koopman_model_predictive_control
from ._minimum_time_control import BangBangPlantElement, minimum_time_control
from ._regulation import (
    CartPendulumElement,
    DrivenOscillatorElement,
    IntegratorElement,
    PIDInteraction,
    StateFeedbackInteraction,
    lqr_oscillator,
    pid_inverted_pendulum,
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
