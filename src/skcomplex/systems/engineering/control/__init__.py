"""Control: feedback loops that keep or drive a system towards a goal, and the controllers that close them."""

from ._control import (
    THERMAL,
    ConvectionElement,
    HeatCapacityElement,
    HeatedRoomSystem,
    HeatingResistorElement,
    PendulumSwingUpSystem,
    PidLoopSystem,
    SatelliteTransferSystem,
    ThermalConductorElement,
    ThermostatSystem,
)
from ._koopman import ControlInputElement, KoopmanModelPredictiveControlSystem, PlantStateElement
from ._minimum_time_control import BangBangPlantElement, MinimumTimeControlSystem
from ._regulation import (
    CartPendulumElement,
    DrivenOscillatorElement,
    IntegratorElement,
    LqrOscillatorSystem,
    PIDInteraction,
    PidInvertedPendulumSystem,
    StateFeedbackInteraction,
)

__all__: list[str] = [
    'THERMAL',
    'BangBangPlantElement',
    'CartPendulumElement',
    'ControlInputElement',
    'ConvectionElement',
    'DrivenOscillatorElement',
    'HeatCapacityElement',
    'HeatedRoomSystem',
    'HeatingResistorElement',
    'IntegratorElement',
    'KoopmanModelPredictiveControlSystem',
    'LqrOscillatorSystem',
    'MinimumTimeControlSystem',
    'PIDInteraction',
    'PendulumSwingUpSystem',
    'PidInvertedPendulumSystem',
    'PidLoopSystem',
    'PlantStateElement',
    'SatelliteTransferSystem',
    'StateFeedbackInteraction',
    'ThermalConductorElement',
    'ThermostatSystem',
]
