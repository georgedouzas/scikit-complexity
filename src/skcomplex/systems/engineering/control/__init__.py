"""Control: feedback loops that keep or drive a system towards a goal.

It also holds the controllers that close them.
"""

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
    CartPendulumInteraction,
    DrivenOscillatorElement,
    LqrOscillatorSystem,
    PIDInteraction,
    PidInvertedPendulumSystem,
    StateFeedbackInteraction,
)

__all__: list[str] = [
    'THERMAL',
    'BangBangPlantElement',
    'CartPendulumInteraction',
    'ControlInputElement',
    'ConvectionElement',
    'DrivenOscillatorElement',
    'HeatCapacityElement',
    'HeatedRoomSystem',
    'HeatingResistorElement',
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
