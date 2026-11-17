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
from ._koopman import (
    ControlInputElement,
    KoopmanModelPredictiveControlSystem,
    PlantInteraction,
    PlantStateElement,
    PredictiveControlInteraction,
)
from ._minimum_time_control import BangBangPlantElement, MinimumTimeControlSystem
from ._regulation import (
    CartPendulumInteraction,
    DrivenOscillatorInteraction,
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
    'DrivenOscillatorInteraction',
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
    'PlantInteraction',
    'PlantStateElement',
    'PredictiveControlInteraction',
    'SatelliteTransferSystem',
    'StateFeedbackInteraction',
    'ThermalConductorElement',
    'ThermostatSystem',
]
