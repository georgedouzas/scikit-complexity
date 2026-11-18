"""Control: feedback loops that keep or drive a system towards a goal.

It also holds the controllers that close them.
"""

from ._control import (
    ConvectionInteraction,
    HeatCapacityInteraction,
    HeatedRoomSystem,
    HeatingResistorInteraction,
    PendulumSwingUpSystem,
    PidLoopSystem,
    SatelliteTransferSystem,
    SwitchInteraction,
    ThermalConductorInteraction,
    ThermostatSystem,
)
from ._koopman import (
    ControlInputElement,
    KoopmanModelPredictiveControlSystem,
    PlantInteraction,
    PlantStateElement,
    PredictiveControlInteraction,
)
from ._minimum_time_control import BangBangPlantElement, BangBangPlantInteraction, MinimumTimeControlSystem
from ._regulation import (
    CartPendulumInteraction,
    DrivenOscillatorInteraction,
    LqrOscillatorSystem,
    PIDInteraction,
    PidInvertedPendulumSystem,
    StateFeedbackInteraction,
)

__all__: list[str] = [
    'BangBangPlantElement',
    'BangBangPlantInteraction',
    'CartPendulumInteraction',
    'ControlInputElement',
    'ConvectionInteraction',
    'DrivenOscillatorInteraction',
    'HeatCapacityInteraction',
    'HeatedRoomSystem',
    'HeatingResistorInteraction',
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
    'SwitchInteraction',
    'ThermalConductorInteraction',
    'ThermostatSystem',
]
