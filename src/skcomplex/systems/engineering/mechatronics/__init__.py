"""Mechatronics: mechanisms joined to circuits and suspensions of masses, springs, and dampers."""

from ._magnetic_levitation import LevitatedBallElement, magnetic_levitation
from ._mechatronics import (
    ArmatureElement,
    InertiaElement,
    MotorCouplingInteraction,
    QuarterCarElement,
    dc_motor,
    quarter_car,
)
from ._vibrating_table import VibratingTableElement, vibrating_table

__all__: list[str] = [
    'ArmatureElement',
    'InertiaElement',
    'LevitatedBallElement',
    'MotorCouplingInteraction',
    'QuarterCarElement',
    'VibratingTableElement',
    'dc_motor',
    'magnetic_levitation',
    'quarter_car',
    'vibrating_table',
]
