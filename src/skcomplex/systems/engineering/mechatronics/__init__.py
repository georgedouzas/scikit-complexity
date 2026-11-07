"""Mechatronics: mechanisms joined to circuits and suspensions of masses, springs, and dampers."""

from ._levitated_ball import LevitatedBallElement
from ._mechatronics import dc_motor, magnetic_levitation, quarter_car, vibrating_table
from ._motor import ArmatureElement, InertiaElement, MotorCouplingInteraction
from ._quarter_car import QuarterCarElement
from ._vibrating_table import VibratingTableElement

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
