"""Mechatronics: mechanisms joined to circuits and suspensions of masses, springs, and dampers."""

from ._mechatronics import dc_motor, quarter_car
from ._motor import ArmatureElement, InertiaElement, MotorCouplingInteraction
from ._quarter_car import QuarterCarElement

__all__: list[str] = [
    'ArmatureElement',
    'InertiaElement',
    'MotorCouplingInteraction',
    'QuarterCarElement',
    'dc_motor',
    'quarter_car',
]
