"""Mechatronics: mechanisms joined to circuits and suspensions of masses, springs, and dampers."""

from ._magnetic_levitation import LevitatedBallElement, MagneticLevitationSystem
from ._mechatronics import (
    ArmatureElement,
    DcMotorSystem,
    InertiaElement,
    MotorCouplingInteraction,
    QuarterCarElement,
    QuarterCarSystem,
)
from ._vibrating_table import VibratingTableElement, VibratingTableSystem

__all__: list[str] = [
    'ArmatureElement',
    'DcMotorSystem',
    'InertiaElement',
    'LevitatedBallElement',
    'MagneticLevitationSystem',
    'MotorCouplingInteraction',
    'QuarterCarElement',
    'QuarterCarSystem',
    'VibratingTableElement',
    'VibratingTableSystem',
]
