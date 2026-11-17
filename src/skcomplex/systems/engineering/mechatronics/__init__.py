"""Mechatronics: mechanisms joined to circuits, and suspensions.

The suspensions are made of masses, springs, and dampers.
"""

from ._magnetic_levitation import DampingInteraction, LevitatedBallElement, MagneticLevitationSystem
from ._mechatronics import (
    ArmatureElement,
    DcMotorSystem,
    InertiaElement,
    MotorCouplingInteraction,
    QuarterCarSystem,
    TireInteraction,
)
from ._vibrating_table import VibratingTableSystem

__all__: list[str] = [
    'ArmatureElement',
    'DampingInteraction',
    'DcMotorSystem',
    'InertiaElement',
    'LevitatedBallElement',
    'MagneticLevitationSystem',
    'MotorCouplingInteraction',
    'QuarterCarSystem',
    'TireInteraction',
    'VibratingTableSystem',
]
