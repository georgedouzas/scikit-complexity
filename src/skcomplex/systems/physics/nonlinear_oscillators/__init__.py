"""Nonlinear oscillator systems.

The package covers Duffing, Toda, Van der Pol, Stuart-Landau, Adler, impulses, billiards, and chaos.
"""

from ._billiards import (
    BilliardsSystem,
    BilliardWallInteraction,
    FreeFlightInteraction,
    MushroomBilliardElement,
    SinaiBilliardElement,
    ThreeDiskBilliardElement,
)
from ._oscillators import (
    AdlerEquationSystem,
    AdlerInteraction,
    DuffingSystem,
    ForcedTodaSystem,
    ForcedVanDerPolInteraction,
    ForcedVanDerPolSystem,
    HenonHeilesElement,
    HenonHeilesSystem,
    ImpulsiveOscillatorElement,
    ImpulsiveOscillatorSystem,
    ImpulsiveSpringInteraction,
    KickInteraction,
    NuclearQuadrupoleSystem,
    QuadrupoleElement,
    QuadrupoleMotionInteraction,
    StuartLandauInteraction,
    StuartLandauSystem,
    TodaInteraction,
    VanDerPolElement,
    VanDerPolSystem,
)

__all__: list[str] = [
    'AdlerEquationSystem',
    'AdlerInteraction',
    'BilliardWallInteraction',
    'BilliardsSystem',
    'DuffingSystem',
    'ForcedTodaSystem',
    'ForcedVanDerPolInteraction',
    'ForcedVanDerPolSystem',
    'FreeFlightInteraction',
    'HenonHeilesElement',
    'HenonHeilesSystem',
    'ImpulsiveOscillatorElement',
    'ImpulsiveOscillatorSystem',
    'ImpulsiveSpringInteraction',
    'KickInteraction',
    'MushroomBilliardElement',
    'NuclearQuadrupoleSystem',
    'QuadrupoleElement',
    'QuadrupoleMotionInteraction',
    'SinaiBilliardElement',
    'StuartLandauInteraction',
    'StuartLandauSystem',
    'ThreeDiskBilliardElement',
    'TodaInteraction',
    'VanDerPolElement',
    'VanDerPolSystem',
]
