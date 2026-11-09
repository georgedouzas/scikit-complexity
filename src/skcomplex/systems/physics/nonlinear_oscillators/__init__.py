"""Nonlinear oscillator systems.

The package covers Duffing, Toda, Van der Pol, Stuart-Landau, Adler, impulses, billiards, and chaos.
"""

from ._billiards import BilliardsSystem, MushroomBilliardElement, SinaiBilliardElement, ThreeDiskBilliardElement
from ._oscillators import (
    AdlerEquationSystem,
    DuffingSystem,
    ForcedTodaSystem,
    ForcedVanDerPolSystem,
    HenonHeilesElement,
    HenonHeilesSystem,
    ImpulsiveOscillatorElement,
    ImpulsiveOscillatorSystem,
    KickInteraction,
    NuclearQuadrupoleSystem,
    QuadrupoleElement,
    StuartLandauSystem,
    TodaElement,
    VanDerPolElement,
    VanDerPolSystem,
)

__all__: list[str] = [
    'AdlerEquationSystem',
    'BilliardsSystem',
    'DuffingSystem',
    'ForcedTodaSystem',
    'ForcedVanDerPolSystem',
    'HenonHeilesElement',
    'HenonHeilesSystem',
    'ImpulsiveOscillatorElement',
    'ImpulsiveOscillatorSystem',
    'KickInteraction',
    'MushroomBilliardElement',
    'NuclearQuadrupoleSystem',
    'QuadrupoleElement',
    'SinaiBilliardElement',
    'StuartLandauSystem',
    'ThreeDiskBilliardElement',
    'TodaElement',
    'VanDerPolElement',
    'VanDerPolSystem',
]
