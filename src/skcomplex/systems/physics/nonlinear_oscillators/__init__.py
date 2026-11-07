"""Nonlinear oscillators: Duffing, Toda, Van der Pol, Stuart-Landau, Adler, impulses, billiards, and chaos."""

from ._billiards import BilliardsSystem, MushroomBilliardElement, SinaiBilliardElement, ThreeDiskBilliardElement
from ._oscillators import (
    AdlerEquationSystem,
    AdlerPhaseElement,
    DuffingElement,
    DuffingSystem,
    ForcedTodaSystem,
    ForcedVanDerPolSystem,
    HenonHeilesElement,
    HenonHeilesSystem,
    ImpulsiveOscillatorElement,
    ImpulsiveOscillatorSystem,
    NuclearQuadrupoleSystem,
    QuadrupoleElement,
    StuartLandauSystem,
    TodaElement,
    VanDerPolElement,
    VanDerPolSystem,
)

__all__: list[str] = [
    'AdlerEquationSystem',
    'AdlerPhaseElement',
    'BilliardsSystem',
    'DuffingElement',
    'DuffingSystem',
    'ForcedTodaSystem',
    'ForcedVanDerPolSystem',
    'HenonHeilesElement',
    'HenonHeilesSystem',
    'ImpulsiveOscillatorElement',
    'ImpulsiveOscillatorSystem',
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
