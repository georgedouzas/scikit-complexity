"""Nonlinear oscillators: Duffing, Toda, Van der Pol, Stuart-Landau, Adler, impulses, billiards, and chaos."""

from ._billiards import MushroomBilliardElement, SinaiBilliardElement, ThreeDiskBilliardElement, billiards
from ._oscillators import (
    AdlerPhaseElement,
    DuffingElement,
    HenonHeilesElement,
    ImpulsiveOscillatorElement,
    QuadrupoleElement,
    TodaElement,
    VanDerPolElement,
    adler_equation,
    duffing,
    forced_toda,
    forced_van_der_pol,
    henon_heiles,
    impulsive_oscillator,
    nuclear_quadrupole,
    stuart_landau,
    van_der_pol,
)

__all__: list[str] = [
    'AdlerPhaseElement',
    'DuffingElement',
    'HenonHeilesElement',
    'ImpulsiveOscillatorElement',
    'MushroomBilliardElement',
    'QuadrupoleElement',
    'SinaiBilliardElement',
    'ThreeDiskBilliardElement',
    'TodaElement',
    'VanDerPolElement',
    'adler_equation',
    'billiards',
    'duffing',
    'forced_toda',
    'forced_van_der_pol',
    'henon_heiles',
    'impulsive_oscillator',
    'nuclear_quadrupole',
    'stuart_landau',
    'van_der_pol',
]
