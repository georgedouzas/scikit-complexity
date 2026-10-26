"""Nonlinear oscillators: Duffing, Van der Pol, Stuart-Landau, Adler, billiards, and Hamiltonian chaos."""

from ._billiards import MushroomBilliardElement, SinaiBilliardElement, billiards
from ._oscillators import (
    AdlerPhaseElement,
    DuffingElement,
    ForcedVanDerPolElement,
    HenonHeilesElement,
    StuartLandauElement,
    VanDerPolElement,
    adler_equation,
    duffing,
    forced_van_der_pol,
    henon_heiles,
    stuart_landau,
    van_der_pol,
)

__all__: list[str] = [
    'AdlerPhaseElement',
    'DuffingElement',
    'ForcedVanDerPolElement',
    'HenonHeilesElement',
    'MushroomBilliardElement',
    'SinaiBilliardElement',
    'StuartLandauElement',
    'VanDerPolElement',
    'adler_equation',
    'billiards',
    'duffing',
    'forced_van_der_pol',
    'henon_heiles',
    'stuart_landau',
    'van_der_pol',
]
