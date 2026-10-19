"""Nonlinear oscillators: Duffing, Van der Pol, Stuart-Landau, Adler, billiards, and Hamiltonian chaos."""

from ._billiards import MushroomBilliard, SinaiBilliard, billiards
from ._oscillators import (
    AdlerPhase,
    Duffing,
    ForcedVanDerPol,
    HenonHeiles,
    StuartLandau,
    VanDerPol,
    adler_equation,
    duffing,
    forced_van_der_pol,
    henon_heiles,
    stuart_landau,
    van_der_pol,
)

__all__: list[str] = [
    'AdlerPhase',
    'Duffing',
    'ForcedVanDerPol',
    'HenonHeiles',
    'MushroomBilliard',
    'SinaiBilliard',
    'StuartLandau',
    'VanDerPol',
    'adler_equation',
    'billiards',
    'duffing',
    'forced_van_der_pol',
    'henon_heiles',
    'stuart_landau',
    'van_der_pol',
]
