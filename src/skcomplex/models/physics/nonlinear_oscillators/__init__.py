"""Nonlinear oscillators: Duffing, Van der Pol, Stuart-Landau, Adler, billiards, and Hamiltonian chaos."""

from ._billiards import MushroomBilliardComponent, SinaiBilliardComponent, billiards
from ._oscillators import (
    AdlerPhaseComponent,
    DuffingComponent,
    ForcedVanDerPolComponent,
    HenonHeilesComponent,
    StuartLandauComponent,
    VanDerPolComponent,
    adler_equation,
    duffing,
    forced_van_der_pol,
    henon_heiles,
    stuart_landau,
    van_der_pol,
)

__all__: list[str] = [
    'AdlerPhaseComponent',
    'DuffingComponent',
    'ForcedVanDerPolComponent',
    'HenonHeilesComponent',
    'MushroomBilliardComponent',
    'SinaiBilliardComponent',
    'StuartLandauComponent',
    'VanDerPolComponent',
    'adler_equation',
    'billiards',
    'duffing',
    'forced_van_der_pol',
    'henon_heiles',
    'stuart_landau',
    'van_der_pol',
]
