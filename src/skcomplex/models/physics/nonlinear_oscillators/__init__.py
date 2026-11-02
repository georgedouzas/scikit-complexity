"""Nonlinear oscillators: Duffing, Van der Pol, Stuart-Landau, Adler, billiards, and Hamiltonian chaos."""

from ._adler_phase import AdlerPhaseElement
from ._billiards import billiards
from ._duffing import DuffingElement
from ._henon_heiles import HenonHeilesElement
from ._mushroom_billiard import MushroomBilliardElement
from ._oscillators import adler_equation, duffing, forced_van_der_pol, henon_heiles, stuart_landau, van_der_pol
from ._sinai_billiard import SinaiBilliardElement
from ._van_der_pol import VanDerPolElement

__all__: list[str] = [
    'AdlerPhaseElement',
    'DuffingElement',
    'HenonHeilesElement',
    'MushroomBilliardElement',
    'SinaiBilliardElement',
    'VanDerPolElement',
    'adler_equation',
    'billiards',
    'duffing',
    'forced_van_der_pol',
    'henon_heiles',
    'stuart_landau',
    'van_der_pol',
]
