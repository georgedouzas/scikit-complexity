"""Nonlinear oscillators: Duffing, Toda, Van der Pol, Stuart-Landau, Adler, impulses, billiards, and chaos."""

from ._adler_phase import AdlerPhaseElement
from ._billiards import billiards
from ._duffing import DuffingElement
from ._henon_heiles import HenonHeilesElement
from ._impulsive_oscillator import ImpulsiveOscillatorElement
from ._mushroom_billiard import MushroomBilliardElement
from ._oscillators import (
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
from ._quadrupole import QuadrupoleElement
from ._sinai_billiard import SinaiBilliardElement
from ._three_disk_billiard import ThreeDiskBilliardElement
from ._toda import TodaElement
from ._van_der_pol import VanDerPolElement

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
