"""Nonlinear dynamics: canonical chaotic flows and maps."""

from ._flows import (
    Duffing,
    HenonHeiles,
    Lorenz,
    Rossler,
    VanDerPol,
    duffing,
    henon_heiles,
    lorenz,
    rossler,
    van_der_pol,
)
from ._hybrid import (
    DelayedCoupling,
    DoubleWell,
    MackeyGlass,
    PhaseOscillator,
    delayed_kuramoto,
    mackey_glass,
    stochastic_resonance,
)
from ._jerk import Jerk, sprott_jerk
from ._maps import HenonMap, LogisticMap, henon_map, logistic_map

__all__: list[str] = [
    'DelayedCoupling',
    'DoubleWell',
    'Duffing',
    'HenonHeiles',
    'HenonMap',
    'Jerk',
    'LogisticMap',
    'Lorenz',
    'MackeyGlass',
    'PhaseOscillator',
    'Rossler',
    'VanDerPol',
    'delayed_kuramoto',
    'duffing',
    'henon_heiles',
    'henon_map',
    'logistic_map',
    'lorenz',
    'mackey_glass',
    'rossler',
    'sprott_jerk',
    'stochastic_resonance',
    'van_der_pol',
]
