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
from ._jerk import Jerk, sprott_jerk
from ._maps import HenonMap, LogisticMap, henon_map, logistic_map

__all__: list[str] = [
    'Duffing',
    'HenonHeiles',
    'HenonMap',
    'Jerk',
    'LogisticMap',
    'Lorenz',
    'Rossler',
    'VanDerPol',
    'duffing',
    'henon_heiles',
    'henon_map',
    'logistic_map',
    'lorenz',
    'rossler',
    'sprott_jerk',
    'van_der_pol',
]
