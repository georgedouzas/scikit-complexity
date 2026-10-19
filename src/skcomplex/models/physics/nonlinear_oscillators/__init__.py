"""Nonlinear oscillators: Duffing, Van der Pol, and Hamiltonian chaos."""

from ._oscillators import Duffing, HenonHeiles, VanDerPol, duffing, henon_heiles, van_der_pol

__all__: list[str] = [
    'Duffing',
    'HenonHeiles',
    'VanDerPol',
    'duffing',
    'henon_heiles',
    'van_der_pol',
]
