"""Excitable kinetics shared by the biology packages: the FitzHugh-Nagumo fast-slow kinetics."""

from ._fitzhugh_nagumo import FitzHughNagumoInteraction

__all__: list[str] = [
    'FitzHughNagumoInteraction',
]
