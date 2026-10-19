"""Physiology: delayed feedback and excitable tissue."""

from ._physiology import FitzHughNagumo, MackeyGlass, fitzhugh_nagumo, mackey_glass

__all__: list[str] = [
    'FitzHughNagumo',
    'MackeyGlass',
    'fitzhugh_nagumo',
    'mackey_glass',
]
