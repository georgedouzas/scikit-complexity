"""Dynamical systems that systems of several fields share.

States evolve by their equations, in continuous or discrete time, on the nodes of a graph, as fields on a lattice, or as
stocks that change through named flows.
"""

from ._dynamical import DynamicalSystem

__all__: list[str] = [
    'DynamicalSystem',
]
