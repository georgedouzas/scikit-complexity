"""Flows that systems of several fields share.

The Lorenz flow is the convection model of mathematics and the response of chaotic synchronization in physics.
"""

from ._lorenz import LorenzElement

__all__: list[str] = [
    'LorenzElement',
]
