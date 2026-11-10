"""Define the flows that the systems of several fields share.

The Lorenz flow is a model of convection in mathematics and the response system of chaotic synchronization in physics.
"""

from ._lorenz import LorenzElement

__all__: list[str] = [
    'LorenzElement',
]
