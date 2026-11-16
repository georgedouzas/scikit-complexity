"""Define the gradient flows that the systems of several fields share.

A state moves down the gradient of a potential, as a particle in a force field, a field that orders, or a crowd that
avoids crowding.
"""

from ._gradient import GradientInteraction

__all__: list[str] = [
    'GradientInteraction',
]
