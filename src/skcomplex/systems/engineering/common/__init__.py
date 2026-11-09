"""Realizations shared by several engineering systems, by family.

The families are mechanics and electrical components, signals, and traffic. The car and its optimal velocity of the
traffic systems sit at the root.
"""

from ._car import CarElement
from ._optimal_velocity import OptimalVelocityInteraction

__all__: list[str] = [
    'CarElement',
    'OptimalVelocityInteraction',
]
