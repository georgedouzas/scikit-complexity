"""Flows shared by the mathematics packages.

The package holds the Rössler flow.
"""

from ._rossler import (
    RosslerInteraction,
)

__all__: list[str] = [
    'RosslerInteraction',
]
