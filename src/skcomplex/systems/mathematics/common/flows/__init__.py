"""Flows shared by the mathematics packages.

The package holds the Rössler flow.
"""

from ._rossler import RosslerElement

__all__: list[str] = [
    'RosslerElement',
]
