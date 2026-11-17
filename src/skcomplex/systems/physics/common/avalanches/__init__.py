"""Avalanches shared by the physics packages.

They are the toppling that relaxes the sites of a sandpile.
"""

from ._toppling import TopplingInteraction

__all__: list[str] = [
    'TopplingInteraction',
]
