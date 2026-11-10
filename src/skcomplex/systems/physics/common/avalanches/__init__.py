"""Avalanches shared by the physics packages.

They are the cells of a sandpile and the toppling that relaxes them.
"""

from ._sandpile_cell import SandpileCellElement
from ._toppling import TopplingInteraction

__all__: list[str] = [
    'SandpileCellElement',
    'TopplingInteraction',
]
