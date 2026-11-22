"""Automata on groups shared by the computer science packages.

The package holds the Cayley graph of a finitely generated group, the memory sets that cells on it read, the
neighbourhood that moves with the value of a cell, and the visual that draws the graph.
"""

from ._cayley import CayleySpace, CayleyVisual, MemoryNeighborhood, StateTranslationNeighborhood

__all__: list[str] = [
    'CayleySpace',
    'CayleyVisual',
    'MemoryNeighborhood',
    'StateTranslationNeighborhood',
]
