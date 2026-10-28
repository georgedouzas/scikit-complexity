"""Segregation: residential sorting of agents and its dissimilarity index."""

from ._blocks import BlockMovesInteraction, DwellerElement, stochastic_schelling
from ._dissimilarity_index import DissimilarityIndexAnalysis
from ._schelling import ResidentElement, schelling

__all__: list[str] = [
    'BlockMovesInteraction',
    'DissimilarityIndexAnalysis',
    'DwellerElement',
    'ResidentElement',
    'schelling',
    'stochastic_schelling',
]
