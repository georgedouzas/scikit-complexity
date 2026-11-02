"""Segregation: residential sorting of agents and its dissimilarity index."""

from ._block_moves import BlockMovesInteraction
from ._blocks import stochastic_schelling
from ._dweller import DwellerElement
from ._dweller_view import DwellerView
from ._relocation import RelocationInteraction
from ._resident import ResidentElement
from ._resident_view import ResidentView
from ._schelling import schelling

__all__: list[str] = [
    'BlockMovesInteraction',
    'DwellerElement',
    'DwellerView',
    'RelocationInteraction',
    'ResidentElement',
    'ResidentView',
    'schelling',
    'stochastic_schelling',
]
