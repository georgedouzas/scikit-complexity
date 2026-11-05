"""Network realizations shared by the biology packages: weighted edges of ecological networks and their draws."""

from ._edge_draw import EdgeDrawInteraction
from ._weighted_edge import WeightedEdgeElement

__all__: list[str] = [
    'EdgeDrawInteraction',
    'WeightedEdgeElement',
]
