"""Network realizations shared by the biology packages.

They cover weighted edges, infected hosts, and Boolean nodes.
"""

from ._boolean_function import BooleanFunctionInteraction
from ._boolean_node import BooleanNodeElement
from ._edge_draw import EdgeDrawInteraction
from ._host import HostElement
from ._spreading import SpreadingInteraction
from ._weighted_edge import WeightedEdgeElement

__all__: list[str] = [
    'BooleanFunctionInteraction',
    'BooleanNodeElement',
    'EdgeDrawInteraction',
    'HostElement',
    'SpreadingInteraction',
    'WeightedEdgeElement',
]
