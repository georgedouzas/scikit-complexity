"""Define the dynamical systems that the systems of several fields share.

States evolve by their equations, in continuous or discrete time, on the nodes of a graph, as fields on a lattice, or as
stocks that change through named flows.
"""

from ._dynamical import DynamicalSystem
from ._network_dynamical import NetworkDynamicalSystem
from ._reaction_diffusion import ReactionDiffusionSystem
from ._stock_flow import StockFlowSystem

__all__: list[str] = [
    'DynamicalSystem',
    'NetworkDynamicalSystem',
    'ReactionDiffusionSystem',
    'StockFlowSystem',
]
