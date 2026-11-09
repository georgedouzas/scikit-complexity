"""Define the spaces that systems live in, with neighbourhoods and time.

The neighbourhoods are those of their members, and the time is that of a system.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._adjacency import AdjacencyNeighborhood
from ._ball import BallNeighborhood
from ._base import BaseNeighborhood, BaseSpace, BaseTime
from ._cayley import CayleySpace
from ._cell_complex import CellComplexSpace
from ._coface import CofaceNeighborhood
from ._continuous import ContinuousSpace
from ._continuous_time import ContinuousTime
from ._delaunay import DelaunayNeighborhood
from ._discrete_time import DiscreteTime
from ._lattice import LatticeSpace
from ._lattice_neighborhood import LatticeNeighborhood
from ._memory import MemoryNeighborhood
from ._mesh import MeshSpace
from ._nearest import NearestNeighborhood
from ._network import NetworkSpace
from ._phase import PhaseSpace
from ._radius import RadiusNeighborhood
from ._simplex import SimplexSpace
from ._state_translation import StateTranslationNeighborhood
from ._static_time import StaticTime

__all__: list[str] = [
    'AdjacencyNeighborhood',
    'BallNeighborhood',
    'BaseNeighborhood',
    'BaseSpace',
    'BaseTime',
    'CayleySpace',
    'CellComplexSpace',
    'CofaceNeighborhood',
    'ContinuousSpace',
    'ContinuousTime',
    'DelaunayNeighborhood',
    'DiscreteTime',
    'LatticeNeighborhood',
    'LatticeSpace',
    'MemoryNeighborhood',
    'MeshSpace',
    'NearestNeighborhood',
    'NetworkSpace',
    'PhaseSpace',
    'RadiusNeighborhood',
    'SimplexSpace',
    'StateTranslationNeighborhood',
    'StaticTime',
]
