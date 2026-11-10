"""Define spaces, neighbourhoods, and times, where and when a system runs.

A space places the members of a group and says who neighbours whom, such as a lattice, a network, or a continuous box. A
neighbourhood picks the members a law reads around a member. A time says whether the system flows, steps, or is solved
once.
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
