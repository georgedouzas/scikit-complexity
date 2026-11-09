"""Define the spaces that systems live in, with neighbourhoods and time.

The neighbourhoods are those of their members, and the time is that of a system.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._adjacency import AdjacencyNeighborhood
from ._base import BaseNeighborhood, BaseSpace, BaseTime
from ._box import BoxNeighborhood
from ._cayley import CayleySpace
from ._cell_complex import CellComplexSpace
from ._coface import CofaceNeighborhood
from ._cone import ConeNeighborhood
from ._continuous import ContinuousSpace
from ._continuous_time import ContinuousTime
from ._delaunay import DelaunayNeighborhood
from ._discrete_time import DiscreteTime
from ._generator import GeneratorNeighborhood
from ._lattice import LatticeSpace
from ._lattice_neighborhood import LatticeNeighborhood
from ._memory import MemoryNeighborhood
from ._mesh import MeshSpace
from ._moore import MooreNeighborhood
from ._nearest import NearestNeighborhood
from ._network import NetworkSpace
from ._phase import PhaseSpace
from ._pseudo_riemannian import PseudoRiemannianSpace
from ._radius import RadiusNeighborhood
from ._simplex import SimplexSpace
from ._state_translation import StateTranslationNeighborhood
from ._static_time import StaticTime
from ._triangular import TriangularNeighborhood
from ._von_neumann import VonNeumannNeighborhood

__all__: list[str] = [
    'AdjacencyNeighborhood',
    'BaseNeighborhood',
    'BaseSpace',
    'BaseTime',
    'BoxNeighborhood',
    'CayleySpace',
    'CellComplexSpace',
    'CofaceNeighborhood',
    'ConeNeighborhood',
    'ContinuousSpace',
    'ContinuousTime',
    'DelaunayNeighborhood',
    'DiscreteTime',
    'GeneratorNeighborhood',
    'LatticeNeighborhood',
    'LatticeSpace',
    'MemoryNeighborhood',
    'MeshSpace',
    'MooreNeighborhood',
    'NearestNeighborhood',
    'NetworkSpace',
    'PhaseSpace',
    'PseudoRiemannianSpace',
    'RadiusNeighborhood',
    'SimplexSpace',
    'StateTranslationNeighborhood',
    'StaticTime',
    'TriangularNeighborhood',
    'VonNeumannNeighborhood',
]
