"""Define the spaces that systems live in, the neighbourhoods of their members, and the time of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._adjacency import AdjacencyNeighborhood
from ._base import BaseNeighborhood, BaseSpace, BaseTime
from ._box import BoxNeighborhood
from ._cayley import CayleySpace
from ._cone import ConeNeighborhood
from ._continuous import ContinuousSpace
from ._continuous_time import ContinuousTime
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
from ._radius import RadiusNeighborhood
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
    'ConeNeighborhood',
    'ContinuousSpace',
    'ContinuousTime',
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
    'RadiusNeighborhood',
    'StaticTime',
    'TriangularNeighborhood',
    'VonNeumannNeighborhood',
]
