"""Define the neighbourhoods that give the members a law reads around a member of a space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._adjacency import AdjacencyNeighborhood
from ._base import BaseNeighborhood
from ._box import BoxNeighborhood
from ._cone import ConeNeighborhood
from ._generator import GeneratorNeighborhood
from ._lattice_neighborhood import LatticeNeighborhood
from ._memory import MemoryNeighborhood
from ._moore import MooreNeighborhood
from ._nearest import NearestNeighborhood
from ._radius import RadiusNeighborhood
from ._triangular import TriangularNeighborhood
from ._von_neumann import VonNeumannNeighborhood

__all__: list[str] = [
    'AdjacencyNeighborhood',
    'BaseNeighborhood',
    'BoxNeighborhood',
    'ConeNeighborhood',
    'GeneratorNeighborhood',
    'LatticeNeighborhood',
    'MemoryNeighborhood',
    'MooreNeighborhood',
    'NearestNeighborhood',
    'RadiusNeighborhood',
    'TriangularNeighborhood',
    'VonNeumannNeighborhood',
]
