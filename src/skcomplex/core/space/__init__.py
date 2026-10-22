"""Spaces and times that systems live in: continuous, lattice, group, network, and mesh spaces, and time kinds."""

from ._cayley import GroupSpace, Memory
from ._groups import (
    FinitelyGeneratedGroup,
    automaton_group,
    finitely_presented_group,
    matrix_group,
    parse_word,
    permutation_group,
)
from ._lattices import LatticeSpace, Neighborhood
from ._meshes import Mesh
from ._neighbours import neighbour_pairs
from ._phases import Phase
from ._spaces import Adjacency, ContinuousSpace, NetworkSpace, Space
from ._time import ContinuousTime, DiscreteTime, Phases, StaticTime, Time

__all__: list[str] = [
    'Adjacency',
    'ContinuousSpace',
    'ContinuousTime',
    'DiscreteTime',
    'FinitelyGeneratedGroup',
    'GroupSpace',
    'LatticeSpace',
    'Memory',
    'Mesh',
    'Neighborhood',
    'NetworkSpace',
    'Phase',
    'Phases',
    'Space',
    'StaticTime',
    'Time',
    'automaton_group',
    'finitely_presented_group',
    'matrix_group',
    'neighbour_pairs',
    'parse_word',
    'permutation_group',
]
