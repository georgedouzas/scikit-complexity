"""Define spaces, neighbourhoods, and times, where and when a system runs.

A space places the members of a group and says who neighbours whom, such as a lattice, a network, or a continuous box. A
neighbourhood picks the members a law reads around a member. A time says whether the system flows, steps, or is solved
once.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._adjacency import AdjacencyNeighborhood
from ._axis_spec import AxisSpec
from ._base import BaseNeighborhood, BaseSpace, BaseTime
from ._cell_complex import CellComplexSpace
from ._coface import CofaceNeighborhood
from ._continuous import ContinuousSpace
from ._continuous_time import ContinuousTime
from ._delaunay import DelaunayNeighborhood
from ._discrete_time import DiscreteTime
from ._lattice import LatticeSpace
from ._lattice_neighborhood import LatticeNeighborhood
from ._layer_spec import LayerSpec
from ._nearest import NearestNeighborhood
from ._network import NetworkSpace
from ._phase import PhaseSpace
from ._radius import RadiusNeighborhood
from ._schedule_spec import ScheduleSpec
from ._static_time import StaticTime

__all__: list[str] = [
    'AdjacencyNeighborhood',
    'AxisSpec',
    'BaseNeighborhood',
    'BaseSpace',
    'BaseTime',
    'CellComplexSpace',
    'CofaceNeighborhood',
    'ContinuousSpace',
    'ContinuousTime',
    'DelaunayNeighborhood',
    'DiscreteTime',
    'LatticeNeighborhood',
    'LatticeSpace',
    'LayerSpec',
    'NearestNeighborhood',
    'NetworkSpace',
    'PhaseSpace',
    'RadiusNeighborhood',
    'ScheduleSpec',
    'StaticTime',
]
