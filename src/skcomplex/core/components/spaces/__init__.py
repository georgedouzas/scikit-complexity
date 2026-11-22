"""Define the spaces members live on and the neighbourhoods laws read.

A space places the members of a group and says who neighbours whom, such as a lattice, a network, or a continuous box. A
neighbourhood picks the members a law reads around a member.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._adjacency import AdjacencyNeighborhood
from ._axis_spec import AxisSpec
from ._base import BaseNeighborhood, BaseSpace
from ._cell_complex import CellComplexSpace
from ._coface import CofaceNeighborhood
from ._continuous import ContinuousSpace
from ._delaunay import DelaunayNeighborhood
from ._finite_volume_spec import FiniteVolumeSpec
from ._lattice import LatticeSpace
from ._lattice_neighborhood import LatticeNeighborhood
from ._layer_spec import LayerSpec
from ._nearest import NearestNeighborhood
from ._network import NetworkSpace
from ._phase import PhaseSpace
from ._radius import RadiusNeighborhood
from ._schedule_spec import ScheduleSpec

__all__: list[str] = [
    'AdjacencyNeighborhood',
    'AxisSpec',
    'BaseNeighborhood',
    'BaseSpace',
    'CellComplexSpace',
    'CofaceNeighborhood',
    'ContinuousSpace',
    'DelaunayNeighborhood',
    'FiniteVolumeSpec',
    'LatticeNeighborhood',
    'LatticeSpace',
    'LayerSpec',
    'NearestNeighborhood',
    'NetworkSpace',
    'PhaseSpace',
    'RadiusNeighborhood',
    'ScheduleSpec',
]
