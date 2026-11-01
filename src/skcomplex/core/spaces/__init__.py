"""Define the spaces that systems live in."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseSpace
from ._cayley import CayleySpace
from ._continuous import ContinuousSpace
from ._lattice import LatticeSpace
from ._mesh import MeshSpace
from ._network import NetworkSpace

__all__: list[str] = [
    'BaseSpace',
    'CayleySpace',
    'ContinuousSpace',
    'LatticeSpace',
    'MeshSpace',
    'NetworkSpace',
]
