"""State of a lattice group in the state vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from dataclasses import dataclass

import numpy as np

from ...core.spaces import BaseSpace
from ._base import BaseValue


@dataclass(frozen=True)
class LatticeState(BaseValue):
    """State of a lattice group in the state vector.

    Args:
        name:
            The dotted name of the state.

        start:
            The index of its first component.

        space:
            The lattice of the group, or the network whose nodes are its sites.
    """

    name: str
    start: int
    space: BaseSpace

    @functools.cached_property
    def shape(self) -> tuple[int, ...]:
        """Shape of the sites of the group."""
        return self.space.shape_

    @functools.cached_property
    def size(self) -> int:
        """Number of sites of the group."""
        return int(np.prod(self.shape))

    @functools.cached_property
    def sites(self) -> slice:
        """Slice of the group's components in the state vector."""
        return slice(self.start, self.start + self.size)
