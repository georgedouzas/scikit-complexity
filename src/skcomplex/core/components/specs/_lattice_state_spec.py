"""Define the lattice state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from ...specs._base_spec import BaseSpec
from ..spaces import BaseSpace


@dataclass(frozen=True)
class LatticeStateSpec(BaseSpec):
    """State of a lattice group in the state vector.

    Args:
        name:
            The dotted name of the state.

        start:
            The index of its first component.

        space:
            The lattice of the group, or the network whose nodes are its sites.

        part:
            The part of the space with one member each, `'sites'`, `'edges'`, or `'hyperedges'`.

        rows:
            The number of members on a part other than the sites, or `None` for one member per site.

        components:
            The number of components of the state of one member, such as the real components of a matrix.
    """

    name: str
    start: int
    space: BaseSpace
    part: str = 'sites'
    rows: int | None = None
    components: int = 1

    @functools.cached_property
    def shape(self) -> tuple[int, ...]:
        """Shape of the sites of the group.

        On a part other than the sites, it is the number of its members.
        """
        return self.space.shape_ if self.rows is None else (self.rows,)

    @functools.cached_property
    def size(self) -> int:
        """Number of sites of the group."""
        return int(np.prod(self.shape)) if self.rows is None else self.rows

    @functools.cached_property
    def sites(self) -> slice:
        """Slice of the group's components in the state vector, member after member."""
        return slice(self.start, self.start + self.size * self.components)
