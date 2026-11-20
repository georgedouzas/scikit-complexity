"""Define the pair read spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ...specs._base_spec import BaseSpec
from ...types import Indices
from ._read_spec import _ReadSpec


@dataclass(frozen=True)
class _PairReadSpec(BaseSpec):
    """A spec of how a reduction over neighbours reads a symbol for each pair.

    Args:
        read:
            The values of the symbol, one per member or one shared value.

        side:
            `'shared'` for one value, `'reader'` for the value of the member that reads the reduction, `'other'`
            for the value of the neighbour, `'edge'` for the value of the member of a group on the edges of a
            network layer that joins them, or `'pair'` for the value of the pair itself, the entry of a shaped
            parameter at the position of the pair among the pairs of every member and every other member.

        image:
            Whether the symbol is a position component of the neighbour, read by its minimum image from the reader.

        table:
            The member that every neighbour stands for, for an `'other'` read whose neighbours are rows of a table,
            such as one other member of the hyperedges around the reader, or `None` for the neighbour itself.

        position:
            The place among the other members of a set of current neighbours of the reader, for an `'other'` read of
            a reduction over neighbour hyperedges, or `None` for a reduction over single neighbours.

        axis:
            The axis of the space a scalar position read by its minimum image runs along, such as 1 for `y` in a
            plane, or `None` for a component of a vector position.

        sites:
            The member every pair reads, or the number of members for a site beyond fixed edges, for an `'other'` read
            of a lattice state with edges of its own, such as `{'u': ('fixed', 0.5)}`, or `None` for the neighbour.

        fill:
            The value of the state beyond fixed edges.
    """

    read: _ReadSpec
    side: str
    image: bool = False
    table: Indices | None = field(default=None, compare=False)
    position: int | None = None
    axis: int | None = None
    sites: Indices | None = field(default=None, compare=False)
    fill: float = 0.0
