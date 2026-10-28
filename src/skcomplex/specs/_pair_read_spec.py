"""Define the pair read spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec
from ._read_spec import ReadSpec


@dataclass(frozen=True)
class PairReadSpec(BaseSpec):
    """How a neighbourhood reduction reads one symbol for every pair of a member and a neighbour.

    Args:
        read:
            The values of the symbol, one per member or one shared value.

        side:
            `'shared'` for one value, `'reader'` for the value of the member that reads the reduction, `'other'`
            for the value of the neighbour, or `'edge'` for the value of the member of a group on the edges of a
            network layer that joins them.

        image:
            Whether the symbol is a position component of the neighbour, read by its minimum image from the reader.
    """

    read: ReadSpec
    side: str
    image: bool = False
