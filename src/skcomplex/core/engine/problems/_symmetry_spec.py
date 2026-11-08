"""Define the symmetry spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...specs._base_spec import BaseSpec
from ...types import Indices


@dataclass(frozen=True, eq=False)
class SymmetrySpec(BaseSpec):
    """Mirror symmetry of a field state on a periodic lattice.

    The rates of the state are projected onto odd or even parts.

    Args:
        indices:
            The state indices of the field state, one per lattice site.

        shape:
            The shape of the lattice.

        signs:
            The axes of the symmetry with their signs, `-1` for an odd and `1` for an even part.
    """

    indices: Indices
    shape: tuple[int, ...]
    signs: tuple[tuple[int, int], ...]
