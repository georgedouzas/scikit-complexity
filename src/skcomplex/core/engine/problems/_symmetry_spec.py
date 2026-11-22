"""Define the symmetry spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...types import Indices


@dataclass(frozen=True, eq=False)
class SymmetrySpec(BaseSpec):
    """A spec of a mirror symmetry of a field on a periodic lattice.

    A field that is odd or even about an axis, such as a velocity that changes sign under a mirror, should keep that
    symmetry, but rounding errors break it over many steps. The engine projects the rates of the field onto the odd or
    even part along every axis of the symmetry, so it holds exactly. It keeps the symmetry of a flow, a wave, and a
    pattern of a reaction.

    Read more in the user guide, under Solvers.

    Args:
        indices:
            The state indices of the field state, one per lattice site.

        shape:
            The shape of the lattice.

        signs:
            The axes of the symmetry with their signs, `-1` for an odd and `1` for an even part.

    See Also:
        JacobianKernel: Reads the symmetries of the fields.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.problems._symmetry_spec import SymmetrySpec
        >>> SymmetrySpec(np.arange(4), (4,), ((0, -1),)).signs
        ((0, -1),)
    """

    indices: Indices
    shape: tuple[int, ...]
    signs: tuple[tuple[int, int], ...]
