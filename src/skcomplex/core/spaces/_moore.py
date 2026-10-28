"""Define Moore neighbourhoods, the sites within a radius along every dimension."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools

from ...types import Offset
from ._base import BaseNeighborhood
from ._lattice_neighborhood import LatticeNeighborhood


class MooreNeighborhood(LatticeNeighborhood):
    """Sites within a radius along every dimension, the 8 surrounding cells in two dimensions for radius 1.

    The offsets come in lexicographic order, after the site itself when it is included.

    Args:
        radius:
            The largest offset along any dimension.

        n_dim:
            The number of dimensions.

        center:
            Whether the site itself comes first.

        label:
            The label of the neighbourhood.

    Examples:
        >>> from skcomplex.core.spaces import MooreNeighborhood
        >>> len(MooreNeighborhood().check().prepare().offsets_)
        8
        >>> len(MooreNeighborhood(center=True).check().prepare().offsets_)
        9
    """

    def __init__(self, radius: int = 1, *, n_dim: int = 2, center: bool = False, label: str = 'neighborhood') -> None:
        BaseNeighborhood.__init__(self, label)
        self.radius = radius
        self.n_dim = n_dim
        self.center = center

    def _get_offsets(self) -> tuple[Offset, ...]:
        """Return the offsets within the radius along every dimension, after the site itself when included."""
        steps = range(-self.radius, self.radius + 1)
        offsets = tuple(offset for offset in itertools.product(steps, repeat=self.n_dim) if any(offset))
        return ((0,) * self.n_dim, *offsets) if self.center else offsets
