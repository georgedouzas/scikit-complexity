"""Define Moore neighbourhoods, the sites within a radius along every dimension."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools

from ...types import IntTuple
from ._base import BaseNeighborhood
from ._lattice_neighborhood import LatticeNeighborhood


class MooreNeighborhood(LatticeNeighborhood):
    """Sites within a radius along every dimension.

    In two dimensions and for radius 1 they are the 8 surrounding cells. The offsets come in lexicographic order, after
    the site itself when it is included.

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
        >>> from skcomplex.core.components.spaces import MooreNeighborhood
        >>> len(MooreNeighborhood().prepare().offsets_)
        8
        >>> len(MooreNeighborhood(center=True).prepare().offsets_)
        9
    """

    def __init__(self, radius: int = 1, *, n_dim: int = 2, center: bool = False, label: str = 'neighborhood') -> None:
        BaseNeighborhood.__init__(self, label)
        self.radius = radius
        self.n_dim = n_dim
        self.center = center

    def _get_offsets(self) -> tuple[IntTuple, ...]:
        """Return the offsets within the radius along every dimension.

        They come after the site itself when included.
        """
        steps = range(-self.radius, self.radius + 1)
        offsets = tuple(offset for offset in itertools.product(steps, repeat=self.n_dim) if any(offset))
        return ((0,) * self.n_dim, *offsets) if self.center else offsets
