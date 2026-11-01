"""Site of a percolation lattice: empty at 0, occupied at 1, burning at 2, or burnt at 3."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....core.laws import BaseLaw, LocalRuleLaw
from ....core.spaces import VonNeumannNeighborhood
from ....specs import StateSpec, VisualSpec
from ....types import Array

BURNING = 2


BURNT = 3


class PercolationSiteElement(Element):
    """Site of a percolation lattice: empty at 0, occupied at 1, burning at 2, or burnt at 3.

    Fire spreads from burning sites to their occupied neighbours and burns out in one step, so a fire lit along one
    edge burns exactly the clusters that touch that edge.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        cover:
            The initial cover, 0, 1, 2, or 3.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import PercolationSiteElement
        >>> PercolationSiteElement('site').cover
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        cover: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.cover = cover

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the spread and burning out of the fire.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRuleLaw(s.cover, VonNeumannNeighborhood(center=True), PercolationSiteElement._burn)]

    @staticmethod
    def _burn(cover: Array, *around: Array) -> Array:
        """Return the next cover of a site, which burns out, catches fire from a burning neighbour, or stays."""
        catches = (cover == 1) & np.any(np.stack(around) == BURNING, axis=0)
        return np.where(cover == BURNING, BURNT, np.where(catches, BURNING, cover))
