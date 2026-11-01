"""Stochastic rule of the Drossel-Schwabl forest fire on a cover of 0 for empty, 1 for a tree, and 2 for burning."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np

from ....core.interactions import LocalRuleInteraction
from ....core.spaces import (
    BaseNeighborhood,
    VonNeumannNeighborhood,
)
from ....specs import RoleSpec
from ....types import Array

NEAREST = VonNeumannNeighborhood(center=True)


BURNING = 2


class FireInteraction(LocalRuleInteraction):
    """Stochastic rule of the Drossel-Schwabl forest fire on a cover of 0 for empty, 1 for a tree, and 2 for burning.

    Every step a tree catches fire from a burning neighbour or from lightning with the probability `lightning` of the
    site, a burning site becomes empty, and a tree grows on an empty site with the probability `growth` of the site.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the lattice or network group of sites.

        neighborhood:
            The neighbours a fire spreads to, the von Neumann neighbourhood of a square lattice, the six neighbours of
            a triangular lattice, or the neighbours of a node, each with the site itself.

    Examples:
        >>> from skcomplex.models.physics.criticality import FireInteraction
        >>> FireInteraction('fire', target='forest').stochastic
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        neighborhood: BaseNeighborhood = NEAREST,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state='cover',
            rule=self._burn,
            neighborhood=neighborhood,
            stochastic=True,
        )

    @staticmethod
    def _burn(cover: Array, *around: Array, growth: float | Array, lightning: float | Array, random: Array) -> Array:
        """Return the covers after trees catch fire, fires burn out, and trees grow."""
        burning = np.any(np.stack(around) == BURNING, axis=0) if around else np.zeros(np.shape(cover), dtype=bool)
        ignites = burning | (random < lightning)
        tree = np.where(ignites, float(BURNING), 1.0)
        empty = np.where(random < growth, 1.0, 0.0)
        return np.where(cover == BURNING, 0.0, np.where(cover == 1, tree, empty))
