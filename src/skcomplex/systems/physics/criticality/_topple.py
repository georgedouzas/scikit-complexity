"""Parallel toppling of a sandpile without drive, where every site at or above the threshold topples once a step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np

from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import VonNeumannNeighborhood
from ....core.specs import RoleSpec
from ....core.types import Array


class ToppleInteraction(LocalRuleInteraction):
    """Parallel toppling of a sandpile without drive, where every site at or above the threshold topples once a step.

    Every step, every site that holds at least `threshold` grains loses `threshold` grains, and every site gains one
    grain from each toppling neighbour, all at once. On a periodic lattice no grain is lost, so the number of grains is
    conserved, and the activity either dies out in an absorbing configuration or lasts forever.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the lattice group of sites.

        threshold:
            The number of grains at which a site topples, the number of its neighbours.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ToppleInteraction
        >>> ToppleInteraction('topple', target='pile').threshold
        4
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], threshold: int = 4) -> None:
        super().__init__(
            label,
            target=target,
            state='height',
            rule=self._topple,
            neighborhood=VonNeumannNeighborhood(center=True),
        )
        self.threshold = threshold

    def _topple(self, height: Array, *around: Array) -> Array:
        """Return the heights after every site at or above the threshold topples once."""
        gained = np.sum([neighbor >= self.threshold for neighbor in around], axis=0)
        return height - self.threshold * (height >= self.threshold) + gained
