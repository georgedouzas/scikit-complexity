"""Slowly driven sandpile: every step drops one grain on a random site, and the full sites topple in avalanches."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np

from ....core.components.interactions import CascadeInteraction
from ....core.specs import RoleSpec
from ....core.types import Array


class AvalancheInteraction(CascadeInteraction):
    """Slowly driven sandpile: every step drops one grain on a random site, and the full sites topple in avalanches.

    A site that holds `threshold` grains topples. In the mode `'redistribute'` it sends one grain to each of its
    neighbours, as in the Bak-Tang-Wiesenfeld sandpile, and in the mode `'random'` every grain goes to a neighbour drawn
    at random, as in the Manna sandpile. Grains that topple over a fixed edge leave the pile, and every avalanche is
    recorded as an event under the label of the interaction.

    Args:
        label:
            The label of the interaction and of its avalanche events.

        target:
            The path of the lattice group of sites.

        state:
            The name of the state that holds the grains.

        threshold:
            The number of grains at which a site topples.

        mode:
            `'redistribute'` or `'random'`.

    Examples:
        >>> from skcomplex.systems.physics.criticality import AvalancheInteraction
        >>> AvalancheInteraction('avalanche', target='pile', threshold=2, mode='random').share
        1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'height',
        threshold: int = 4,
        mode: str = 'redistribute',
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            threshold=threshold,
            share=1,
            mode=mode,
            drive=self._drop_grain,
        )

    @staticmethod
    def _drop_grain(heights: Array, *, random: np.random.Generator) -> Array:
        """Return the heights with one grain added on a random site."""
        following = heights.copy()
        following[tuple(random.integers(size) for size in heights.shape)] += 1
        return following
