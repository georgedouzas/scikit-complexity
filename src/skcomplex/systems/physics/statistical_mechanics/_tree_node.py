"""Node of a tree that holds a number `n` of energy quanta, drawn at a fixed position."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.specs import StateSpec, VisualSpec
from ._particle_site import ParticleSiteElement


class TreeNodeElement(ParticleSiteElement):
    """Node of a tree that holds a number `n` of energy quanta, drawn at a fixed position.

    Args:
        label:
            The label of the node or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of quanta.

        x:
            The position of the node in the drawing of the tree.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import TreeNodeElement
        >>> TreeNodeElement('root').n
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec(
            'points',
            {'position': 'x', 'radius': 0.04, 'color_by': 'n', 'color_range': (0.0, 2.0), 'states': (), 'trail': 0},
        ),
        *,
        n: float = 0.0,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, n=n)
        self.x = x
