"""Self-organized branching process: avalanches on a binary tree whose branching probability the avalanches tune."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class BranchingTreeElement(Element):
    """Self-organized branching process: avalanches on a binary tree whose branching probability the avalanches tune.

    The probability `p` that an active site relaxes into two active sites of the next generation, the number `z` of
    active sites of the current generation of the avalanche, and its generation `g`. Map interactions advance the
    avalanche by one generation per step.

    Args:
        label:
            The label of the tree.

        visual:
            How the tree is drawn, or `None` for `default_visual_`.

        p:
            The initial branching probability.

        z:
            The initial number of active sites.

        g:
            The initial generation.

    Attributes:
        default_visual_:
            The visual of the tree when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.criticality import BranchingTreeElement
        >>> BranchingTreeElement('tree').p
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.3,
        z: Annotated[float, StateSpec(domain=NN)] = 1,
        g: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p = p
        self.z = z
        self.g = g
