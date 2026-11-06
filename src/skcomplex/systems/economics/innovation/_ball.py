"""Define the balls of an urn, one per colour with its multiplicity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._ball_view import BallView


class BallElement(Element):
    """Balls of one colour in the urn of a node, with how many there are and whether the node has drawn the colour.

    A ball agent stands for every ball of its colour in the urn of the node it sits on, so the set of ball agents on a
    node is the set of colours of its urn, and their multiplicities make the urn a multiset.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        color:
            The colour, a number shared by the balls of that colour in every urn.

        count:
            The number of balls of the colour in the urn.

        drawn:
            1 when the node has drawn the colour, and 0 when it has not.

    Examples:
        >>> from skcomplex.systems.economics.innovation import BallElement
        >>> BallElement('balls', color=3, count=2).count
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        color: Annotated[float, StateSpec(domain=NN)] = 0,
        count: Annotated[float, StateSpec(domain=NN)] = 1,
        drawn: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.color = color
        self.count = count
        self.drawn = drawn

    @classmethod
    def _get_view(cls) -> type[BallView]:
        """Return the view of the agents of the element."""
        return BallView
