"""Node of a popularity-similarity network, with an angle on the circle next to its age and degree."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....specs import StateSpec, VisualSpec
from ._vertex import VertexElement


class SimilarityVertexElement(VertexElement):
    """Node of a popularity-similarity network, with an angle on the circle next to its age and degree.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

        angle:
            The angle of the node on the circle, its similarity coordinate.

    Examples:
        >>> from skcomplex.models.mathematics.networks import SimilarityVertexElement
        >>> SimilarityVertexElement('nodes', angle=1.5).angle
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: float = 0.0,
        degree: float = 0.0,
        angle: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, age=age, degree=degree)
        self.angle = angle
