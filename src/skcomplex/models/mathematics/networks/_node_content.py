"""Content of a node of a network, which diffuses along the links with the coefficient `D`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class NodeContentElement(Element):
    """Content of a node of a network, which diffuses along the links with the coefficient `D`.

    Args:
        label:
            The label of the node or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial content.

        D:
            The diffusion coefficient of every link.

    Examples:
        >>> from skcomplex.models.mathematics.networks import NodeContentElement
        >>> NodeContentElement('node', D=0.5).D
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.D = D
