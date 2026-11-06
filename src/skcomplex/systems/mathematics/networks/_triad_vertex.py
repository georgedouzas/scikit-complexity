"""Node of a Holme-Kim network, which closes triangles with the neighbours of its first contact when it joins."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._vertex import VertexElement


class TriadVertexElement(VertexElement):
    """Node of a Holme-Kim network, which closes triangles with the neighbours of its first contact when it joins.

    One step after a node joins with one edge by preferential attachment, the triad formation of the system adds
    `extra` edges to it. Each new end is a neighbour of its first contact with probability `triad`, and a node drawn by
    preferential attachment otherwise. The state `near` marks the nodes next to the newest node.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

        near:
            The number of neighbours of the node that joined in the previous step.

        triad:
            The probability that a new edge closes a triangle.

        extra:
            The number of edges a node adds after its first one.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import TriadVertexElement
        >>> TriadVertexElement('nodes', triad=0.5).triad
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: float = 0.0,
        degree: float = 0.0,
        near: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        triad: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        extra: Annotated[int, ParameterSpec(domain=NN)] = 1,
    ) -> None:
        super().__init__(label, visual=visual, age=age, degree=degree)
        self.near = near
        self.triad = triad
        self.extra = extra
