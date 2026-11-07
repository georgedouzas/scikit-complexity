"""Node of a growing network with an intrinsic fitness, which scales its chance to win new edges."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.specs import StateSpec, VisualSpec
from ._vertex import VertexElement


class FitnessVertexElement(VertexElement):
    """Node of a growing network with an intrinsic fitness, which scales its chance to win new edges.

    A new edge joins the node with probability proportional to its fitness times its degree, as in the model of
    Bianconi and Barabási. The fitness is drawn when the node joins and stays fixed.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

        fitness:
            The fitness of the node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import FitnessVertexElement
        >>> FitnessVertexElement('nodes', fitness=0.5).fitness
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: float = 0.0,
        degree: float = 0.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual, age=age, degree=degree)
        self.fitness = fitness
