"""Tipping cascades: networks of interacting tipping elements that can push one another past their tipping points."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement
from ....core.interactions import CouplingInteraction
from ....core.laws import BaseLaw, RateLaw
from ....core.spaces import NetworkSpace
from ....core.system import System
from ....core.times import ContinuousTime
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model

FOLD = math.sqrt(4 / 27)


def _push(s: SimpleNamespace) -> Expression:
    """Return the push `d (x_j + 1)` of a neighbour `j` on a member, about `2 d` from a tipped neighbour."""
    return s.member.d * (s.other.x + 1)


class TippingElementElement(Element):
    """Tipping element in the normal form of a fold, pushed toward its upper state by tipped neighbours.

    The state follows `dx_i/dt = -x_i^3 + x_i + sqrt(4/27) c_i + d Σ_j A_ij (x_j + 1)`. Alone, the element has a
    lower and an upper stable state for `|c| < 1` and only the upper one beyond the critical value `c = 1`. A neighbour
    in its lower state `x_j ≈ -1` adds almost nothing, while a tipped neighbour at `x_j ≈ 1` adds about `2 d`. The
    element carries the fold, and the coupling of `tipping_cascade_network` adds the pushes of the neighbours.

    Args:
        label:
            The label of the element or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial state, near `-1` for an element in its lower state.

        c:
            The control parameter, critical at 1.

        d:
            The coupling strength of every link.

    Examples:
        >>> from skcomplex.models.earth.climate import TippingElementElement
        >>> TippingElementElement('ice sheet').x
        -1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = -1.0,
        c: Annotated[float, ParameterSpec()] = 0.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.c = c
        self.d = d

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the fold of the element.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law of the fold.
        """
        return [RateLaw(s.x, -(s.x**3) + s.x + FOLD * s.c)]


@register_model(
    field='earth/climate',
    tags=('tipping points', 'cascades', 'network', 'bistability', 'ode'),
    references=('Wunderling et al. 2021', 'Klose et al. 2020'),
    defaults={'until': 40.0, 'dt': 0.1},
)
def tipping_cascade_network(
    graph: nx.Graph | None = None,
    *,
    c: float | Sequence[float] = 0.0,
    d: float = 0.2,
    x: float | Sequence[float] = -1.0,
) -> System:
    """Return a network of tipping elements, where one element pushed past its tipping point can tip its neighbours.

    Every node is a fold `dx_i/dt = -x_i^3 + x_i + sqrt(4/27) c_i + d Σ_j A_ij (x_j + 1)` that starts in its lower
    state. An element whose control passes the critical value 1 tips to its upper state, and its coupling then raises
    the effective control of each neighbour by `2 d / sqrt(4/27)`, which can tip neighbours that would not tip alone.

    Args:
        graph:
            The NetworkX graph of the interactions, or `None` for a path of four elements.

        c:
            The control of every element, one value or one per node in node order.

        d:
            The coupling strength of every link.

        x:
            The initial states, one value or one per node in node order.

    Returns:
        system:
            The model, with the states `elements.x`, one value per node.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.earth.climate import tipping_cascade_network
        >>> tipping_cascade_network(nx.cycle_graph(5), c=[1.2, 0, 0, 0, 0]).elements[0].check().prepare().size_
        5
    """
    graph = nx.path_graph(4) if graph is None else graph
    size = graph.number_of_nodes()
    values = {
        'x': np.broadcast_to(np.asarray(x, dtype=float), size).copy(),
        'c': np.broadcast_to(np.asarray(c, dtype=float), size).copy(),
        'd': d,
    }
    elements = FieldElement('elements', element=TippingElementElement, space=NetworkSpace(graph), values=values)
    coupling = CouplingInteraction('coupling', target='elements', function=_push)
    return System(elements=[elements], interactions=[coupling], time=ContinuousTime())
