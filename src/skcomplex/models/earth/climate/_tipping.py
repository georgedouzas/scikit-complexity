"""Tipping cascades: networks of interacting tipping elements that can push one another past their tipping points."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import networkx as nx
import numpy as np

from ....core.base import Domain, parameter, state
from ....core.components import Element, Field
from ....core.laws import Law, Rate, Symbols, laplacian
from ....core.runtime import register_model
from ....core.space import ContinuousTime, NetworkSpace
from ....core.system import System

FOLD = math.sqrt(4 / 27)


class TippingElement(Element):
    """Tipping element in the normal form of a fold, pushed toward its upper state by tipped neighbours.

    The state follows `dx_i/dt = -x_i^3 + x_i + sqrt(4/27) c_i + d Σ_j A_ij (x_j + 1)`. Alone, the element has a
    lower and an upper stable state for `|c| < 1` and only the upper one beyond the critical value `c = 1`. A neighbour
    in its lower state `x_j ≈ -1` adds almost nothing, while a tipped neighbour at `x_j ≈ 1` adds about `2 d`. The
    coupling sum is written with the graph Laplacian as `laplacian(x) + k_i (x_i + 1)`, where `k_i` is the degree of
    the node.

    Args:
        label:
            The label of the element or the field.

        x:
            The initial state, near `-1` for an element in its lower state.

        c:
            The control parameter, critical at 1.

        d:
            The coupling strength of every link.

        degree:
            The number of neighbours of the node.

    Examples:
        >>> from skcomplex.models.earth.climate import TippingElement
        >>> TippingElement('ice sheet').x
        -1.0
    """

    x: float = state(default=-1.0)
    c: float = parameter(default=0.0)
    d: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    degree: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the fold with its coupling to the neighbours.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        coupling = laplacian(s.x) + s.degree * (s.x + 1)
        return [Rate(s.x, -(s.x**3) + s.x + FOLD * s.c + s.d * coupling)]


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
        >>> tipping_cascade_network(nx.cycle_graph(5), c=[1.2, 0, 0, 0, 0]).elements[0].size
        5
    """
    graph = nx.path_graph(4) if graph is None else graph
    size = graph.number_of_nodes()
    values = {
        'x': np.broadcast_to(np.asarray(x, dtype=float), size).copy(),
        'c': np.broadcast_to(np.asarray(c, dtype=float), size).copy(),
        'd': d,
        'degree': np.array([graph.degree[node] for node in graph.nodes], dtype=float),
    }
    elements = Field('elements', TippingElement, NetworkSpace(graph), values=values)
    return System(components=[elements], time=ContinuousTime())
