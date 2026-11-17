"""Tipping cascades.

Networks of interacting tipping elements can push one another past their tipping points.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System

FOLD = math.sqrt(4 / 27)


class TippingElementElement(Element):
    """Tipping elements in the normal form of a fold, their states and controls.

    Alone, an element has a lower and an upper stable state for `|c| < 1` and only the upper one beyond the critical
    value `c = 1`. `TippingCascadeInteraction` gives the fold and the pushes of the neighbours.

    Args:
        label:
            The label of the elements.

        x:
            The initial state, near `-1` for an element in its lower state, one value or one per element.

        c:
            The control parameter, critical at 1, one value or one per element.

        space:
            The network of the elements, or `None` for one element.

    Examples:
        >>> from skcomplex.systems.physics.climate import TippingElementElement
        >>> TippingElementElement('ice sheet').x
        -1.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: object = -1.0,
        c: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x
        self.c = c


class TippingCascadeInteraction(Interaction):
    """Fold of every tipping element, pushed by its tipped neighbours.

    The state follows `dx_i/dt = -x_i^3 + x_i + sqrt(4/27) c_i + d Σ_j A_ij (x_j + 1)`. A neighbour in its lower state
    `x_j ≈ -1` adds almost nothing, while a tipped neighbour at `x_j ≈ 1` adds about `2 d`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the elements on a network.

        d:
            The coupling strength of every link.

    Examples:
        >>> from skcomplex.systems.physics.climate import TippingCascadeInteraction
        >>> TippingCascadeInteraction('cascade', target='elements', d=0.1).d
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        d: Annotated[float, RealSet.closed_open(0, oo)] = 0.2,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.d = d

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the fold of every element and the pushes of its neighbours.

        Args:
            context:
                The namespace of the elements, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, c = context.target.x, context.target.c
            push = neighbors(context.d * (context.target.other.x + 1))
            self.laws_ = [RateLaw(x.diff(context.t), -(x**3) + x + FOLD * c + push)]


class TippingCascadeNetworkSystem(BaseSystem):
    """A network of tipping elements.

    One element pushed past its tipping point can tip its neighbours.

    Every node is a fold `dx_i/dt = -x_i^3 + x_i + sqrt(4/27) c_i + d Σ_j A_ij (x_j + 1)` that starts in its lower
    state. An element whose control passes the critical value 1 tips to its upper state, and its coupling then raises
    the effective control of each neighbour by `2 d / sqrt(4/27)`, which can tip neighbours that would not tip alone.
    The system has the states `elements.x`, one value per node.

    Args:
        graph:
            The NetworkX graph of the interactions, or `None` for a path of four elements.

        c:
            The control of every element, one value or one per node in node order.

        d:
            The coupling strength of every link.

        x:
            The initial states, one value or one per node in node order.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.physics.climate import TippingCascadeNetworkSystem
        >>> system = TippingCascadeNetworkSystem(nx.cycle_graph(5), c=[1.2, 0, 0, 0, 0]).prepare()
        >>> system.simulate(until=1.0).get_values('elements.x').shape
        (11, 5)
    """

    field = 'physics/climate'
    topics = ('tipping points', 'cascades', 'network', 'bistability', 'ode')
    references = ('Wunderling et al. 2021', 'Klose et al. 2020')
    until = 40.0
    dt = 0.1

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        c: float | Sequence[float] = 0.0,
        d: float = 0.2,
        x: float | Sequence[float] = -1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.graph = graph
        self.c = c
        self.d = d
        self.x = x

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the tipping elements on the network and their cascade.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        graph = nx.path_graph(4) if self.graph is None else self.graph
        size = graph.number_of_nodes()
        elements = TippingElementElement(
            'elements',
            x=np.broadcast_to(np.asarray(self.x, dtype=float), size).copy(),
            c=np.broadcast_to(np.asarray(self.c, dtype=float), size).copy(),
            space=NetworkSpace(graph),
        )
        cascade = TippingCascadeInteraction('coupling', target='elements', d=self.d)
        self.system_ = System(self.label, elements=[elements], interactions=[cascade])
