"""Define the dynamical system with states on every node of a graph."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from types import SimpleNamespace

import networkx as nx

from ....core.components.elements import BaseElement, Element
from ....core.components.interactions import BaseInteraction
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ._dynamical import DynamicalSystem
from ._equation import EquationInteraction
from .types import Equations, Noises


class NetworkDynamicalSystem(DynamicalSystem):
    """A dynamical system with states on every node of a graph and coupled neighbours.

    The states and parameters are the values of an element with one member on every node of the `NetworkSpace` of the
    graph. The equations are the local rates of every node. For every state with a coupling `g`, the rate of node `i`
    gains `sum_j g(x_i, x_j)` over its neighbours `j`, where `g` reads node `i` as `s.member` and its neighbour as
    `s.other`. An initial value is one number for every node or an array with one value per node. The system runs in
    continuous time.

    A subclass returns the graph from `_get_graph` and the couplings from `_get_coupling`, besides the hooks of
    `DynamicalSystem`.

    Unlike `DynamicalSystem`, which holds one copy of its states, it puts one copy on every node of a network space and
    sums the couplings over the neighbours. Choose it when identical units interact along the edges of a graph.

    It appears in many fields. Heat diffuses on a graph in mathematics, cells exchange ions through gap junctions in
    physiology, and oscillators synchronize on a network in physics.

    Attributes:
        system_:
            The `System` of the nodes, their equations, and their couplings built at the last `prepare`.

    Examples:
        Diffusion on a path of three nodes, which spreads the content of the first node evenly:

        >>> import networkx as nx
        >>> from skcomplex.systems.general.dynamics import NetworkDynamicalSystem
        >>> diffusion = NetworkDynamicalSystem(
        ...     nx.path_graph(3),
        ...     states={'u': [1.0, 0.0, 0.0]},
        ...     coupling={'u': lambda s: s.member.D * (s.other.u - s.member.u)},
        ...     parameters={'D': 1.0},
        ... )
        >>> trajectory = diffusion.prepare().simulate(until=20.0, dt=1.0)
        >>> [round(float(value), 3) for value in trajectory.get_values('nodes.u')[-1]]
        [0.333, 0.333, 0.333]

        Cells coupled by gap junctions, each relaxing to its own rest value while it exchanges with its neighbours:

        >>> cells = NetworkDynamicalSystem(
        ...     nx.cycle_graph(4),
        ...     states={'v': [1.0, 0.0, 0.0, 0.0]},
        ...     equations={'v': lambda s: -s.target.v / s.target.tau},
        ...     coupling={'v': lambda s: s.member.g * (s.other.v - s.member.v)},
        ...     parameters={'tau': 10.0, 'g': 0.5},
        ...     element='cells',
        ... )
        >>> trajectory = cells.prepare().simulate(until=5.0, dt=1.0)
        >>> bool(trajectory.get_values('cells.v')[-1][1] > 0)
        True
    """

    def __init__(
        self,
        graph: nx.Graph | None = None,
        states: Mapping[str, object] | None = None,
        equations: Equations | None = None,
        coupling: Equations | None = None,
        *,
        parameters: Mapping[str, object] | None = None,
        noises: Noises | None = None,
        convention: str = 'ito',
        element: str = 'nodes',
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        """Store the graph, states, equations, couplings, and the rest of the data.

        Args:
            graph:
                The NetworkX graph whose nodes hold the states, required.

            states:
                The initial value of every state by name, a number for every node or one value per node.

            equations:
                The local rate of every state by name, as a function of the namespace, or `None`.

            coupling:
                The term `g(x_i, x_j)` of one neighbour for every coupled state by name, as a function of a namespace
                whose `member` holds the node and `other` its neighbour, or `None`.

            parameters:
                The value of every parameter by name, or `None` for none.

            noises:
                The noise amplitude of every state that has one, by name, or `None` for none.

            convention:
                The stochastic calculus of the noises, `'ito'` or `'stratonovich'`.

            element:
                The label of the field of the nodes.

            label:
                The label of the system, which names it when it is nested in another one.

            solver:
                The solver of the problem, or `'auto'` for the one the system chooses.

            backend:
                The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.
        """
        super().__init__(
            states,
            equations,
            parameters=parameters,
            noises=noises,
            convention=convention,
            element=element,
            label=label,
            solver=solver,
            backend=backend,
        )
        self.graph = graph
        self.coupling = coupling

    def _get_time(self) -> str:
        """Return the continuous time of the rates."""
        return 'continuous'

    def _get_graph(self) -> nx.Graph:
        """Return the graph whose nodes hold the states.

        A subclass with a graph of its own returns it.

        Returns:
            graph:
                The graph given to the system.

        Raises:
            ValidationError:
                No graph is given.
        """
        if self.graph is None:
            error_msg = f'{type(self).__name__} has no graph. Give a NetworkX graph whose nodes hold the states.'
            raise ValidationError(error_msg)
        return self.graph

    def _get_coupling(self) -> Equations:
        """Return the term of one neighbour of every coupled state.

        A subclass with a coupling returns it.
        """
        return self.coupling or {}

    def _check_equations(self) -> None:
        """Raise an error for a missing graph, or an equation or coupling of no state.

        Raises:
            ValidationError:
                No graph is given, or an equation, a noise, or a coupling names a state the system does not declare.
        """
        super()._check_equations()
        self._get_graph()
        states = self._get_state_names()
        unknown = [name for name in self._get_coupling() if name not in states]
        if unknown:
            error_msg = (
                f'{type(self).__name__} has a coupling for the state {unknown[0]!r}, which it does not declare. '
                f'The declared states are {states}.'
            )
            raise ValidationError(error_msg)

    def _get_element(self) -> BaseElement:
        """Return the element with one member on every node, which holds the values."""
        values = {**self._get_states(), **self._get_parameters()}
        return Element(self._get_element_label(), values=values, space=NetworkSpace(self._get_graph()))

    @staticmethod
    def _get_neighbor_sum(coupling: Callable[[SimpleNamespace], object]) -> Callable[[SimpleNamespace], object]:
        """Return the rate of a coupling, its term summed over the neighbours of a node."""
        return lambda s: neighbors(coupling(SimpleNamespace(member=s.target, other=s.target.other, t=s.t)))

    def _get_coupling_interactions(self) -> list[EquationInteraction]:
        """Return a coupling of neighbours for every coupled state.

        Each coupling is labelled `<state>_coupling`.
        """
        element = self._get_element_label()
        return [
            EquationInteraction(
                f'{state}_coupling',
                target=element,
                state=state,
                equation=self._get_neighbor_sum(coupling),
            )
            for state, coupling in self._get_coupling().items()
        ]

    def _get_interactions(self) -> list[BaseInteraction]:
        """Return the local equations and the couplings of the nodes."""
        return [*super()._get_interactions(), *self._get_coupling_interactions()]
