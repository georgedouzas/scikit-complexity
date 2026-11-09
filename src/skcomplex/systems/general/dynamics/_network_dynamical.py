"""Define network dynamical systems.

A network dynamical system puts the states of a dynamical system on every node of a graph and couples neighbours.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import networkx as nx

from ....core.components.elements import BaseElement, DeclaredElement, FieldElement
from ....core.components.interactions import BaseInteraction, CouplingInteraction
from ....core.components.spaces import NetworkSpace
from ....core.errors import ValidationError
from ....core.specs import StateSpec
from ._dynamical import DynamicalSystem
from .types import Equations, Noises


class NetworkDynamicalSystem(DynamicalSystem):
    """States on every node of a graph, with local equations and a coupling of neighbours.

    The states and the parameters are those of a `DeclaredElement`, a member of a `FieldElement` on the `NetworkSpace`
    of the graph. The equations are the local rates of every node, and for every state with a coupling `g` the rate of
    node `i` gains the sum over its neighbours `sum_j A_ij g(x_i, x_j)`, a `CouplingInteraction` that reads node `i` as
    `s.member` and its neighbour `j` as `s.other`. An initial value is a number for every node or an array with one
    value per node. The system runs in continuous time.

    Diffusion on a graph in mathematics and the exchange of gap junctions between cells in physiology are network
    dynamical systems. A subclass returns the graph from `_get_graph` and the couplings from `_get_coupling`, besides
    the hooks of `DynamicalSystem`.

    Attributes:
        system_:
            The composite of the nodes, their equations, and their couplings built at the last `prepare`.

        laws_:
            The laws the system contributes for the namespace of its last run.

    Examples:
        Diffusion on a path of three nodes, which spreads the content of the first node evenly:

        >>> import networkx as nx
        >>> from skcomplex.systems.general.dynamics import NetworkDynamicalSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> diffusion = NetworkDynamicalSystem(
        ...     nx.path_graph(3),
        ...     states={'u': [1.0, 0.0, 0.0]},
        ...     coupling={'u': lambda s: s.member.D * (s.other.u - s.member.u)},
        ...     parameters={'D': 1.0},
        ... )
        >>> trajectory = simulate(diffusion, until=20.0, dt=1.0)
        >>> [round(float(value), 3) for value in trajectory.values('nodes.u')[-1]]
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
        >>> trajectory = simulate(cells, until=5.0, dt=1.0)
        >>> bool(trajectory.values('cells.v')[-1][1] > 0)
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
    ) -> None:
        """Store the graph, the states, the equations, the couplings, and the rest of the data.

        Args:
            graph:
                The NetworkX graph whose nodes hold the states, required.

            states:
                The initial value of every state by name, a number for every node or one value per node.

            equations:
                The local rate of every state by name, as a function of the namespace, or `None`.

            coupling:
                The term `g(x_i, x_j)` of one neighbour for every coupled state by name, as a function of the
                namespace of a `CouplingInteraction`, or `None`.

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
        """
        super().__init__(
            states,
            equations,
            parameters=parameters,
            noises=noises,
            convention=convention,
            element=element,
            label=label,
        )
        self.graph = graph
        self.coupling = coupling

    def _get_time(self) -> str:
        """Return the continuous time of the rates."""
        return 'continuous'

    def _get_graph(self) -> nx.Graph:
        """Return the graph whose nodes hold the states.

        A subclass with a graph of its own returns it.

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
        """Raise an error for a missing graph, or for the equation or the coupling of an unknown state.

        Raises:
            ValidationError:
                No graph is given, or an equation, a noise, or a coupling names a state the system does not declare.
        """
        super()._check_equations()
        self._get_graph()
        states = [name for name, variable in self._get_variables().items() if isinstance(variable.spec, StateSpec)]
        unknown = [name for name in self._get_coupling() if name not in states]
        if unknown:
            error_msg = (
                f'{type(self).__name__} has a coupling for the state {unknown[0]!r}, which it does not declare. '
                f'The declared states are {states}.'
            )
            raise ValidationError(error_msg)

    def _get_element(self) -> BaseElement:
        """Return the field of the nodes whose members hold the states and the parameters."""
        return FieldElement(
            self._get_element_label(),
            element=DeclaredElement,
            space=NetworkSpace(self._get_graph()),
            values={'variables': self._get_variables()},
        )

    def _get_coupling_interactions(self) -> list[CouplingInteraction]:
        """Return a coupling of neighbours for every coupled state, labelled `<state>_coupling`."""
        element = self._get_element_label()
        return [
            CouplingInteraction(f'{state}_coupling', target=element, read=state, write=state, function=function)
            for state, function in self._get_coupling().items()
        ]

    def _get_interactions(self) -> list[BaseInteraction]:
        """Return the local equations and the couplings of the nodes."""
        return [*super()._get_interactions(), *self._get_coupling_interactions()]
