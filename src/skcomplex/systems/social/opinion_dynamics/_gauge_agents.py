"""Gauge theory of agents on a network.

It has phases on nodes, connections on edges, and frustrated loops.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import sin

from ....core.components import Observable
from ....core.components.elements import Element, Period
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import BaseSpace, CellComplexSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role


class TransportedCouplingInteraction(Interaction):
    """Relaxation of agent phases towards the phases of their neighbours.

    It adds `theta_i' += K sin(theta_j - theta_i - A_ij)`. The connection `A_ij` lives on the edge of the network from
    `i` to `j`, read as `A` of the element on the edges, and `A_ji = -A_ij`. The laws run once for every edge.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        among:
            The path of the group of agents on the nodes.

        K:
            The coupling strength.

        along:
            The edges the laws run on, `True` for the edges of the network the agents live on.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import TransportedCouplingInteraction
        >>> TransportedCouplingInteraction('coupling', among='agents', K=0.5).K
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, among: Role, K: float = 1.0, along: bool | str = True) -> None:
        super().__init__(label, among=among, along=along)
        self.K = K

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of both ends of an edge, reading the connection along and against its direction."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            source, target = context.source, context.target
            flow = context.K * sin(target.theta - source.theta - context.edge.A)
            self.laws_: list[BaseLaw] = [RateLaw(source.theta, flow), RateLaw(target.theta, -flow)]


class PhaseAgentElement(Element):
    """Agent with a phase `theta`.

    Args:
        label:
            The label of the agent or the group on the nodes.

        theta:
            The initial phase.

    Attributes:
        default_visual_:
            How the agent is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import PhaseAgentElement
        >>> PhaseAgentElement('agent', theta=0.5).theta
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[object, Period(2 * np.pi)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.theta = theta


class ConnectionElement(Element):
    """Edge of a network with a connection `A`, the phase an agent transports along it.

    Args:
        label:
            The label of the edge or the group on the edges.

        A:
            The connection along the direction of the edge.

    Attributes:
        default_visual_:
            How the edge is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import ConnectionElement
        >>> ConnectionElement('edge', A=0.5).A
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        A: object = 0.0,
        space: BaseSpace | None = None,
        on: str | None = None,
    ) -> None:
        super().__init__(label, space=space, on=on)
        self.A = A


class GaugeNetworkAgentsSystem(BaseSystem):
    """Agent phases on the nodes of a network.

    They relax towards the transported phases of their neighbours. Agent `i` follows
    `theta_i' = K sum_j sin(theta_j - theta_i - A_ij)` over its neighbours. A connection whose oriented sum around a
    loop is not a multiple of `2 pi` frustrates the loop: the transported differences cannot all vanish, so the agents
    keep a residual energy `sum (1 - cos(theta_j - theta_i - A_ij))` and a twist around the loop.

    Args:
        n_agents:
            The number of agents on a ring, when no graph is given.

        flux:
            The oriented sum of the connection around the first face of the minimum cycle basis, spread evenly over the
            edges of that face.

        K:
            The coupling strength.

        spread:
            The largest absolute initial phase.

        connections:
            The connection of every edge, in place of `flux`, or `None`.

        graph:
            The network, or `None` for a ring of `n_agents`.

        seed:
            The seed of the initial phases.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            A flux is given on a network without faces.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import GaugeNetworkAgentsSystem
        >>> [element.label for element in GaugeNetworkAgentsSystem(4).prepare().system_.elements]
        ['agents', 'links']
    """

    field = 'social/opinion_dynamics'
    topics = ('gauge', 'opinions', 'frustration', 'network', 'cell complex')
    references = ('Mihaylov and Cacciatori 2025',)
    until = 100.0
    dt = 0.5

    @staticmethod
    def _get_differences(edges: scipy.sparse.csr_array, values: Mapping[str, object]) -> Array:
        """Return the differences of the phases along the edges minus the connection.

        They are wrapped into one period around 0.
        """
        phases = np.ravel(np.asarray(values['agents.theta'], dtype=float))
        connection = np.ravel(np.asarray(values['links.A'], dtype=float))
        return np.angle(np.exp(1j * (edges @ phases - connection)))

    @staticmethod
    def _get_energy(edges: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
        """Return the sum over the edges of one minus the cosine of the difference.

        The difference is the transported one.
        """
        del t
        return float((1.0 - np.cos(GaugeNetworkAgentsSystem._get_differences(edges, values))).sum())

    @staticmethod
    def _get_twist(
        edges: scipy.sparse.csr_array,
        faces: scipy.sparse.csr_array,
        t: float,
        values: Mapping[str, object],
    ) -> float:
        """Return the largest absolute oriented sum of the transported differences.

        The sums run around the faces.
        """
        del t
        return float(np.abs(faces @ GaugeNetworkAgentsSystem._get_differences(edges, values)).max(initial=0.0))

    @staticmethod
    def _get_agent_observables(system: System) -> list[Observable]:
        """Return the energy of the transported differences and their twist.

        The twist is around the faces.
        """
        space = system.elements[0].space.prepare()
        return [
            Observable('energy', function=partial(GaugeNetworkAgentsSystem._get_energy, space.incidences_[0])),
            Observable('twist', function=partial(GaugeNetworkAgentsSystem._get_twist, *space.incidences_[:2])),
        ]

    def __init__(
        self,
        n_agents: int = 12,
        *,
        flux: float = 0.0,
        K: float = 1.0,
        spread: float = 1.0,
        connections: Sequence[float] | None = None,
        graph: nx.Graph | None = None,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.flux = flux
        self.K = K
        self.spread = spread
        self.connections = connections
        self.graph = graph
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        network = nx.cycle_graph(self.n_agents) if self.graph is None else self.graph
        space = CellComplexSpace(NetworkSpace(network)).prepare()
        faces = space.incidences_[1]
        if self.flux and not faces.shape[0] and self.connections is None:
            error_msg = (
                f'A flux of {self.flux} needs a loop, but the network has no faces. Give a network with a cycle.'
            )
            raise ValidationError(error_msg)
        if self.connections is not None:
            connection = np.asarray(self.connections, dtype=float)
        elif self.flux:
            first = faces[[0], :].toarray().ravel()
            connection = self.flux * first / np.abs(first).sum()
        else:
            connection = np.zeros(faces.shape[1])
        phases = np.random.default_rng(self.seed).uniform(-self.spread, self.spread, space.size_)
        agents = PhaseAgentElement('agents', theta=phases, space=space)
        links = ConnectionElement('links', A=connection, space=space, on='edges')
        coupling = TransportedCouplingInteraction('coupling', among='agents', K=self.K)
        self.system_ = System(self.label, elements=[agents, links], interactions=[coupling])
        if not self.observables_:
            self.observables_ = tuple(GaugeNetworkAgentsSystem._get_agent_observables(self.system_))
