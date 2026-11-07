"""Gauge theory of agents on a network: phases on nodes, connections on edges, and frustrated loops."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import pi, sin

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import CouplingInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import CellComplexSpace, NetworkSpace
from ....core.components.system import BaseSystem, RunSettings, System
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class TransportedCouplingInteraction(CouplingInteraction):
    """Relaxation of agent phases towards the phases of their neighbours, `theta_i' += K sin(theta_j - theta_i - A_ij)`.

    The connection `A_ij` lives on the edge of the network from `i` to `j`, and `A_ji = -A_ij`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of agents on the nodes.

        links:
            The path of the group on the edges, whose parameter `A` holds the connection.

        K:
            The coupling strength.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import TransportedCouplingInteraction
        >>> TransportedCouplingInteraction('coupling', target='agents', links='links', K=0.5).K
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, target=target, read='theta', write='theta')
        self.links = links
        self.K = K

    def _get_edge_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of both ends of an edge, with the connection read along and against its direction."""
        source, target = namespace.source, namespace.target
        flow = namespace.K * sin(target.theta - source.theta - namespace.edge.A)
        return [RateLaw(source.theta, flow), RateLaw(target.theta, -flow)]


class PhaseAgentElement(Element):
    """Agent with a phase `theta`.

    Args:
        label:
            The label of the agent or the group on the nodes.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        theta:
            The initial phase.

    Attributes:
        default_visual_:
            The visual of the agent when it sets none.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import PhaseAgentElement
        >>> PhaseAgentElement('agent', theta=0.5).theta
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta


class ConnectionElement(Element):
    """Edge of a network with a connection `A`, the phase an agent transports along it.

    Args:
        label:
            The label of the edge or the group on the edges.

        visual:
            How the edge is drawn, or `None` for `default_visual_`.

        A:
            The connection along the direction of the edge.

    Attributes:
        default_visual_:
            The visual of the edge when it sets none.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import ConnectionElement
        >>> ConnectionElement('edge', A=0.5).A
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A


def _differences(edges: scipy.sparse.csr_array, values: Mapping[str, object]) -> Array:
    """Return the differences of the phases along the edges minus the connection, wrapped into one period around 0."""
    phases = np.ravel(np.asarray(values['agents.theta'], dtype=float))
    connection = np.ravel(np.asarray(values['links.A'], dtype=float))
    return np.angle(np.exp(1j * (edges @ phases - connection)))


def _energy(edges: scipy.sparse.csr_array, t: float, values: Mapping[str, object]) -> float:
    """Return the sum over the edges of one minus the cosine of the transported difference."""
    del t
    return float((1.0 - np.cos(_differences(edges, values))).sum())


def _twist(
    edges: scipy.sparse.csr_array,
    faces: scipy.sparse.csr_array,
    t: float,
    values: Mapping[str, object],
) -> float:
    """Return the largest absolute oriented sum of the transported differences around a face."""
    del t
    return float(np.abs(faces @ _differences(edges, values)).max(initial=0.0))


def _agent_observables(system: System) -> list[ObservableSpec]:
    """Return the energy of the transported differences and their twist around the faces."""
    space = system.elements[0].space.prepare()
    return [
        ObservableSpec('energy', function=partial(_energy, space.incidences_[0])),
        ObservableSpec('twist', function=partial(_twist, *space.incidences_[:2])),
    ]


class GaugeNetworkAgentsSystem(BaseSystem):
    """Agent phases on the nodes of a network that relax towards the transported phases of their neighbours.

    Agent `i` follows `theta_i' = K sum_j sin(theta_j - theta_i - A_ij)` over its neighbours. A connection whose
    oriented sum around a loop is not a multiple of `2 pi` frustrates the loop: the transported differences cannot all
    vanish, so the agents keep a residual energy `sum (1 - cos(theta_j - theta_i - A_ij))` and a twist around the loop.

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
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.flux = flux
        self.K = K
        self.spread = spread
        self.connections = connections
        self.graph = graph
        self.seed = seed

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
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
        agents = GroupElement('agents', element=PhaseAgentElement, space=space, on='sites', values={'theta': phases})
        links = GroupElement(
            'links',
            element=ConnectionElement,
            space=space,
            on='edges',
            values={'A': connection},
        )
        coupling = TransportedCouplingInteraction('coupling', target='agents', links='links', K=self.K)
        return System(self.label, elements=[agents, links], interactions=[coupling])

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings, with the observables of the parts built from the model parameters."""
        settings = super().get_run_settings()
        settings['observables'] = _agent_observables(self._build_system())
        return settings
