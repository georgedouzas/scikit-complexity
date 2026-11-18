"""Neural network systems: Hopfield's associative memory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.components.types import SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver
from ....core.specs import Role, VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array


class NeuronView(AgentView):
    """View of a neuron of a Hopfield network, with its states as attributes."""

    spin: float
    weights: Array


class NeuronElement(Element):
    """Binary neuron of a Hopfield network.

    The update aligns it with the sign of its weighted input. The neuron sits on its node of a complete graph and holds
    the weights of its incoming links, one per node.

    Args:
        label:
            The label of the agent or group.

        spin:
            The initial state, -1 or 1.

        weights:
            The weights of the links from every node, an array with one value per node.

        values:
            Further values by name, such as the node `node` of every neuron.

        size:
            The number of neurons, or `None` for one.

        space:
            The complete graph whose nodes hold the neurons, or `None` for none.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import NeuronElement
        >>> group = NeuronElement('neurons', spin=1.0, size=4)
        >>> group.label, group.size
        ('neurons', 4)
    """

    visual = VisualSpec('agents', {'color_by': 'spin', 'color_range': (-1.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        spin: Annotated[object, ZZ] = 1.0,
        weights: object = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.spin = spin
        self.weights = weights

    @classmethod
    def _get_view(cls) -> type[NeuronView]:
        """Return the view of the neurons.

        It types their states.
        """
        return NeuronView


class HopfieldUpdateInteraction(Interaction):
    """Asynchronous update of a Hopfield network.

    Every neuron in turn takes the sign of its weighted input. The neurons update one after the other in a new random
    order every step, each reading the states the neurons before it set, so the energy `-sum w_ij s_i s_j / 2` never
    increases. A neuron whose input is zero keeps its state.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of neurons.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import HopfieldUpdateInteraction
        >>> HopfieldUpdateInteraction('update', target='neurons').target
        'neurons'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the update of every neuron, which the agent solver runs in a new random order every step."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self.update)]

    def update(self, agent: NeuronView, system: SystemViewProtocol) -> None:
        """Set a neuron to the sign of its weighted input from the other neurons, or keep it for a zero input.

        Args:
            agent:
                The neuron.

            system:
                The system.
        """
        del system
        weights = np.asarray(agent.weights)
        neighbors = [other for other in agent.get_neighbors() if isinstance(other, NeuronView)]
        field = sum(weights[int(np.asarray(other.location))] * other.spin for other in neighbors)
        agent.spin = float(np.sign(field)) if field != 0 else float(agent.spin)


class HopfieldSystem(BaseSystem):
    """A Hopfield network that stores patterns and recalls the first.

    It stores the patterns with Hebb's rule and recalls the first from a corrupted copy. Neurons update one at a time in
    random order, so the energy `-Σ w_ij s_i s_j / 2` never increases and the network settles in a stored pattern when
    few patterns are stored, about fewer than `0.14 n` for `n` neurons. The system has the group `neurons` of states
    `spin` and `weights` on a complete graph.

    Args:
        patterns:
            The stored patterns of -1 and 1, one row per pattern, or `None` for random patterns.

        n_neurons:
            The number of neurons of random patterns.

        n_patterns:
            The number of random patterns.

        noise:
            The probability that a neuron starts flipped from the first pattern.

        random_state:
            The seed or generator of the random patterns and of the corruption.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import HopfieldSystem
        >>> HopfieldSystem(n_neurons=16, n_patterns=2).prepare().system_.elements[0].size
        16
    """

    field = 'computer_science/neural_networks'
    topics = ('neural network', 'associative memory', 'agents', 'attractor')
    references = ('Hopfield 1982', 'Trzaska 2023')
    until = 10

    def __init__(
        self,
        patterns: npt.ArrayLike | None = None,
        *,
        n_neurons: int = 64,
        n_patterns: int = 3,
        noise: float = 0.2,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.patterns = patterns
        self.n_neurons = n_neurons
        self.n_patterns = n_patterns
        self.noise = noise
        self.random_state = random_state

    @staticmethod
    def _get_hebbian_weights(patterns: Array) -> Array:
        """Return the Hebbian weights `w_ij = Σ_μ ξ_i ξ_j / n` that store patterns.

        There is no self-coupling.
        """
        weights = patterns.T @ patterns / patterns.shape[1]
        np.fill_diagonal(weights, 0.0)
        return weights

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        stored: Array = np.asarray(
            rng.choice([-1.0, 1.0], size=(self.n_patterns, self.n_neurons)) if self.patterns is None else self.patterns,
            dtype=np.float64,
        )
        size = stored.shape[1]
        spins = np.where(rng.random(size) < self.noise, -stored[0], stored[0])
        graph = NetworkSpace(nx.complete_graph(size))
        neurons = NeuronElement(
            'neurons',
            spin=spins,
            weights=self._get_hebbian_weights(stored),
            values={'node': np.arange(size)},
            size=size,
            space=graph,
        )
        self.system_ = System(
            self.label,
            elements=[neurons],
            interactions=[HopfieldUpdateInteraction('update', target='neurons')],
            space=graph,
        )
