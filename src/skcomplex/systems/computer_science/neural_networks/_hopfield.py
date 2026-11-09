"""Neural network systems: Hopfield's associative memory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.spaces import NetworkSpace
from ....core.components.systems import BaseSystem, System
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import RoleSpec, StateSpec, VisualSpec
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

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        spin:
            The initial state, -1 or 1.

        weights:
            The weights of the links from every node, an array with one value per node.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import NeuronElement
        >>> group = GroupElement('neurons', element=NeuronElement, size=4, values={'spin': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('neurons', 'NeuronElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'spin', 'color_range': (-1.0, 1.0)}),
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        weights: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.spin = spin
        self.weights = weights

    @classmethod
    def _get_view(cls) -> type[NeuronView]:
        """Return the view of the neurons.

        It types their states.
        """
        return NeuronView


class HopfieldUpdateInteraction(MapInteraction):
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
        >>> HopfieldUpdateInteraction('update', target='neurons').schedule
        'random-sequential'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            state='spin',
            function=self._get_spin,
            schedule='random-sequential',
            form='agent',
        )

    def _get_spin(self, agent: NeuronView, system: SystemViewProtocol) -> float:
        """Return the sign of the weighted input of a neuron from the other neurons.

        A zero input returns its state.
        """
        del system
        weights = np.asarray(agent.weights)
        neighbors = [other for other in agent.neighbors() if isinstance(other, NeuronView)]
        field = sum(weights[int(np.asarray(other.location))] * other.spin for other in neighbors)
        return float(np.sign(field)) if field != 0 else float(agent.spin)


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

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import HopfieldSystem
        >>> HopfieldSystem(n_neurons=16, n_patterns=2).prepare().system_.elements[0].prepare().size_
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
    ) -> None:
        super().__init__(label)
        self.patterns = patterns
        self.n_neurons = n_neurons
        self.n_patterns = n_patterns
        self.noise = noise
        self.random_state = random_state

    @staticmethod
    def _hebbian_weights(patterns: Array) -> Array:
        """Return the Hebbian weights `w_ij = Σ_μ ξ_i ξ_j / n` that store patterns.

        There is no self-coupling.
        """
        weights = patterns.T @ patterns / patterns.shape[1]
        np.fill_diagonal(weights, 0.0)
        return weights

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        stored: Array = np.asarray(
            rng.choice([-1.0, 1.0], size=(self.n_patterns, self.n_neurons)) if self.patterns is None else self.patterns,
            dtype=np.float64,
        )
        size = stored.shape[1]
        spins = np.where(rng.random(size) < self.noise, -stored[0], stored[0])
        neurons = GroupElement(
            'neurons',
            element=NeuronElement,
            size=size,
            values={'spin': spins, 'weights': self._hebbian_weights(stored)},
        )
        return System(
            self.label,
            elements=[neurons],
            interactions=[HopfieldUpdateInteraction('update', target='neurons')],
            space=NetworkSpace(nx.complete_graph(size)),
        )
