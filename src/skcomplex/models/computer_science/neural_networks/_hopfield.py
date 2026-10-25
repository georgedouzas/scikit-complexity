"""Neural network models: Hopfield's associative memory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, cast

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.base import BaseLaw, NumericDomain, VisualSpec, state
from ....core.components import ElementComponent
from ....core.laws import Symbols
from ....core.space import NetworkSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....formalisms.agents import AgentView, ModelView, RuleLaw

type Array = npt.NDArray[np.float64]


def _hebbian_weights(patterns: Array) -> Array:
    """Return the Hebbian weights `w_ij = Σ_μ ξ_i ξ_j / n` that store patterns, with no self-coupling."""
    weights = patterns.T @ patterns / patterns.shape[1]
    np.fill_diagonal(weights, 0.0)
    return weights


class NeuronComponent(ElementComponent):
    """Binary neuron of a Hopfield network, which aligns with the sign of its weighted input when it updates.

    The neuron sits on its node of a complete graph and holds the weights of its incoming links, one per node.

    Args:
        label:
            The label of the agent or group.

        spin:
            The initial state, -1 or 1.

        weights:
            The weights of the links from every node, an array with one value per node.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import NeuronComponent
        >>> NeuronComponent.group('neurons', 4, spin=1.0)
        GroupComponent('neurons', element=NeuronComponent, size=4)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'spin', 'color_range': (-1.0, 1.0)})
    spin: float = state(default=1.0, domain=NumericDomain.INTEGER)
    weights: object = state(default=0.0)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the update rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.update)]

    def update(self, agent: AgentView, model: ModelView) -> None:
        """Take the sign of the weighted input from the other neurons, and keep the state when the input is zero.

        Args:
            agent:
                The neuron.

            model:
                The model.
        """
        weights = np.asarray(agent.state('weights'))
        field = sum(weights[cast('int', other.location)] * other.state('spin') for other in agent.neighbors())
        if field != 0:
            agent.set_state('spin', float(np.sign(field)))


@register_model(
    field='computer_science/neural_networks',
    tags=('neural network', 'associative memory', 'agents', 'attractor'),
    references=('Hopfield 1982', 'Trzaska 2023'),
    defaults={'until': 10},
)
def hopfield(
    patterns: npt.ArrayLike | None = None,
    *,
    n_neurons: int = 64,
    n_patterns: int = 3,
    noise: float = 0.2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a Hopfield network that stores patterns with Hebb's rule and recalls the first from a corrupted copy.

    Neurons update one at a time in random order, so the energy `-Σ w_ij s_i s_j / 2` never increases and the network
    settles in a stored pattern when few patterns are stored, about fewer than `0.14 n` for `n` neurons.

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

    Returns:
        system:
            The model, with the group `neurons` of states `spin` and `weights` on a complete graph.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import hopfield
        >>> hopfield(n_neurons=16, n_patterns=2).elements[0].size
        16
    """
    rng = np.random.default_rng(random_state)
    stored: Array = np.asarray(
        rng.choice([-1.0, 1.0], size=(n_patterns, n_neurons)) if patterns is None else patterns,
        dtype=np.float64,
    )
    size = stored.shape[1]
    spins = np.where(rng.random(size) < noise, -stored[0], stored[0])
    neurons = NeuronComponent.group('neurons', size, spin=spins, weights=_hebbian_weights(stored))
    return System(components=[neurons], space=NetworkSpace(nx.complete_graph(size)))
