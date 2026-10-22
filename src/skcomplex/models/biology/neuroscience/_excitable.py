"""Excitable networks: cyclic excitable elements that excite their neighbours with a transmission probability."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, state
from ....core.components import Element
from ....core.laws import Law, LocalRule, Symbols
from ....core.runtime import register_model
from ....core.space import Adjacency, DiscreteTime, NetworkSpace
from ....core.system import Observable, System

type Array = npt.NDArray[np.float64]

EXCITED = 1


def _excited_fraction(phases: Array) -> float:
    """Return the fraction of elements that are excited."""
    return float(np.mean(np.asarray(phases) == EXCITED))


class ExcitableElement(Element):
    """Cyclic excitable element of the Kinouchi-Copelli network: quiescent at 0, excited at 1, refractory up to `m - 1`.

    Every step, a quiescent element is excited by the external stimulus with probability `1 - exp(-r)`, where `r` is
    the stimulus rate per step, or by each excited neighbour with the transmission probability `p`, so it stays
    quiescent with probability `exp(-r) (1 - p)^n` when `n` neighbours are excited. An excited or refractory element
    moves to the next state, and the last refractory state returns to quiescence.

    Args:
        label:
            The label of the element or the site group.

        phase:
            The initial state, 0 for quiescent, 1 for excited, and 2 to `n_states - 1` for refractory.

        p:
            The probability that an excited neighbour excites the element in one step.

        stimulus:
            The rate `r` of the external stimulus per step.

        n_states:
            The number of states `m` of the cycle.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import ExcitableElement
        >>> ExcitableElement('neuron').n_states
        5
    """

    phase: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    p: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    stimulus: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    n_states: int = 5

    def laws(self, s: Symbols) -> list[Law]:
        """Return the stochastic rule of excitation by the stimulus and the neighbours, and of the refractory cycle.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        n_states = self.n_states

        def excite(phase: Array, *around: Array, p: Any, stimulus: Any, random: Array) -> Array:  # noqa: ANN401
            excited = sum((item == EXCITED for item in around), np.zeros(np.shape(phase)))
            quiet = np.exp(-stimulus) * (1 - p) ** excited
            fires = np.where(random >= quiet, float(EXCITED), 0.0)
            return np.where(phase == 0, fires, (phase + 1) % n_states)

        return [LocalRule(s.phase, Adjacency(), excite, stochastic=True)]


@register_model(
    field='biology/neuroscience',
    tags=('excitable', 'network', 'criticality', 'dynamic range', 'stochastic'),
    references=('Kinouchi and Copelli 2006', 'Larremore et al. 2011'),
    defaults={'until': 1000, 'dt': 1},
    observables=lambda system: [Observable('activity', group='neurons', variable='phase', reduce=_excited_fraction)],
)
def kinouchi_copelli(
    graph: nx.Graph | None = None,
    *,
    sigma: float = 1.0,
    stimulus: float = 0.0,
    n_states: int = 5,
    excited: float | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Kinouchi-Copelli network of excitable elements, whose dynamic range peaks at the critical point.

    Every element excites each quiescent neighbour with the transmission probability `p = sigma / K`, where `K` is
    the mean degree of the graph, so `sigma` is the branching ratio, the mean number of elements one excitation
    excites. Below `sigma = 1` activity dies out without a stimulus, and above it activity sustains itself. The
    response, the mean firing rate against the stimulus rate, spans the most decades of stimulus at `sigma = 1`.

    Args:
        graph:
            The NetworkX graph, or `None` for an Erdős-Rényi graph of 1000 nodes with mean degree 10.

        sigma:
            The branching ratio.

        stimulus:
            The rate `r` of the external stimulus per step, which excites a quiescent element with probability
            `1 - exp(-r)`.

        n_states:
            The number of states of the cycle, one quiescent, one excited, and `n_states - 2` refractory.

        excited:
            The initial fraction of excited elements, with the others quiescent, or `None` for states drawn uniformly
            over the cycle.

        random_state:
            The seed or generator of the default graph and of the initial states.

    Returns:
        system:
            The model, with the states `neurons.phase`, one value per node. Its standard observable `activity` is the
            fraction of excited elements, the firing rate per step.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.biology.neuroscience import kinouchi_copelli
        >>> kinouchi_copelli(nx.random_regular_graph(4, 20, seed=0), sigma=1.0).elements[0].values['p']
        0.25
    """
    rng = np.random.default_rng(random_state)
    graph = nx.gnp_random_graph(1000, 10 / 999, seed=int(rng.integers(2**31))) if graph is None else graph
    size = graph.number_of_nodes()
    degree = 2 * graph.number_of_edges() / size
    phase = (
        rng.integers(0, n_states, size=size).astype(float)
        if excited is None
        else (rng.random(size) < excited).astype(float)
    )
    neurons = ExcitableElement.lattice(
        'neurons',
        NetworkSpace(graph),
        phase=phase,
        p=sigma / degree,
        stimulus=stimulus,
        n_states=n_states,
    )
    return System(components=[neurons], time=DiscreteTime())
