"""Excitable networks: cyclic excitable elements that excite their neighbours with a transmission probability."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import TransitionInteraction
from ....core.components.laws.utils import neighbors, where
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import AnyArray

EXCITED = 1


def _excited_fraction(phases: AnyArray) -> float:
    """Return the fraction of elements that are excited."""
    return float(np.mean(np.asarray(phases) == EXCITED))


class ExcitableElement(Element):
    """Cyclic excitable element of the Kinouchi-Copelli network: quiescent at 0, excited at 1, refractory up to `m - 1`.

    Transitions move its phase every step. A quiescent element is excited by the external stimulus with probability
    `1 - exp(-r)`, where `r` is the stimulus rate per step, or by each excited neighbour with the transmission
    probability `p`, so it stays quiescent with probability `exp(-r) (1 - p)^n` when `n` neighbours are excited. An
    excited or refractory element moves to the next state, and the last refractory state returns to quiescence.

    Args:
        label:
            The label of the element or the site group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        phase:
            The initial state, 0 for quiescent, 1 for excited, and 2 to `n_states - 1` for refractory.

        p:
            The probability that an excited neighbour excites the element in one step.

        stimulus:
            The rate `r` of the external stimulus per step.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import ExcitableElement
        >>> ExcitableElement('neuron').p
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phase: Annotated[float, StateSpec(domain=NN)] = 0,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        stimulus: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phase = phase
        self.p = p
        self.stimulus = stimulus


class KinouchiCopelliSystem(BaseSystem):
    """The Kinouchi-Copelli network of excitable elements, whose dynamic range peaks at the critical point.

    Every element excites each quiescent neighbour with the transmission probability `p = sigma / K`, where `K` is the
    mean degree of the graph, so `sigma` is the branching ratio, the mean number of elements one excitation excites.
    Below `sigma = 1` activity dies out without a stimulus, and above it activity sustains itself. The response, the
    mean firing rate against the stimulus rate, spans the most decades of stimulus at `sigma = 1`. The system has the
    states `neurons.phase`, one value per node. Its standard observable `activity` is the fraction of excited elements,
    the firing rate per step.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.biology.neuroscience import KinouchiCopelliSystem
        >>> system = KinouchiCopelliSystem(nx.random_regular_graph(4, 20, seed=0), sigma=1.0).prepare().system_
        >>> system.elements[0].values['p']
        0.25
    """

    field = 'biology/neuroscience'
    topics = ('excitable', 'network', 'criticality', 'dynamic range', 'stochastic')
    references = ('Kinouchi and Copelli 2006', 'Larremore et al. 2011')
    until = 1000
    dt = 1
    observables = (ObservableSpec('activity', group='neurons', variable='phase', reduce=_excited_fraction),)

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        sigma: float = 1.0,
        stimulus: float = 0.0,
        n_states: int = 5,
        excited: float | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.sigma = sigma
        self.stimulus = stimulus
        self.n_states = n_states
        self.excited = excited
        self.random_state = random_state

    @staticmethod
    def _get_excitation(s: SimpleNamespace) -> Expression:
        """Return the probability that the stimulus or an excited neighbour excites a quiescent element in a step."""
        excited = neighbors(where(s.target.other.phase == EXCITED, 1, 0))
        probability: Expression = 1 - exp(-s.target.stimulus) * (1 - s.target.p) ** excited
        return probability

    @staticmethod
    def _get_cycle(n_states: int) -> list[TransitionInteraction]:
        """Return the moves of an excited or refractory element to the next state of its cycle, the last to rest."""
        return [
            TransitionInteraction(
                f'cycle_{phase}',
                target='neurons',
                state='phase',
                source=phase,
                destination=(phase + 1) % n_states,
                condition=lambda s: 1,
            )
            for phase in range(EXCITED, n_states)
        ]

    def _build_system(self) -> System:
        """Return the parts of the Kinouchi-Copelli network of excitable elements."""
        graph = self.graph
        rng = np.random.default_rng(self.random_state)
        graph = nx.gnp_random_graph(1000, 10 / 999, seed=int(rng.integers(2**31))) if graph is None else graph
        size = graph.number_of_nodes()
        degree = 2 * graph.number_of_edges() / size
        phase = (
            rng.integers(0, self.n_states, size=size).astype(float)
            if self.excited is None
            else (rng.random(size) < self.excited).astype(float)
        )
        neurons = LatticeElement(
            'neurons',
            element=ExcitableElement,
            space=NetworkSpace(graph),
            values={'phase': phase, 'p': self.sigma / degree, 'stimulus': self.stimulus},
        )
        excitation = TransitionInteraction(
            'excitation',
            target='neurons',
            state='phase',
            source=0,
            destination=EXCITED,
            probability=self._get_excitation,
        )
        return System(
            self.label,
            elements=[neurons],
            interactions=[excitation, *self._get_cycle(self.n_states)],
            time=DiscreteTime(),
        )
