"""Neural network systems: the asymmetric diluted Hopfield network.

Its updates are parallel and stochastic.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import ZZ, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws.utils import total
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.agents import ChoiceInteraction


class DilutedNeuronElement(Element):
    """Binary neuron of an asymmetric diluted Hopfield network.

    It is updated in parallel with the others at every step.

    The logit choice of `DilutedHopfieldSystem` draws the value `s = ±1` of the neuron with probability
    `1 / (1 + exp(-2 s β h))`, where `h = Σ_j J_ij s_j` is its local field, so its mean is `tanh(β h)`.

    Args:
        label:
            The label of the neuron or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        spin:
            The initial value, -1 or 1.

        pattern:
            The value of the first stored pattern at the neuron, whose overlap with the network is recorded.

        beta:
            The inverse temperature `β`.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import DilutedNeuronElement
        >>> DilutedNeuronElement('neuron').beta
        0.06
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        pattern: Annotated[float, ParameterSpec()] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.06,
    ) -> None:
        super().__init__(label, visual=visual)
        self.spin = spin
        self.pattern = pattern
        self.beta = beta


def _get_overlap(member: SimpleNamespace) -> Expression:
    """Return the agreement of a neuron with the first stored pattern."""
    return member.pattern * member.spin


class DilutedHopfieldSystem(BaseSystem):
    """The asymmetric diluted Hopfield network.

    It recalls a stored pattern below its memory capacity. The network stores `p = load K` random patterns in the
    couplings `J_ij = C_ij Σ_μ ξ_i ξ_j`, where every `C_ij` is 1 with probability `K / N` independently of `C_ji`, so
    the couplings are sparse and asymmetric. All neurons update together at the inverse temperature
    `β = temperature_scale / K`. Started with an overlap `m` with the first pattern, the overlap follows
    `m(t + 1) = f(m(t))` for large `K`, which keeps a nonzero fixed point below the critical load `2 / π`. The system
    has the values `neurons.spin` and the observable `overlap`.

    Args:
        n_neurons:
            The number of neurons `N`.

        connectivity:
            The mean number `K` of inputs of a neuron.

        load:
            The ratio `p / K` of the number of patterns to the connectivity.

        temperature_scale:
            The product `β K` of the inverse temperature and the connectivity.

        overlap:
            The initial overlap with the first pattern.

        random_state:
            The seed or generator of the patterns, the dilution, and the initial values.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import DilutedHopfieldSystem
        >>> DilutedHopfieldSystem(200, connectivity=10).prepare().system_.elements[0].prepare().size_
        200
    """

    field = 'computer_science/neural_networks'
    topics = ('neural network', 'associative memory', 'spin glass', 'discrete map')
    references = ('Derrida, Gardner, and Zippelius 1987', 'Bertin 2021')
    until = 20
    dt = 1
    observables = (ObservableSpec('overlap', _get_overlap, group='neurons'),)

    def __init__(
        self,
        n_neurons: int = 4000,
        *,
        connectivity: float = 50.0,
        load: float = 0.2,
        temperature_scale: float = 3.0,
        overlap: float = 0.3,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_neurons = n_neurons
        self.connectivity = connectivity
        self.load = load
        self.temperature_scale = temperature_scale
        self.overlap = overlap
        self.random_state = random_state

    @staticmethod
    def _couplings(patterns: np.ndarray, connectivity: float, rng: np.random.Generator) -> scipy.sparse.csr_array:
        """Return the couplings `J_ij = C_ij Σ_μ ξ_i ξ_j` on a random digraph.

        The digraph has the mean in-degree `connectivity`.
        """
        size = patterns.shape[1]
        count = rng.binomial(size * (size - 1), min(connectivity / size, 1.0))
        pairs = np.unique(rng.integers(size, size=(count, 2)), axis=0)
        rows, columns = pairs[pairs[:, 0] != pairs[:, 1]].T
        values = (patterns[:, rows] * patterns[:, columns]).sum(axis=0).astype(float)
        return scipy.sparse.csr_array((values, (rows, columns)), shape=(size, size))

    @staticmethod
    def _alignment(s: SimpleNamespace, option: float, *, couplings: scipy.sparse.csr_array) -> Expression:
        """Return the utility `β s h` of a value `s` of a neuron in its local field.

        The local field is `h = Σ_j J_ij s_j`.
        """
        return s.target.beta * option * total(s.target.spin, weights=couplings)

    def _build_system(self) -> CompositeSystem:
        """Return the asymmetric diluted Hopfield network.

        It recalls a stored pattern below its memory capacity.
        """
        rng = np.random.default_rng(self.random_state)
        n_patterns = max(1, round(self.load * self.connectivity))
        patterns = rng.choice([-1.0, 1.0], size=(n_patterns, self.n_neurons))
        spins = np.where(rng.random(self.n_neurons) < (1 + self.overlap) / 2, patterns[0], -patterns[0])
        neurons = GroupElement(
            'neurons',
            element=DilutedNeuronElement,
            size=self.n_neurons,
            values={
                'spin': spins,
                'pattern': patterns[0],
                'beta': self.temperature_scale / self.connectivity,
            },
        )
        glauber = ChoiceInteraction(
            'glauber',
            target='neurons',
            state='spin',
            options=(-1.0, 1.0),
            utility=functools.partial(self._alignment, couplings=self._couplings(patterns, self.connectivity, rng)),
        )
        return CompositeSystem(self.label, elements=[neurons], interactions=[glauber])
