"""Neural network systems: the asymmetric diluted Hopfield network with parallel stochastic updates."""

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
from ....core.components.interactions import ChoiceInteraction
from ....core.components.laws.utils import total
from ....core.components.system import System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


def _overlap(member: SimpleNamespace) -> Expression:
    """Return the agreement of a neuron with the first stored pattern."""
    return member.pattern * member.spin


def _couplings(patterns: np.ndarray, connectivity: float, rng: np.random.Generator) -> scipy.sparse.csr_array:
    """Return the couplings `J_ij = C_ij Σ_μ ξ_i ξ_j` on a random directed graph of mean in-degree `connectivity`."""
    size = patterns.shape[1]
    count = rng.binomial(size * (size - 1), min(connectivity / size, 1.0))
    pairs = np.unique(rng.integers(size, size=(count, 2)), axis=0)
    rows, columns = pairs[pairs[:, 0] != pairs[:, 1]].T
    values = (patterns[:, rows] * patterns[:, columns]).sum(axis=0).astype(float)
    return scipy.sparse.csr_array((values, (rows, columns)), shape=(size, size))


def _alignment(s: SimpleNamespace, option: float, *, couplings: scipy.sparse.csr_array) -> Expression:
    """Return the utility `β s h` of a value `s` of a neuron in its local field `h = Σ_j J_ij s_j`."""
    return s.target.beta * option * total(s.target.spin, weights=couplings)


class DilutedNeuronElement(Element):
    """Binary neuron of an asymmetric diluted Hopfield network, updated in parallel with the others at every step.

    The logit choice of `diluted_hopfield` draws the value `s = ±1` of the neuron with probability
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


@register_system(
    field='computer_science/neural_networks',
    tags=('neural network', 'associative memory', 'spin glass', 'discrete map'),
    references=('Derrida, Gardner, and Zippelius 1987', 'Bertin 2021'),
    defaults={'until': 20, 'dt': 1},
    observables=lambda system: [ObservableSpec('overlap', _overlap, group='neurons')],
)
def diluted_hopfield(
    n_neurons: int = 4000,
    *,
    connectivity: float = 50.0,
    load: float = 0.2,
    temperature_scale: float = 3.0,
    overlap: float = 0.3,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the asymmetric diluted Hopfield network, which recalls a stored pattern below its memory capacity.

    The network stores `p = load K` random patterns in the couplings `J_ij = C_ij Σ_μ ξ_i ξ_j`, where every `C_ij` is
    1 with probability `K / N` independently of `C_ji`, so the couplings are sparse and asymmetric. All neurons update
    together at the inverse temperature `β = temperature_scale / K`. Started with an overlap `m` with the first pattern,
    the overlap follows `m(t + 1) = f(m(t))` for large `K`, which keeps a nonzero fixed point below the critical load
    `2 / π`.

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

    Returns:
        system:
            The system, with the values `neurons.spin` and the observable `overlap`.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import diluted_hopfield
        >>> diluted_hopfield(200, connectivity=10).elements[0].prepare().size_
        200
    """
    rng = np.random.default_rng(random_state)
    n_patterns = max(1, round(load * connectivity))
    patterns = rng.choice([-1.0, 1.0], size=(n_patterns, n_neurons))
    spins = np.where(rng.random(n_neurons) < (1 + overlap) / 2, patterns[0], -patterns[0])
    neurons = GroupElement(
        'neurons',
        element=DilutedNeuronElement,
        size=n_neurons,
        values={
            'spin': spins,
            'pattern': patterns[0],
            'beta': temperature_scale / connectivity,
        },
    )
    glauber = ChoiceInteraction(
        'glauber',
        target='neurons',
        state='spin',
        options=(-1.0, 1.0),
        utility=functools.partial(_alignment, couplings=_couplings(patterns, connectivity, rng)),
    )
    return System(elements=[neurons], interactions=[glauber])
