"""Neural network models: the asymmetric diluted Hopfield network with parallel stochastic updates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Element, Law, Observable, Symbols, System, Update, total, uniform, where
from ....core.runtime import register_model


def _overlap(member: Symbols) -> Expression:
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


class DilutedNeuron(Element):
    """Binary neuron of an asymmetric diluted Hopfield network, updated in parallel with the others at every step.

    The neuron takes the value `s = ±1` with probability `1 / (1 + exp(-2 s β h))`, where `h = Σ_j J_ij s_j` is its
    local field, so its mean is `tanh(β h)`.

    Args:
        label:
            The label of the neuron or the group.

        spin:
            The initial value, -1 or 1.

        pattern:
            The value of the first stored pattern at the neuron, whose overlap with the network is recorded.

        beta:
            The inverse temperature `β`.

        couplings:
            The coupling matrix `J`, one row and one column per neuron.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import DilutedNeuron
        >>> DilutedNeuron('neuron').beta
        0.06
    """

    spin: float = state(default=1.0, domain=Domain.INTEGER)
    pattern: float = parameter(default=1.0)
    beta: float = parameter(default=0.06, domain=Domain.NONNEGATIVE)
    couplings: Any = None

    def laws(self, s: Symbols) -> list[Law]:
        """Return the parallel stochastic update of the neuron.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        field = total(s.spin, weights=self.couplings)
        up = 1 / (1 + exp(-2 * s.beta * field))
        return [Update(s.spin, where(uniform('u') < up, 1, -1))]


@register_model(
    field='computer_science/neural_networks',
    tags=('neural network', 'associative memory', 'spin glass', 'discrete map'),
    references=('Derrida, Gardner, and Zippelius 1987', 'Bertin 2021'),
    defaults={'until': 20, 'dt': 1},
    observables=lambda system: [Observable('overlap', _overlap, group='neurons')],
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
            The model, with the values `neurons.spin` and the observable `overlap`.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import diluted_hopfield
        >>> diluted_hopfield(200, connectivity=10).elements[0].size
        200
    """
    rng = np.random.default_rng(random_state)
    n_patterns = max(1, round(load * connectivity))
    patterns = rng.choice([-1.0, 1.0], size=(n_patterns, n_neurons))
    spins = np.where(rng.random(n_neurons) < (1 + overlap) / 2, patterns[0], -patterns[0])
    neurons = DilutedNeuron.group(
        'neurons',
        n_neurons,
        spin=spins,
        pattern=patterns[0],
        beta=temperature_scale / connectivity,
        couplings=_couplings(patterns, connectivity, rng),
    )
    return System(components=[neurons])
