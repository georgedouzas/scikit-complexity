"""Stochastic process models: Markov and hidden Markov sources of symbols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.base import Domain, state
from ....core.model import Element, Emission, Law, Symbols, System, Transition
from ....core.runtime import register_model

type Matrix = Sequence[Sequence[float]]

CHAIN = ((0.9, 0.1), (0.4, 0.6))
EMISSION = ((0.7, 0.3), (0.2, 0.8))


class MarkovSource(Element):
    """Source of symbols `0` to `n - 1` that follow a discrete-time Markov chain.

    Args:
        label:
            The label of the source.

        symbol:
            The initial symbol.

        transition:
            The transition matrix, whose entry `(i, j)` is the probability that symbol `j` follows symbol `i`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import MarkovSource
        >>> MarkovSource('source').transition
        ((0.9, 0.1), (0.4, 0.6))
    """

    symbol: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    transition: Matrix = CHAIN

    def laws(self, s: Symbols) -> list[Law]:
        """Return the step of the chain.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Transition(s.symbol, self.transition)]


class HiddenMarkovSource(Element):
    """Source of symbols emitted from the states of a hidden Markov chain, one symbol per step.

    Args:
        label:
            The label of the source.

        hidden:
            The initial hidden state.

        symbol:
            The initial symbol.

        transition:
            The transition matrix of the hidden chain.

        emission:
            The emission matrix, whose entry `(i, j)` is the probability that hidden state `i` emits symbol `j`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import HiddenMarkovSource
        >>> HiddenMarkovSource('source').emission
        ((0.7, 0.3), (0.2, 0.8))
    """

    hidden: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    symbol: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    transition: Matrix = CHAIN
    emission: Matrix = EMISSION

    def laws(self, s: Symbols) -> list[Law]:
        """Return the step of the hidden chain and the emission of the symbol.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Transition(s.hidden, self.transition), Emission(s.symbol, s.hidden, self.emission)]


@register_model(
    field='mathematics/stochastic_processes',
    tags=('markov chain', 'information theory', 'entropy rate', 'discrete map'),
    references=('Lindgren 2024',),
    defaults={'until': 300, 'dt': 1},
)
def markov_source(transition: Matrix = CHAIN, *, symbol: int = 0) -> System:
    """Return a Markov source, a sequence of symbols in which each symbol depends on the one before it.

    The symbol after `i` is `j` with probability `P_ij`. The entropy rate of the source is
    `h = -Σ_i π_i Σ_j P_ij log2 P_ij` bits per symbol, where `π` is the stationary distribution of the chain, and the
    block entropies of the sequence approach it from above.

    Args:
        transition:
            The transition matrix `P`, one row and one column per symbol, with rows that sum to 1.

        symbol:
            The initial symbol.

    Returns:
        system:
            The model, with the symbols `source.symbol`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import markov_source
        >>> markov_source().elements[0].transition
        ((0.9, 0.1), (0.4, 0.6))
    """
    return System(components=[MarkovSource('source', symbol=symbol, transition=transition)])


@register_model(
    field='mathematics/stochastic_processes',
    tags=('hidden markov model', 'information theory', 'markov chain', 'discrete map'),
    references=('Lindgren 2024',),
    defaults={'until': 300, 'dt': 1},
)
def hidden_markov_source(transition: Matrix = CHAIN, emission: Matrix = EMISSION) -> System:
    """Return a hidden Markov source, whose symbols are emitted at random from the states of a hidden Markov chain.

    The hidden state follows the transition matrix `P`, and in hidden state `i` the source emits symbol `j` with
    probability `E_ij`. The frequency of symbol `j` is `Σ_i π_i E_ij` for the stationary distribution `π` of the hidden
    chain, and the symbols alone are not a Markov chain of any finite order in general.

    Args:
        transition:
            The transition matrix `P` of the hidden chain.

        emission:
            The emission matrix `E`, one row per hidden state and one column per symbol.

    Returns:
        system:
            The model, with the hidden states `source.hidden` and the symbols `source.symbol`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import hidden_markov_source
        >>> hidden_markov_source().elements[0].transition
        ((0.9, 0.1), (0.4, 0.6))
    """
    return System(components=[HiddenMarkovSource('source', transition=transition, emission=emission)])
