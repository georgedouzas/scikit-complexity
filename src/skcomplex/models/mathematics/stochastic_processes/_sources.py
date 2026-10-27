"""Stochastic process models: Markov and hidden Markov sources of symbols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....core.laws import BaseLaw, EmissionLaw, TransitionLaw
from ....core.specs import StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model

type Matrix = Sequence[Sequence[float]]

CHAIN = ((0.9, 0.1), (0.4, 0.6))
EMISSION = ((0.7, 0.3), (0.2, 0.8))


class MarkovSourceElement(Element):
    """Source of symbols `0` to `n - 1` that follow a discrete-time Markov chain.

    Args:
        label:
            The label of the source.

        transition:
            The transition matrix, whose entry `(i, j)` is the probability that symbol `j` follows symbol `i`.

        **values:
            The values of the declared variables, by name:

            symbol:
                The initial symbol.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import MarkovSourceElement
        >>> MarkovSourceElement('source').transition
        ((0.9, 0.1), (0.4, 0.6))
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        symbol: Annotated[float, StateSpec(domain=NN)] = 0,
        transition: Matrix = CHAIN,
    ) -> None:
        super().__init__(label, visual=visual)
        self.symbol = symbol
        self.transition = transition

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the step of the chain.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [TransitionLaw(s.symbol, self.transition)]


class HiddenMarkovSourceElement(Element):
    """Source of symbols emitted from the states of a hidden Markov chain, one symbol per step.

    Args:
        label:
            The label of the source.

        transition:
            The transition matrix of the hidden chain.

        emission:
            The emission matrix, whose entry `(i, j)` is the probability that hidden state `i` emits symbol `j`.

        **values:
            The values of the declared variables, by name:

            hidden:
                The initial hidden state.

            symbol:
                The initial symbol.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import HiddenMarkovSourceElement
        >>> HiddenMarkovSourceElement('source').emission
        ((0.7, 0.3), (0.2, 0.8))
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        hidden: Annotated[float, StateSpec(domain=NN)] = 0,
        symbol: Annotated[float, StateSpec(domain=NN)] = 0,
        transition: Matrix = CHAIN,
        emission: Matrix = EMISSION,
    ) -> None:
        super().__init__(label, visual=visual)
        self.hidden = hidden
        self.symbol = symbol
        self.transition = transition
        self.emission = emission

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the step of the hidden chain and the emission of the symbol.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [TransitionLaw(s.hidden, self.transition), EmissionLaw(s.symbol, s.hidden, self.emission)]


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
    return System(elements=[MarkovSourceElement('source', symbol=symbol, transition=transition)])


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
    return System(elements=[HiddenMarkovSourceElement('source', transition=transition, emission=emission)])
