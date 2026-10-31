"""Stochastic process models: Markov and hidden Markov sources of symbols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....core.interactions import TransitionInteraction
from ....core.system import System
from ....specs import StateSpec, VisualSpec
from ..._catalog import register_model
from .types import Matrix

CHAIN = ((0.9, 0.1), (0.4, 0.6))
EMISSION = ((0.7, 0.3), (0.2, 0.8))


class MarkovSourceElement(Element):
    """Source of symbols `0` to `n - 1`, whose symbol a transition matrix steps in discrete time.

    Args:
        label:
            The label of the source.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        symbol:
            The initial symbol.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import MarkovSourceElement
        >>> MarkovSourceElement('source', symbol=1).symbol
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        symbol: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.symbol = symbol


class HiddenMarkovSourceElement(Element):
    """Source of symbols emitted from the states of a hidden Markov chain, one symbol per step.

    Args:
        label:
            The label of the source.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        hidden:
            The initial hidden state.

        symbol:
            The initial symbol.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import HiddenMarkovSourceElement
        >>> HiddenMarkovSourceElement('source', hidden=1).hidden
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        hidden: Annotated[float, StateSpec(domain=NN)] = 0,
        symbol: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.hidden = hidden
        self.symbol = symbol


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
        >>> markov_source().interactions[0].state
        'symbol'
    """
    chain = TransitionInteraction('chain', target='source', state='symbol', matrix=lambda s: transition)
    return System(elements=[MarkovSourceElement('source', symbol=symbol)], interactions=[chain])


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
        >>> hidden_markov_source().interactions[0].observed
        'symbol'
    """
    chain = TransitionInteraction(
        'chain',
        target='source',
        state='hidden',
        matrix=lambda s: transition,
        emission=lambda s: emission,
        observed='symbol',
    )
    return System(elements=[HiddenMarkovSourceElement('source')], interactions=[chain])
