"""Stochastic process systems of Markov and hidden Markov sources of symbols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.components.interactions import TransitionInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import StateSpec, VisualSpec
from .types import Matrix

CHAIN = ((0.9, 0.1), (0.4, 0.6))
EMISSION = ((0.7, 0.3), (0.2, 0.8))


class HiddenMarkovSourceElement(Element):
    """Source of symbols emitted from the states of a hidden Markov chain.

    The source emits one symbol per step.

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
        >>> from skcomplex.systems.mathematics.stochastic_processes import HiddenMarkovSourceElement
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


class MarkovSourceElement(Element):
    """Source of symbols `0` to `n - 1`.

    A transition matrix steps the symbol in discrete time.

    Args:
        label:
            The label of the source.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        symbol:
            The initial symbol.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import MarkovSourceElement
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


class MarkovSourceSystem(BaseSystem):
    """A Markov source.

    Each symbol of the sequence depends on the one before it.

    The symbol after `i` is `j` with probability `P_ij`. The entropy rate of the source is `h = -Σ_i π_i Σ_j P_ij log2
    P_ij` bits per symbol, where `π` is the stationary distribution of the chain, and the block entropies of the
    sequence approach it from above. The system has the symbols `source.symbol`.

    Args:
        transition:
            The transition matrix `P`, one row and one column per symbol, with rows that sum to 1.

        symbol:
            The initial symbol.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import MarkovSourceSystem
        >>> MarkovSourceSystem().prepare().system_.interactions[0].state
        'symbol'
    """

    field = 'mathematics/stochastic_processes'
    topics = ('markov chain', 'information theory', 'entropy rate', 'discrete map')
    references = ('Lindgren 2024',)
    until = 300
    dt = 1

    def __init__(
        self,
        transition: Matrix = CHAIN,
        *,
        symbol: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.transition = transition
        self.symbol = symbol

    def _build_system(self) -> CompositeSystem:
        """Return a Markov source.

        Each symbol of the sequence depends on the one before it.
        """
        chain = TransitionInteraction('chain', target='source', state='symbol', matrix=lambda s: self.transition)
        return CompositeSystem(
            self.label,
            elements=[MarkovSourceElement('source', symbol=self.symbol)],
            interactions=[chain],
        )


class HiddenMarkovSourceSystem(BaseSystem):
    """A hidden Markov source.

    Its symbols are emitted at random from the states of a hidden Markov chain.

    The hidden state follows the transition matrix `P`, and in hidden state `i` the source emits symbol `j` with
    probability `E_ij`. The frequency of symbol `j` is `Σ_i π_i E_ij` for the stationary distribution `π` of the hidden
    chain, and the symbols alone are not a Markov chain of any finite order in general. The system has the hidden states
    `source.hidden` and the symbols `source.symbol`.

    Args:
        transition:
            The transition matrix `P` of the hidden chain.

        emission:
            The emission matrix `E`, one row per hidden state and one column per symbol.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import HiddenMarkovSourceSystem
        >>> HiddenMarkovSourceSystem().prepare().system_.interactions[0].observed
        'symbol'
    """

    field = 'mathematics/stochastic_processes'
    topics = ('hidden markov system', 'information theory', 'markov chain', 'discrete map')
    references = ('Lindgren 2024',)
    until = 300
    dt = 1

    def __init__(
        self,
        transition: Matrix = CHAIN,
        emission: Matrix = EMISSION,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.transition = transition
        self.emission = emission

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        chain = TransitionInteraction(
            'chain',
            target='source',
            state='hidden',
            matrix=lambda s: self.transition,
            emission=lambda s: self.emission,
            observed='symbol',
        )
        return CompositeSystem(self.label, elements=[HiddenMarkovSourceElement('source')], interactions=[chain])
