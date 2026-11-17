"""Stochastic process systems of Markov and hidden Markov sources of symbols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, Expression

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import categorical, where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from .types import Matrix

CHAIN = ((0.9, 0.1), (0.4, 0.6))
EMISSION = ((0.7, 0.3), (0.2, 0.8))


class HiddenMarkovSourceElement(Element):
    """Source of symbols emitted from the states of a hidden Markov chain.

    Args:
        label:
            The label of the source.

        hidden:
            The initial hidden state.

        symbol:
            The initial symbol.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import HiddenMarkovSourceElement
        >>> HiddenMarkovSourceElement('source', hidden=1).hidden
        1
    """

    def __init__(self, label: str, *, hidden: Annotated[float, NN] = 0, symbol: Annotated[float, NN] = 0) -> None:
        super().__init__(label)
        self.hidden = hidden
        self.symbol = symbol


class MarkovSourceElement(Element):
    """Source of symbols `0` to `n - 1`.

    Args:
        label:
            The label of the source.

        symbol:
            The initial symbol.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import MarkovSourceElement
        >>> MarkovSourceElement('source', symbol=1).symbol
        1
    """

    def __init__(self, label: str, *, symbol: Annotated[float, NN] = 0) -> None:
        super().__init__(label)
        self.symbol = symbol


class MarkovChainInteraction(Interaction):
    """A Markov chain that draws the next value of a discrete state from the row of a transition matrix.

    The value after `i` is `j` with probability `P_ij`. With an emission matrix `E`, a second state, the symbol, is
    drawn in the same step from the row of `E` that the new value selects, as a hidden Markov chain emits its symbols.

    Args:
        label:
            The label of the interaction.

        target:
            The element that holds the state and, with an emission matrix, the symbol.

        transition:
            The transition matrix `P`, one row and one column per value, with rows that sum to 1.

        emission:
            The emission matrix `E`, one row per value of the state and one column per symbol, or `None` for none.

        state:
            The name of the state the chain moves.

        observed:
            The name of the symbol the emissions draw.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import MarkovChainInteraction
        >>> MarkovChainInteraction('chain', target='source', transition=((0.5, 0.5), (0.1, 0.9))).state
        'symbol'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'source',
        transition: Matrix = CHAIN,
        emission: Matrix | None = None,
        state: str = 'symbol',
        observed: str = 'symbol',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.transition = transition
        self.emission = emission
        self.state = state
        self.observed = observed

    @staticmethod
    def _draw_row(name: str, matrix: Matrix, value: Expression) -> Expression:
        """Return the index drawn from the row of a matrix that a value selects."""
        weights = [
            sum(where(value == row, matrix[row][column], 0) for row in range(len(matrix)))
            for column in range(len(matrix[0]))
        ]
        return categorical(name, weights)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the draw of the next value and, with an emission matrix, of the symbol as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            state = getattr(context.target, self.state)
            following = self._draw_row(f'{self.label}_{self.state}', self.transition, state)
            self.laws_: list[BaseComponent] = [UpdateLaw(state, following)]
            if self.emission is not None:
                observed = getattr(context.target, self.observed)
                emitted = self._draw_row(f'{self.label}_{self.observed}', self.emission, following)
                self.laws_.append(UpdateLaw(observed, emitted))


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import MarkovSourceSystem
        >>> MarkovSourceSystem().prepare().simulate(5, random_state=0).get_values('source.symbol').tolist()
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.transition = transition
        self.symbol = symbol

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the source and its chain."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[MarkovSourceElement('source', symbol=self.symbol)],
            interactions=[MarkovChainInteraction('chain', target='source', transition=self.transition)],
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import HiddenMarkovSourceSystem
        >>> HiddenMarkovSourceSystem().prepare().states_
        ('source.hidden', 'source.symbol')
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.transition = transition
        self.emission = emission

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the source and its hidden chain."""
        super()._prepare(context)
        chain = MarkovChainInteraction(
            'chain',
            target='source',
            transition=self.transition,
            emission=self.emission,
            state='hidden',
            observed='symbol',
        )
        self.system_ = System(self.label, elements=[HiddenMarkovSourceElement('source')], interactions=[chain])
