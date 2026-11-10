"""Define transition laws, the steps and emissions of Markov chains."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import CallSpec, RequirementSpec
from ..spaces import DiscreteTime
from ._base import BaseLaw


class TransitionLaw(BaseLaw):
    """Step of a discrete-time Markov chain.

    The state is over a finite alphabet and drawn from a row of a transition matrix. The state's current value selects
    the row, and the next value is drawn from it. The state takes a categorical domain or the integers from 0 to `n -
    1`. A transition is a kind of update, so it combines with update laws.

    With a source state, the law is the observation of a hidden Markov source. The value of the source state selects the
    row instead, and the law reads it after the same step, so the hidden and observed sequences align. The matrix then
    has one row per value of the source state and one column per value of the target, and it is symbolic only.

    Args:
        target:
            The symbol of a scalar state.

        matrix:
            The square transition matrix, whose entry `(i, j)` is the probability of the step from `i` to `j`, as
            numbers or expressions of parameters, or a function `f(t, values)` that returns it, one matrix for all
            members or one per member, or `CallSpec(f, relative=True)` to read the names of the component. A function
            runs in Python and may read states. With a source, the emission matrix, one row per value of the source
            state and one column per value of the target, as numbers or expressions of parameters.

        source:
            The symbol of the hidden state whose next value selects the row, which an update or transition law of the
            same element sets, or `None` for the target itself.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import TransitionLaw
        >>> x, q = var('x'), var('q')
        >>> TransitionLaw(x, [[q, 1 - q], [0.4, 0.6]]).prepare().run({'q': 0.1}).value_[0]
        [0.1, 0.9]
        >>> observed, hidden = var('observed'), var('hidden')
        >>> TransitionLaw(observed, [[0.7, 0.3], [0.2, 0.8]], source=hidden).prepare().targets_
        (observed,)
    """

    law_tag: ClassVar[str] = 'jump'

    def __init__(
        self,
        target: Expression,
        matrix: Sequence[Sequence[object]] | Callable[..., object] | CallSpec,
        *,
        source: Expression | None = None,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.matrix = matrix
        self.source = source

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the transition steps."""
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for a matrix that is neither a function nor a nonempty matrix.

        It also raises it for a function with a source.

        A nonempty matrix is a sequence of rows of equal length.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        if self.source is not None and self._is_function(self.matrix):
            error_msg = (
                f'The matrix of the transition law of {self.target} with the source {self.source} is a function, but '
                'an emission matrix is symbolic only, because it draws from the next value of the source state. Give a '
                'symbolic matrix, or an UpdateLaw with a `CallSpec`.'
            )
            raise ValidationError(error_msg)
        if not self._is_function(self.matrix):
            self._check_matrix('matrix')

    def _run(self, inputs: object) -> None:
        """Set `value_` to the transition matrix at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        matrix = self.matrix
        if isinstance(matrix, CallSpec) or callable(matrix):
            self.value_ = matrix
            return
        self.value_ = self._evaluate([list(row) for row in matrix], inputs)
