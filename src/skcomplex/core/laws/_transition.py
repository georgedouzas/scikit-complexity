"""Define transition laws, the steps of discrete-time Markov chains."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

from sage.all__sagemath_symbolics import Expression

from ..spaces import DiscreteTime
from ..specs import RequirementSpec
from ._base import BaseLaw


class TransitionLaw(BaseLaw):
    """Step of a discrete-time Markov chain, a state over a finite alphabet drawn from a row of a transition matrix.

    The state's current value selects the row, and the next value is drawn from it. The state takes a categorical
    domain or the integers from 0 to `n - 1`. A transition is a kind of update, so it combines with update laws.

    Args:
        target:
            The symbol of a scalar state.

        matrix:
            The square transition matrix, whose entry `(i, j)` is the probability of the step from `i` to `j`, as
            numbers or expressions of parameters.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import TransitionLaw
        >>> x, q = var('x'), var('q')
        >>> TransitionLaw(x, [[q, 1 - q], [0.4, 0.6]]).check().run({'q': 0.1})[0]
        [0.1, 0.9]
    """

    def __init__(self, target: Expression, matrix: Sequence[Sequence[object]], *, label: str = '') -> None:
        super().__init__(label)
        self.target = target
        self.matrix = matrix

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the transition steps."""
        return (self.target,)

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _check(self) -> Self:
        """Raise an error for a matrix that is not a nonempty sequence of rows of equal length.

        Returns:
            law:
                The law itself.
        """
        self._check_matrix('matrix')
        return self

    def _run(self, inputs: object) -> object:
        """Return the transition matrix at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            matrix:
                The rows of the matrix, numbers when nothing symbolic is left.
        """
        return self._evaluate([list(row) for row in self.matrix], inputs)
