"""Define emission laws, the observations of hidden Markov sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import DiscreteTime
from ._base import BaseLaw


class EmissionLaw(BaseLaw):
    """Observation of a hidden Markov source.

    The observation is a state drawn from the row of an emission matrix a hidden state selects. The emission reads the
    hidden state's value after the same step, so the hidden and observed sequences align.

    Args:
        target:
            The symbol of the observed scalar state.

        source:
            The symbol of the hidden state, which an update or transition law of the same element sets.

        matrix:
            The emission matrix, one row per value of the hidden state and one column per value of the observed state,
            as numbers or expressions of parameters.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import EmissionLaw
        >>> observed, hidden = var('observed'), var('hidden')
        >>> EmissionLaw(observed, hidden, [[0.7, 0.3], [0.2, 0.8]]).prepare().targets_
        (observed,)
    """

    law_tag: ClassVar[str] = 'jump'

    def __init__(
        self,
        target: Expression,
        source: Expression,
        matrix: Sequence[Sequence[object]],
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.matrix = matrix

    def _get_targets(self) -> tuple[object, ...]:
        """Return the observed state."""
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _check(self) -> None:
        """Raise an error for a matrix that is not a nonempty sequence of equal rows.

        The rows must be of equal length.
        """
        if self._is_function(self.matrix):
            error_msg = (
                'The matrix of an EmissionLaw is a function, but an emission matrix is symbolic only, because it draws '
                'from the next value of the hidden state. Use a TransitionLaw, or an UpdateLaw with a `CallSpec`.'
            )
            raise ValidationError(error_msg)
        self._check_matrix('matrix')

    def _run(self, inputs: object) -> None:
        """Set `value_` to the emission matrix at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate([list(row) for row in self.matrix], inputs)
