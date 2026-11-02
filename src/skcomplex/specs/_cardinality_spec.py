"""Define a cardinality constraint, a bound on how many literals of the Boolean states of a group hold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec

SENSES = ('at-least', 'at-most')


@dataclass(frozen=True)
class CardinalitySpec(BaseSpec):
    """Constraint that at least or at most a bound of its literals hold.

    A literal is a member index from 1, negative for its negation.

    Args:
        literals:
            The literals of the constraint.

        bound:
            The number of literals that bounds those that hold.

        sense:
            `'at-least'` or `'at-most'`.

    Raises:
        ValidationError:
            A literal is 0, the bound lies outside the number of literals, or the sense is unknown, raised by `check`.

    Examples:
        >>> from skcomplex.specs import CardinalitySpec
        >>> CardinalitySpec((1, -2, 3), 2).check().sense
        'at-least'
    """

    literals: Sequence[int]
    bound: int
    sense: str = 'at-least'

    def _check(self) -> None:
        """Raise an error for a literal 0, a bound outside the number of literals, or an unknown sense.

        Raises:
            ValidationError:
                A literal is 0, the bound lies outside the number of literals, or the sense is unknown.
        """
        if self.sense not in SENSES or 0 in self.literals or not 0 <= self.bound <= len(self.literals):
            error_msg = (
                f'The cardinality constraint of {tuple(self.literals)} has bound {self.bound} and sense '
                f'{self.sense!r}. Use literals that are not 0, a bound up to their number, and a sense of '
                f'{", ".join(SENSES)}.'
            )
            raise ValidationError(error_msg)
