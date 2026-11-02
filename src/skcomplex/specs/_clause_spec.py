"""Define a clause, a disjunction of literals of the Boolean states of a group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ClauseSpec(BaseSpec):
    """Disjunction of literals, where a literal is a member index from 1, negative for its negation.

    Args:
        literals:
            The literals of the clause.

    Raises:
        ValidationError:
            A literal is 0 or the clause holds none, raised by `check`.

    Examples:
        >>> from skcomplex.specs import ClauseSpec
        >>> ClauseSpec((1, -2)).check().literals
        (1, -2)
    """

    literals: Sequence[int]

    def _check(self) -> None:
        """Raise an error for a clause without literals or with a literal 0.

        Raises:
            ValidationError:
                A literal is 0 or the clause holds none.
        """
        if not len(self.literals) or 0 in self.literals:
            error_msg = f'The clause {tuple(self.literals)} needs literals, member indices from 1, none of them 0.'
            raise ValidationError(error_msg)
