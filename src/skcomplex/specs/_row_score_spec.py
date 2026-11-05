"""Define the row score spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class RowScoreSpec(BaseSpec):
    """Spec of an objective that scores a whole row of a lattice state, the values along its last axis.

    Args:
        function:
            The function `function(row, **parameters)` of the values of a row and the parameters of the system that it
            names, which returns one number.

        proposals:
            The number of moves proposed per row and unit of time.

    Raises:
        ValidationError:
            The function is not callable, or the number of proposals is not positive.

    Examples:
        >>> from skcomplex.specs import RowScoreSpec
        >>> RowScoreSpec(sum, proposals=2).proposals
        2
    """

    function: Callable[..., float]
    proposals: int = 1

    def _check(self) -> None:
        """Raise an error for a function that is not callable, or a number of proposals that is not positive.

        Raises:
            ValidationError:
                The function is not callable, or the number of proposals is not positive.
        """
        if not callable(self.function):
            error_msg = f'The function of a row score must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
        if self.proposals < 1:
            error_msg = f'A row score proposes at least one move per row, but it has proposals={self.proposals}.'
            raise ValidationError(error_msg)
