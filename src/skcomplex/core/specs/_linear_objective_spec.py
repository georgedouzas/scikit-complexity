"""Define the linear objective spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec

SENSES = ('max', 'min')


@dataclass(frozen=True)
class LinearObjectiveSpec(BaseSpec):
    """A spec of the objective of a linear program.

    It is linear in its variables or a sum of their absolute values.

    Args:
        expression:
            The objective, a symbolic expression of the variables of the program.

        sense:
            `'max'` to maximise the objective or `'min'` to minimise it.

    Raises:
        ValidationError:
            The sense is unknown, raised when its owner checks it.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.specs import LinearObjectiveSpec
        >>> LinearObjectiveSpec(2 * var('v'), sense='min').sense
        'min'
    """

    expression: object
    sense: str = 'max'

    def _check(self) -> None:
        """Raise an error for an unknown sense.

        Raises:
            ValidationError:
                The sense is neither `'max'` nor `'min'`.
        """
        if self.sense not in SENSES:
            error_msg = f'Unknown sense {self.sense!r} of an objective. Use one of {", ".join(SENSES)}.'
            raise ValidationError(error_msg)
