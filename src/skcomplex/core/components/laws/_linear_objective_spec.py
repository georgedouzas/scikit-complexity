"""Define the linear objective spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError

SENSES = ('max', 'min')


@dataclass(frozen=True)
class LinearObjectiveSpec(BaseSpec):
    """A spec of the cost of a linear program, to make least or greatest.

    The compiler builds it from an `ObjectiveLaw` whose value is linear in its targets, such as
    `LinearObjectiveSpec(-v_growth, sense='min')` for the greatest growth of a cell, and a linear program makes its
    expression least or greatest under the constraints of the system. The expression is linear in the variables of
    the program or a sum of their absolute values. It finds the fluxes of a cell at its greatest growth, the cheapest
    dispatch of power plants, and the plan of a factory of greatest profit.

    Read more in the user guide, under Laws.

    Args:
        expression:
            The objective, a symbolic expression of the variables of the program.

        sense:
            `'max'` to maximise the objective or `'min'` to minimise it.

    Raises:
        ValidationError:
            The sense is unknown, raised when its owner checks it.

    See Also:
        ObjectiveLaw: Gives a cost of values, which the system makes least.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import LinearObjectiveSpec
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
