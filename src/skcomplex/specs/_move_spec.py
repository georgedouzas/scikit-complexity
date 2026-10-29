"""Define the move spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec

MOVES = ('flip', 'random-value', 'rotate')


@dataclass(frozen=True)
class MoveSpec(BaseSpec):
    """Spec of the proposed change of the value of a lattice site in a Monte Carlo sweep.

    Args:
        target:
            The symbol of the lattice state the move changes.

        kind:
            `'flip'` for the negative value, the spin flip of Ising models, `'random-value'` for a value drawn from
            `values`, as in Potts models, or `'rotate'` for the value plus a uniform number in `[-width, width]`, as in
            XY models.

        values:
            The values a random-value move draws from.

        width:
            The largest change of a rotation.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.specs import MoveSpec
        >>> MoveSpec(var('color'), 'random-value', values=(0.0, 1.0, 2.0)).check().values
        (0.0, 1.0, 2.0)
    """

    target: object
    kind: str = 'flip'
    values: tuple[float, ...] = ()
    width: float = 0.0

    def _check(self) -> None:
        """Raise an error for an unknown kind of move.

        Raises:
            ValidationError:
                The kind of move is unknown.
        """
        if self.kind not in MOVES:
            error_msg = f'Unknown move {self.kind!r}. Use one of {", ".join(MOVES)}.'
            raise ValidationError(error_msg)
