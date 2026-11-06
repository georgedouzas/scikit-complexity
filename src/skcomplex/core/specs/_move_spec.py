"""Define the move spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec

MOVES = ('flip', 'random-value', 'rotate', 'gaussian', 'function', 'cluster')


@dataclass(frozen=True)
class MoveSpec(BaseSpec):
    """Spec of the proposed change of the value of a lattice site in a Monte Carlo sweep.

    Args:
        target:
            The symbol of the lattice state the move changes.

        kind:
            `'flip'` for the negative value, the spin flip of Ising systems, `'random-value'` for a value drawn from
            `values`, as in Potts systems, `'rotate'` for the value plus a uniform number in `[-width, width]`, as in
            XY systems, `'gaussian'` for the value plus a normal number of standard deviation `width`, `'function'`
            for the values `function` proposes, or `'cluster'` for the flip of a cluster grown by the Wolff bonds of the
            energy.

        values:
            The values a random-value move draws from.

        width:
            The largest change of a rotation, or the standard deviation of a Gaussian move.

        function:
            The function `function(values, random)` of the array of the current values of some sites and the random
            generator that returns their proposed values, for the kind `'function'`.

    Raises:
        ValidationError:
            The kind of move is unknown, or a function move has no callable function.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.specs import MoveSpec
        >>> MoveSpec(var('color'), 'random-value', values=(0.0, 1.0, 2.0)).values
        (0.0, 1.0, 2.0)
    """

    target: object
    kind: str = 'flip'
    values: tuple[float, ...] = ()
    width: float = 0.0
    function: Callable[..., object] | None = None

    def _check(self) -> None:
        """Raise an error for an unknown kind of move, or a function move without a callable function.

        Raises:
            ValidationError:
                The kind of move is unknown, or a function move has no callable function.
        """
        if self.kind not in MOVES:
            error_msg = f'Unknown move {self.kind!r}. Use one of {", ".join(MOVES)}.'
            raise ValidationError(error_msg)
        if self.kind == 'function' and not callable(self.function):
            error_msg = f'A function move proposes values by a callable function, but got {self.function!r}.'
            raise ValidationError(error_msg)
