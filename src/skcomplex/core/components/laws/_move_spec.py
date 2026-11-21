"""Define the move spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ...base import BaseSpec
from ...enums import MoveKind
from ...errors import ValidationError


@dataclass(frozen=True)
class MoveSpec(BaseSpec):
    """A spec of the change a Monte Carlo sweep proposes for the value of a site.

    A `MetropolisSolver` with moves builds it for the value of every site of a lattice, such as
    `MoveSpec(s.magnet.spin, 'flip')`, which proposes to turn a spin from 1 to -1, and its `SamplerSpec` keeps or
    rejects the change. A move flips a value, draws one of a list of values, adds a uniform or a normal number, calls a
    function, or flips a whole cluster at once. It flips the spins of a magnet, changes the opinions of voters, and
    turns the letters of a sequence of DNA.

    Read more in the user guide, under Laws.

    Args:
        target:
            The symbol of the value of a lattice that the move changes.

        kind:
            `'flip'` for the negative value, such as a spin of -1 for 1, `'random-value'` for a value drawn from
            `values`, `'rotate'` for the value plus a uniform number in `[-width, width]`, such as the angle of a
            compass needle, `'gaussian'` for the value plus a normal number of standard deviation `width`,
            `'function'` for the values `function` proposes, or `'cluster'` for the flip of a whole cluster of like
            neighbours, grown at random from the change of the cost.

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

    See Also:
        SamplerSpec: Keeps or rejects the moves of a sweep.
        ObjectiveSpec: The cost of one site that a move changes.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import MoveSpec
        >>> MoveSpec(var('color'), 'random-value', values=(0.0, 1.0, 2.0)).values
        (0.0, 1.0, 2.0)
    """

    target: object
    kind: str = 'flip'
    values: tuple[float, ...] = ()
    width: float = 0.0
    function: Callable[..., object] | None = None

    def _check(self) -> None:
        """Raise an error for an unknown move, or a function move without a function.

        Raises:
            ValidationError:
                The kind of move is unknown, or a function move has no callable function.
        """
        if self.kind not in MoveKind:
            error_msg = f'Unknown move {self.kind!r}. Use one of {", ".join(MoveKind)}.'
            raise ValidationError(error_msg)
        if self.kind == 'function' and not callable(self.function):
            error_msg = f'A function move proposes values by a callable function, but got {self.function!r}.'
            raise ValidationError(error_msg)
