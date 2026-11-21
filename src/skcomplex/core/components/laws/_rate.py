"""Define the law that adds to the highest time derivative of a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression
from sage.structure.element import Matrix

from ..spaces._continuous_time import _ContinuousTime
from ..types import Target
from ._base import BaseLaw
from ._requirement_spec import RequirementSpec


@dataclass(frozen=True)
class RateLaw(BaseLaw):
    """A law that adds a rate to a time derivative of a value.

    It says how fast a value changes in continuous time, such as `RateLaw(s.ball.x.diff(s.t, 2), -s.g)` inside the
    interaction of a falling ball, which gives the height `x` the acceleration `-g`. A rate on `x` is its first
    derivative, and a rate on `x.diff(t, k)` is its derivative of order `k`. Rates on the same value add up, so each
    cause of a change writes its own law, and a system of rate laws alone is a system of ordinary differential
    equations. A flow moves an amount between two stocks and maps each stock to the sign of its change, such as
    `RateLaw({s.alive: -1, s.dead: +1}, f, label='deaths')`, where `f` leaves the living and enters the dead. Infected
    people recover at a rate, a balance grows by interest, a reactant is used up by a reaction, and a pendulum swings
    under gravity.

    Read more in the user guide, under Laws.

    Args:
        target:
            The symbol of a scalar value, a derivative of it such as `x.diff(t, 2)`, the column of symbols of a vector
            value, the matrix of a tensor value, or, for a flow, a mapping from the symbol of each stock to the sign of
            its change, `-1` or `+1`. A flow from outside the system names only the stock it enters, such as
            `{s.water: +1}`.

        value:
            A symbolic expression, a matrix of the shape of a tensor value, a `CallSpec`, or a function
            `f(t, values)`, where `values` maps the dotted name of every value and parameter to its current value.

        label:
            The name of a flow, which the trajectory records under the element of its stocks, such as
            `population.deaths`, the name of the rate in messages, or the empty string for none.

    See Also:
        NoiseLaw: Adds a random change to a derivative of a value.
        UpdateLaw: Gives the next value after one step of discrete time.
        ConstraintLaw: States an equation that the compiler solves for the derivatives.

    Examples:
        >>> from sage.all__sagemath_symbolics import function, var
        >>> from skcomplex.core.components.laws import RateLaw
        >>> t, g = var('t g')
        >>> x = function('x')(t)
        >>> RateLaw(x.diff(t, 2), -g).value
        -g
    """

    law_tag: ClassVar[str] = 'rate'

    target: Target | Matrix | Mapping[Expression, int]
    value: object
    _: KW_ONLY
    label: str = ''

    def _get_order(self) -> float | None:
        """Return the order of the derivative the rate gives, `None` for its state."""
        return None

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the rate changes, or the stocks of a flow."""
        if isinstance(self.target, Mapping):
            return tuple(self.target)
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(_ContinuousTime,))

    def _check(self) -> None:
        """Raise an error for a value that is a NumPy array."""
        super()._check()
        self._check_array('value')

    def _get_value(self, inputs: object) -> object:
        """Return the rate at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        return self._evaluate(self.value, inputs)
