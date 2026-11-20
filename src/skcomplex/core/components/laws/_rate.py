"""Define the law that adds to the highest time derivative of a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression
from sage.structure.element import Matrix

from ..spaces._continuous_time import _ContinuousTime
from ..types import Target
from ._base import BaseLaw
from ._requirement_spec import RequirementSpec


class RateLaw(BaseLaw):
    """A law that adds a rate to the highest time derivative of a state.

    On a state of order 1 the rate is its first derivative, and on a state of order `k` it adds to the `k`th derivative.
    Rates on the same state add up, so separate mechanisms superpose, and a system with only rate laws is a system of
    ordinary differential equations.

    A flow moves an amount between stocks, such as people from the living to the dead. Its target maps each stock to
    the sign of its change, `RateLaw({s.alive: -1, s.dead: +1}, f, label='deaths')`, so the amount `f` leaves the
    first stock and enters the second, and the trajectory records `f` under the label. A flow from outside the system
    names only the stock it enters, `RateLaw({s.water: +1}, f, label='tap')`, and a flow out of it only the stock it
    leaves.

    Unlike `UpdateLaw`, which gives the next value in discrete time, it gives a derivative in continuous time. Choose it
    when a state flows, and add a `NoiseLaw` for a random part.

    It appears in many fields. Infected people recover at a rate, a balance grows by interest, a reactant is consumed by
    mass action, and a pendulum accelerates under gravity.

    Args:
        target:
            The symbol of a scalar state, the column of symbols of a vector state, the matrix of a tensor state, or, for
            a flow, a mapping from the symbol of each stock to the sign of its change, `-1` or `+1`.

        value:
            A symbolic expression, a matrix of the shape of a tensor state, a `Call`, or a callable `f(t, values)` where
            `values` maps dotted state and parameter names to their current values.

        order:
            The order of the derivative the rate gives, checked against the state's order, or `None` to use it.

        label:
            The name of a flow, which the trajectory records under the element of its stocks, such as
            `population.deaths`, or the name of the rate in messages, or the empty string for none.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import RateLaw
        >>> n = var('n')
        >>> RateLaw(n, 2 * n).prepare().targets_
        (n,)
        >>> alive, dead = var('alive dead')
        >>> RateLaw({alive: -1, dead: +1}, 0.1 * alive, label='deaths').prepare().targets_
        (alive, dead)
    """

    law_tag: ClassVar[str] = 'rate'

    def __init__(
        self,
        target: Target | Matrix | Mapping[Expression, int],
        value: object,
        order: int | None = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.value = value
        self.order = order

    def _get_targets(self) -> tuple[object, ...]:
        """Return the state the rate changes, or the stocks of a flow."""
        if isinstance(self.target, Mapping):
            return tuple(self.target)
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: continuous time."""
        return RequirementSpec(time=(_ContinuousTime,))

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for a value that is a NumPy array.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        self._check_array('value')

    def _get_value(self, inputs: object) -> object:
        """Return the rate at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        return self._evaluate(self.value, inputs)
