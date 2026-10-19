"""Compile the hybrid parts of laws: delayed states, discrete events, and the breakpoints of time-dependent inputs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import Expression, ValidationError, symbols_of
from ._printing import numpy_function

if TYPE_CHECKING:
    from ._kernels import Reference

type Vector = npt.NDArray[np.float64]
type Scalars = Callable[[float, Vector, Vector], list[float]]
RELATIONS = (operator.lt, operator.le, operator.gt, operator.ge)


@dataclass(frozen=True)
class Delay:
    """Delayed state read by laws through a hidden parameter that solvers fill from the run's history.

    Args:
        parameters:
            The flat indices of the hidden parameter, one per member.

        states:
            The flat indices of the delayed state component, aligned with `parameters`.

        tau:
            The function `tau(t, x, p)` that returns the delay.

        minimum:
            The smallest value of the delay, which bounds the steps of the method of steps.

        symbol:
            The symbol that replaces the delayed value in the equations.

        state:
            The symbol of the delayed state.

        delay:
            The symbolic delay.
    """

    parameters: npt.NDArray[np.intp]
    states: npt.NDArray[np.intp]
    tau: Callable[[float, Vector, Vector], float]
    minimum: float
    symbol: Expression
    state: Expression
    delay: Expression


@dataclass(frozen=True)
class CompiledEvent:
    """Event compiled into a guard function and a reset function of the state vector.

    Args:
        label:
            The name of the event.

        guard:
            The function `g(t, x, p)` whose crossing of zero triggers the event.

        reset:
            The function `r(t, x, p)` that returns the state vector after the event, or `None` to keep it.

        direction:
            `1`, `-1`, or `0` for crossings upwards, downwards, or both.

        terminal:
            Whether the event ends the run.
    """

    label: str
    guard: Callable[[float, Vector, Vector], float]
    reset: Callable[[float, Vector, Vector], Vector] | None
    direction: int
    terminal: bool


@dataclass(frozen=True)
class EventRecord:
    """Event that happened during a run.

    Args:
        label:
            The name of the event.

        time:
            The time of the crossing.

        before:
            The state vector at the crossing, before the reset.

        after:
            The state vector after the reset.
    """

    label: str
    time: float
    before: Vector
    after: Vector


@dataclass(frozen=True)
class Breakpoints:
    """Times where time-dependent inputs jump, at which solvers restart.

    Args:
        fixed:
            Single times.

        periodic:
            Pairs of an origin and a period, each giving the times `origin + k period`.

        unresolved:
            Whether a condition on time has jumps at times that depend on parameters, so solvers restart at every
            frame instead.
    """

    fixed: tuple[float, ...] = ()
    periodic: tuple[tuple[float, float], ...] = field(default=())
    unresolved: bool = False

    def next_after(self, time: float) -> float | None:
        """Return the first breakpoint strictly after a time, or `None` when none follows.

        Args:
            time:
                The time.

        Returns:
            breakpoint:
                The first later breakpoint.
        """
        candidates = [item for item in self.fixed if item > time]
        for origin, period in self.periodic:
            following = origin + (math.floor((time - origin) / period) + 1) * period
            while following <= time:
                following += period
            candidates.append(following)
        return min(candidates, default=None)

    def between(self, start: float, end: float) -> list[float]:
        """Return the breakpoints strictly between two times, in order.

        Args:
            start:
                The first time.

            end:
                The last time.

        Returns:
            times:
                The breakpoints.
        """
        times = {time for time in self.fixed if start < time < end}
        for origin, period in self.periodic:
            first = math.floor((start - origin) / period) + 1
            last = math.ceil((end - origin) / period)
            times.update(origin + k * period for k in range(first, last) if start < origin + k * period < end)
        return sorted(times)


def flat_sources(references: Mapping[Expression, 'Reference'], owner: str) -> dict[Expression, tuple[bool, int]]:
    """Return the vector and flat index of every symbol of single components in a namespace.

    Args:
        references:
            The references of a namespace's symbols.

        owner:
            The component the namespace belongs to, for error messages.

    Returns:
        sources:
            A mapping from symbol to whether it is a state and its flat index, for components outside groups.
    """
    del owner
    return {
        item: (reference.is_state, reference.entry.start + reference.column)
        for item, reference in references.items()
        if reference.entry.members is None
    }


def scalar_function(
    expressions: Sequence[Any],
    sources: Mapping[Expression, tuple[bool, int]],
    time: Expression,
    owner: str,
) -> Scalars:
    """Return a function of time, the state vector, and the parameter vector that evaluates scalar expressions.

    Args:
        expressions:
            The expressions.

        sources:
            A mapping from every symbol the expressions use to whether it is a state and its flat index.

        time:
            The time symbol.

        owner:
            The component the expressions belong to, for error messages.

    Returns:
        function:
            The function `f(t, x, p)` that returns one float per expression.

    Raises:
        ValidationError:
            An expression uses a symbol of a group member, which has no single value.
    """
    values = [SR(expression) for expression in expressions]
    arguments = sorted(symbols_of(values) - {time}, key=str)
    unknown = [item for item in arguments if item not in sources]
    if unknown:
        error_msg = (
            f'An event or delay of {owner} uses {", ".join(map(str, unknown))}, which belong to group members. Events '
            'and delays read states and parameters of single elements and interactions.'
        )
        raise ValidationError(error_msg)
    function = numpy_function(time, arguments, values)
    places = [sources[item] for item in arguments]

    def evaluate(t: float, x: Vector, p: Vector) -> list[float]:
        return [float(value) for value in function(t, *((x if is_state else p)[index] for is_state, index in places))]

    return evaluate


def _linear_root(expression: Expression, time: Expression) -> float | None:
    """Return where an expression that is linear in time with numeric coefficients is zero, or `None`."""
    slope = expression.diff(time)
    if not slope.is_numeric() or slope.is_zero() or (expression.variables() and set(expression.variables()) - {time}):
        return None
    return float(-expression.subs({time: 0}) / slope)


def breakpoints_of(expressions: Iterable[Any], time: Expression) -> Breakpoints:
    """Return the times where conditions on time in expressions change, as from step, pulse, and series inputs.

    A condition `u(t) < c` with `u` linear in time gives the root of `u(t) - c`, and a condition on `mod(u(t), P)` gives
    the periodic times where `u(t)` is 0 or `c` modulo `P`.

    Args:
        expressions:
            The expressions of laws.

        time:
            The time symbol.

    Returns:
        breakpoints:
            The fixed and periodic breakpoints.
    """
    fixed: set[float] = set()
    periodic: set[tuple[float, float]] = set()
    unresolved = False
    stack = [SR(expression) for expression in expressions]
    while stack:
        item = stack.pop()
        operation = item.operator()
        if operation is None:
            continue
        if operation in RELATIONS:
            left, right = item.operands()
            difference = left - right
            root = _linear_root(difference, time)
            resolved = root is not None or time not in difference.variables()
            if root is not None:
                fixed.add(root)
            for call in _mod_calls(difference):
                argument, period = call.operands()
                zero = _linear_root(argument, time)
                if zero is not None and period.is_numeric():
                    span = float(period)
                    offset = float(-(difference - call).subs({time: 0}))
                    periodic.update({(zero, span), (zero + offset, span)})
                    resolved = True
            unresolved = unresolved or not resolved
        stack.extend(item.operands())
    return Breakpoints(tuple(sorted(fixed)), tuple(sorted(periodic)), unresolved=unresolved)


def _mod_calls(expression: Expression) -> list[Expression]:
    """Return the calls of `mod` inside an expression."""
    found, stack = [], [expression]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        if getattr(item.operator(), 'name', lambda: '')() == 'mod':
            found.append(item)
        stack.extend(item.operands())
    return found
