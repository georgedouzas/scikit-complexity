"""Define the law kinds that describe how state changes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..base import Expression, SymbolicVector
from ._lattices import Drive, Energy, LocalRule, Move, Relaxation, Sampler

if TYPE_CHECKING:
    from ._fields import Flux

type Target = Expression | SymbolicVector
type NumericFunction = Callable[[float, Mapping[str, Any]], Any]


@dataclass(frozen=True, init=False)
class Call:
    """Law value computed by a function of time and scalar arguments, called once per member or pair.

    The Numba backend calls a Numba-compiled function inside its compiled kernels, and the NumPy backend calls any
    function from Python. The function returns a number for a scalar target, or one number per component for a vector
    target.

    Args:
        function:
            The function `f(t, *arguments)`.

        *arguments:
            The symbolic expressions whose values are passed to the function.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Call, Rate
        >>> n = symbol('n')
        >>> law = Rate(n, Call(lambda t, n: 2 * n, n))
        >>> law.value.arguments
        (n,)
    """

    function: object
    arguments: tuple[object, ...]

    def __init__(self, function: object, *arguments: object) -> None:
        object.__setattr__(self, 'function', function)
        object.__setattr__(self, 'arguments', arguments)


@dataclass(frozen=True)
class Rate:
    """Contribution to the highest time derivative of a state, the first for states of order 1.

    Rates on the same state add up, so a system with only rate laws is a system of ordinary differential equations. On
    a state of order `k`, a rate contributes to the `k`-th derivative, so independent mechanisms superpose.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression or a callable `f(t, values)` where `values` maps dotted state and parameter names to
            their current values.

        order:
            The order of the derivative the rate gives, checked against the state's order, or `None` to use it.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Rate
        >>> n = symbol('n')
        >>> Rate(n, 2 * n).value
        2*n
    """

    target: Target
    value: object
    order: int | None = None


@dataclass(frozen=True)
class Noise:
    """Amplitude of an independent Wiener process that drives the highest time derivative of a state.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression or a callable `f(t, values)`. A vector target gets one Wiener process per component.

        order:
            The order of the derivative the noise drives, checked against the state's order, or `None` to use it.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Noise
        >>> price = symbol('price')
        >>> Noise(price, 0.2 * price).value
        0.2*price
    """

    target: Target
    value: object
    order: int | None = None


@dataclass(frozen=True)
class Jump:
    """Random event that changes states by fixed increments at a given rate.

    Args:
        rate:
            The propensity of the event, as a symbolic expression or a callable `f(t, values)`.

        changes:
            A mapping from the symbol of a scalar state to its increment when the event fires.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Jump
        >>> s, i = symbol('s'), symbol('i')
        >>> Jump(0.1 * s * i, {s: -1, i: 1}).changes[i]
        1
    """

    rate: object
    changes: Mapping[Expression, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Update:
    """Next value of a state after one discrete time step.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression or a callable `f(t, values)` that gives the next value.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Update
        >>> x = symbol('x')
        >>> Update(x, 3.7 * x * (1 - x)).target
        x
    """

    target: Target
    value: object


@dataclass(frozen=True)
class Boundary:
    """Value a state takes after every solver step, such as a reflection at a wall or a wrap around a periodic box.

    Boundary laws do not change the form of a system, so they combine with every other law kind.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression or a callable `f(t, values)` that gives the new value.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Boundary, mod
        >>> x = symbol('x')
        >>> Boundary(x, mod(x, 1)).value
        mod(x, 1)
    """

    target: Target
    value: object


@dataclass(frozen=True)
class Event:
    """Discrete event: when a guard crosses zero, states take new values, or the run ends.

    Solvers locate the crossing within their tolerance. Events do not change the form of a system, so they combine with
    rate laws, and the trajectory records every event with its time and the states before and after it.

    Args:
        guard:
            A symbolic expression of states and time whose crossing of zero triggers the event.

        reset:
            A mapping from state symbol to its new value, symbolic, applied all at once, or `None` to keep the states.

        direction:
            `1` for crossings from negative to positive, `-1` for the opposite, and `0` for both.

        terminal:
            Whether the event ends the run.

        label:
            The name of the event in the trajectory, or `None` for the component's path.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Event
        >>> height, velocity = symbol('height'), symbol('velocity')
        >>> Event(height, {velocity: -0.8 * velocity}, direction=-1).direction
        -1
    """

    guard: object
    reset: Mapping[Any, object] | None = None
    direction: int = 0
    terminal: bool = False
    label: str | None = None


@dataclass(frozen=True)
class Rule:
    """Behaviour written as code, run once per step for every agent of an agent type, or once per step for a model.

    A rule of an agent type receives the agent and the model, `function(agent, model)`. A rule of an interaction
    receives the model alone, `function(model)`. A system with rules is agent-based and runs on Mesa.

    Args:
        function:
            The function that runs the rule.

    Examples:
        >>> from skcomplex.core.model import Rule
        >>> rule = Rule(lambda agent, model: None)
        >>> callable(rule.function)
        True
    """

    function: Callable[..., None]


@dataclass(frozen=True, init=False)
class Equation:
    """Implicit relation in which the highest time derivatives of states appear, such as Newton's second law.

    The compiler solves all equations of a system jointly for the highest derivatives, which must appear linearly, and
    adds the rate laws on the same states to the solution.

    Args:
        lhs:
            The left side, or a Sage relation `lhs == rhs`.

        rhs:
            The right side, or `None` when `lhs` is a relation.

    Examples:
        >>> from sage.all__sagemath_symbolics import sin
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Equation, derivative
        >>> theta, g, l = symbol('theta'), symbol('g'), symbol('l')
        >>> Equation(l * derivative(theta, 2) == -g * sin(theta)).rhs
        -g*sin(theta)
    """

    lhs: object
    rhs: object

    def __init__(self, lhs: object, rhs: object = None) -> None:
        relation: Any = lhs
        if rhs is None and hasattr(relation, 'is_relational') and relation.is_relational():
            lhs, rhs = relation.lhs(), relation.rhs()
        if rhs is None:
            error_msg = 'An equation takes a left and a right side, or one relation `lhs == rhs`.'
            raise ValueError(error_msg)
        object.__setattr__(self, 'lhs', lhs)
        object.__setattr__(self, 'rhs', rhs)


type Law = (
    Rate
    | Noise
    | Jump
    | Update
    | Boundary
    | Rule
    | Equation
    | Event
    | LocalRule
    | Drive
    | Relaxation
    | Energy
    | Move
    | Sampler
    | Flux
)
