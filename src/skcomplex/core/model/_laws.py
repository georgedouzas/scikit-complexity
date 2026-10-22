"""Define the law kinds that describe how state changes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from sage.all__sagemath_symbolics import SR

from ..base import (
    DomainKind,
    Expression,
    LawKind,
    Requires,
    SymbolicVector,
    TimeKind,
    ValidationError,
)
from ._lattices import Drive, Energy, Extremal, LocalRule, Move, Relaxation, Sampler

if TYPE_CHECKING:
    from ._fields import Flux
    from ._flows import Auxiliary, Flow

type Target = Expression | SymbolicVector
type NumericFunction = Callable[[float, Mapping[str, Any]], Any]


def _checked_value(description: str, value: object) -> object:
    """Return a law value with a NumPy scalar turned into a Sage number, after rejecting NumPy arrays."""
    if isinstance(value, np.ndarray):
        error_msg = (
            f'The {description} is a NumPy array, which NumPy builds when a NumPy scalar multiplies a symbolic '
            'vector. Convert the NumPy scalar with `float(...)` first.'
        )
        raise ValidationError(error_msg)
    if isinstance(value, np.generic):
        return SR(value.item())
    return value


def _checked_matrix(description: str, matrix: object) -> tuple[tuple[object, ...], ...]:
    """Return a matrix as a tuple of rows of Sage values, after checking that it is a nonempty rectangle."""
    rows = [list(row) for row in matrix] if isinstance(matrix, list | tuple | np.ndarray) else []
    widths = {len(row) for row in rows}
    if not rows or len(widths) != 1 or 0 in widths:
        error_msg = (
            f'The matrix of the {description} must be a nonempty sequence of rows of equal length, but got {matrix!r}.'
        )
        raise ValidationError(error_msg)
    return tuple(
        tuple(_checked_value(f'entry of the matrix of the {description}', entry) for entry in row) for row in rows
    )


def _check_value(law: object, name: str, description: str) -> None:
    """Replace a value field of a frozen law by its checked value."""
    object.__setattr__(law, name, _checked_value(description, getattr(law, name)))


def _is_number(value: object) -> bool:
    """Check whether a value is a number, with no symbol in it."""
    try:
        complex(cast('Any', value))
    except (TypeError, ValueError):
        return False
    return True


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

    law_kind: ClassVar[LawKind] = LawKind.RATE
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.CONTINUOUS},
        domains={DomainKind.REAL, DomainKind.ANGLE, DomainKind.COMPLEX, DomainKind.VECTOR},
    )

    target: Target
    value: object
    order: int | None = None

    def __post_init__(self) -> None:
        """Check the value of the law."""
        _check_value(self, 'value', f'value of the {type(self).__name__.lower()} law of {self.target}')


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

    law_kind: ClassVar[LawKind] = LawKind.NOISE
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.CONTINUOUS},
        domains={DomainKind.REAL, DomainKind.ANGLE, DomainKind.VECTOR},
    )

    target: Target
    value: object
    order: int | None = None

    def __post_init__(self) -> None:
        """Check the value of the law."""
        _check_value(self, 'value', f'value of the {type(self).__name__.lower()} law of {self.target}')


@dataclass(frozen=True)
class Jump:
    """Random event that changes states by fixed increments at a given rate.

    Args:
        rate:
            The propensity of the event, as a symbolic expression or a callable `f(t, values)`.

        changes:
            A mapping from the symbol of a scalar state to its increment when the event fires. In a law of a group,
            a target may also be a member offset read `shift_member(s.n, k)`, so the event of member `i` changes
            member `i + k` with the boundary of the read. Past a fixed end that change is left out, so the quantity
            leaves the group. Offsets follow the member order, which on a lattice of more than one dimension does not
            follow lattice neighbours.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Jump, shift_member
        >>> s, i = symbol('s'), symbol('i')
        >>> Jump(0.1 * s * i, {s: -1, i: 1}).changes[i]
        1
        >>> n = symbol('n')
        >>> Jump(n, {n: -1, shift_member(n, 1): 1}).changes[shift_member(n, 1)]
        1
    """

    law_kind: ClassVar[LawKind] = LawKind.JUMP
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.CONTINUOUS},
        domains={DomainKind.REAL, DomainKind.INTEGER, DomainKind.BOOLEAN, DomainKind.CATEGORICAL},
    )

    rate: object
    changes: Mapping[Expression, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the rate and the increments of the jump."""
        _check_value(self, 'rate', f'rate of the jump of {", ".join(map(str, self.changes))}')
        for state, increment in self.changes.items():
            if not _is_number(increment):
                error_msg = (
                    f'The jump of {state} has the increment {increment}, which holds a symbol. Give the increment as a '
                    'number.'
                )
                raise ValidationError(error_msg)


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

    law_kind: ClassVar[LawKind] = LawKind.UPDATE
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE})

    target: Target
    value: object

    def __post_init__(self) -> None:
        """Check the value of the law."""
        _check_value(self, 'value', f'value of the {type(self).__name__.lower()} law of {self.target}')


@dataclass(frozen=True)
class Transition:
    """Step of a discrete-time Markov chain, a state over a finite alphabet drawn from a row of a transition matrix.

    The state's current value selects the row, and the next value is drawn from it. The state takes a categorical
    domain or the integers from 0 to `n - 1`. A transition is a kind of update law, so it combines with update laws.

    Args:
        target:
            The symbol of a scalar state.

        matrix:
            The square transition matrix, whose entry `(i, j)` is the probability of the step from `i` to `j`, as
            numbers or expressions of parameters.

    Raises:
        ValidationError:
            The matrix is not a nonempty sequence of rows of equal length.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Transition
        >>> x, q = symbol('x'), symbol('q')
        >>> Transition(x, [[q, 1 - q], [0.4, 0.6]]).matrix[0]
        (q, -q + 1)
    """

    law_kind: ClassVar[LawKind] = LawKind.TRANSITION
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.DISCRETE},
        domains={DomainKind.CATEGORICAL, DomainKind.INTEGER},
    )

    target: Expression
    matrix: Sequence[Sequence[object]]

    def __post_init__(self) -> None:
        """Check the matrix of the law."""
        object.__setattr__(self, 'matrix', _checked_matrix(f'transition law of {self.target}', self.matrix))


@dataclass(frozen=True)
class Emission:
    """Observation of a hidden Markov source, a state drawn from the row of an emission matrix a hidden state selects.

    The emission reads the hidden state's value after the same step, so the hidden and observed sequences align.

    Args:
        target:
            The symbol of the observed scalar state.

        source:
            The symbol of the hidden state, which an update or transition law of the same component sets.

        matrix:
            The emission matrix, one row per value of the hidden state and one column per value of the observed state,
            as numbers or expressions of parameters.

    Raises:
        ValidationError:
            The matrix is not a nonempty sequence of rows of equal length.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Emission
        >>> symbol_, hidden = symbol('symbol'), symbol('hidden')
        >>> len(Emission(symbol_, hidden, [[0.7, 0.3], [0.2, 0.8]]).matrix)
        2
    """

    law_kind: ClassVar[LawKind] = LawKind.TRANSITION
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.DISCRETE},
        domains={DomainKind.CATEGORICAL, DomainKind.INTEGER},
    )

    target: Expression
    source: Expression
    matrix: Sequence[Sequence[object]]

    def __post_init__(self) -> None:
        """Check the matrix of the law."""
        object.__setattr__(self, 'matrix', _checked_matrix(f'emission law of {self.target}', self.matrix))


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

    law_kind: ClassVar[LawKind] = LawKind.BOUNDARY
    requires: ClassVar[Requires] = Requires()

    target: Target
    value: object

    def __post_init__(self) -> None:
        """Check the value of the law."""
        _check_value(self, 'value', f'value of the {type(self).__name__.lower()} law of {self.target}')


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

    law_kind: ClassVar[LawKind] = LawKind.EVENT
    requires: ClassVar[Requires] = Requires(time={TimeKind.CONTINUOUS})

    guard: object
    reset: Mapping[Any, object] | None = None
    direction: int = 0
    terminal: bool = False
    label: str | None = None


@dataclass(frozen=True, init=False)
class FractionalNoise(Noise):
    """Amplitude of a fractional Brownian motion of a Hurst index that drives the highest time derivative of a state.

    The increments are fractional Gaussian noise, correlated over the whole run. A Hurst index of 1/2 gives the
    process of a `Noise` law, below 1/2 the increments anticorrelate, and above 1/2 they correlate. For other indices
    than 1/2 the amplitude must not depend on any state.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression of parameters and time. A vector target gets one fractional Brownian motion per
            component.

        hurst:
            The Hurst index `0 < H < 1`, a number or a parameter.

        order:
            The order of the derivative the noise drives, checked against the state's order, or `None` to use it.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import FractionalNoise
        >>> x, D = symbol('x'), symbol('D')
        >>> FractionalNoise(x, 2 * D, hurst=0.7).hurst
        0.7
    """

    hurst: object = 0.5

    def __init__(self, target: Target, value: object, hurst: object, order: int | None = None) -> None:
        object.__setattr__(self, 'target', target)
        object.__setattr__(self, 'value', value)
        object.__setattr__(self, 'hurst', hurst)
        object.__setattr__(self, 'order', order)
        self.__post_init__()


@dataclass(frozen=True)
class Renewal:
    """Renewal of states after waiting times drawn anew at every renewal, such as the jumps of a random walk.

    Every member of a group renews on its own clock. A renewal applies its reset to the states all at once, then
    draws the next waiting time from the states after the reset. Draws in the waiting time and the reset take a fresh
    value at every renewal of every member, and the trajectory records every renewal and counts them in the observable
    `<label>.renewals`. Renewals combine with rate and noise laws, which evolve the states between renewals.

    Args:
        waiting:
            The waiting time until the next renewal, usually a draw from a distribution with nonnegative support,
            such as `draw('wait', 'pareto', b=alpha)`.

        reset:
            A mapping from state symbol to its new value, which may hold draws, or `None` to keep the states.

        label:
            The name of the renewal in the trajectory, or `None` for the component's path.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Renewal, draw, normal
        >>> x = symbol('x')
        >>> renewal = Renewal(draw('wait', 'expon'), {x: x + normal('jump')})
        >>> renewal.reset[x]
        x + normal_jump(0, 1)
    """

    law_kind: ClassVar[LawKind] = LawKind.JUMP
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.CONTINUOUS},
        domains={DomainKind.REAL, DomainKind.INTEGER, DomainKind.BOOLEAN, DomainKind.CATEGORICAL},
    )

    waiting: object
    reset: Mapping[Any, object] | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        """Check the waiting time of the law."""
        _check_value(self, 'waiting', 'waiting time of a renewal law')


@dataclass(frozen=True)
class Rule:
    """Behaviour written as code, run once per step for every agent of an agent type, or once per step for a model.

    A rule of an agent type receives the agent and the model, `function(agent, model)`. A rule of an interaction
    receives the model alone, `function(model)`. A system with rules is agent-based and runs on Mesa.

    Args:
        function:
            The function that runs the rule.

        phase:
            The name of the phase of the step the rule runs in, one of the system's `phases`, or `None` for the
            default phase.

    Examples:
        >>> from skcomplex.core.model import Rule
        >>> rule = Rule(lambda agent, model: None)
        >>> callable(rule.function)
        True
    """

    law_kind: ClassVar[LawKind] = LawKind.RULE
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE})

    function: Callable[..., None]
    phase: str | None = None


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

    law_kind: ClassVar[LawKind] = LawKind.EQUATION
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.CONTINUOUS},
        domains={DomainKind.REAL, DomainKind.ANGLE, DomainKind.COMPLEX, DomainKind.VECTOR},
    )

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
    | Transition
    | Emission
    | Boundary
    | Rule
    | Equation
    | Event
    | Renewal
    | LocalRule
    | Drive
    | Extremal
    | Relaxation
    | Energy
    | Move
    | Sampler
    | Flux
    | Flow
    | Auxiliary
)
