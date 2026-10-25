"""Define the law kinds that describe how state changes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np
from sage.all__sagemath_symbolics import SR

from ..base import (
    BaseLaw,
    DomainKind,
    Expression,
    LawClass,
    LawKind,
    Requires,
    SymbolicVector,
    TimeKind,
    Trait,
    ValidationError,
    register_laws,
)

type Target = Expression | SymbolicVector
type Process = str | tuple[str, ...] | Expression | SymbolicVector | None
NOISE_CONVENTIONS = {'ito': 0.0, 'stratonovich': 0.5, 'anti-ito': 1.0}
SENSES = ('min', 'max')
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


def _is_process(process: object) -> bool:
    """Check whether a value names the Wiener process of a noise law: `None`, a label, labels, or handles."""
    if process is None or isinstance(process, str | Expression | SymbolicVector):
        return True
    return isinstance(process, tuple) and bool(process) and all(isinstance(item, str) for item in process)


def _check_clock_options(jump: 'JumpLaw') -> None:
    """Raise an error for a jump that takes an option of the clock it does not use."""
    if jump.waiting is not None and jump.changes:
        given, fix = 'changes', 'Give the new values of the states as `reset`.'
    elif jump.waiting is None and (jump.reset is not None or jump.label is not None):
        given = 'a reset' if jump.reset is not None else 'a label'
        fix = 'Give the increments of the states as `changes`, or a waiting time instead of the rate.'
    else:
        return
    clock = 'a waiting-time clock' if jump.waiting is not None else 'an exponential clock'
    error_msg = (
        f'A jump on {clock} takes {given}, but changes belong to the exponential clock of a rate, and resets and '
        f'labels to the waiting-time clock. {fix}'
    )
    raise ValidationError(error_msg)


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
        >>> from skcomplex.core.laws import Call, RateLaw
        >>> n = symbol('n')
        >>> law = RateLaw(n, Call(lambda t, n: 2 * n, n))
        >>> law.value.arguments
        (n,)
    """

    function: object
    arguments: tuple[object, ...]

    def __init__(self, function: object, *arguments: object) -> None:
        object.__setattr__(self, 'function', function)
        object.__setattr__(self, 'arguments', arguments)


@dataclass(frozen=True)
class RateLaw(BaseLaw):
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
        >>> from skcomplex.core.laws import RateLaw
        >>> n = symbol('n')
        >>> RateLaw(n, 2 * n).value
        2*n
    """

    law_kind: ClassVar[LawKind] = LawKind.RATE
    time_class: ClassVar[LawClass] = LawClass.CONTINUOUS
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.CONTINUOUS},
        domains={DomainKind.REAL, DomainKind.ANGLE, DomainKind.COMPLEX, DomainKind.VECTOR},
    )

    target: Target
    value: object
    order: int | None = None

    def __post_init__(self) -> None:
        """Check the value of the law."""
        _check_value(self, 'value', f'value of the {type(self).__name__} of {self.target}')


@dataclass(frozen=True)
class NoiseLaw(BaseLaw):
    """Amplitude of a Wiener process that drives the highest time derivative of a state.

    Without a process, every component of the target and every member of a group gets its own independent Wiener
    process. Laws of one component, or of one member, that name the same local process share its increment. Laws that
    name a process of a `Wiener` element through a role share it across components and members.

    A conserved noise on a state of a field on a lattice adds `div(value eta)` instead, the finite-difference divergence
    of a random flux with one Wiener process per link between neighbouring sites. Every increment a site gains, its
    neighbour loses, so the total of the field changes only by its rate laws. Edges that are not periodic carry no
    process on their outer links.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression or a callable `f(t, values)`. A vector target gets one Wiener process per component.

        order:
            The order of the derivative the noise drives, checked against the state's order, or `None` to use it.

        convention:
            The stochastic calculus of the law, `'ito'`, `'stratonovich'`, or `'anti-ito'`. The compiler adds the
            drift correction of the convention, so every solver integrates the Itô form.

        process:
            `None` for independent processes, a label local to the component, a tuple of labels with one per
            component of a vector target, or the handle `s.<role>.dW` or `s.<role>.dW[k]` of a `Wiener` element.

        conserved:
            Whether the noise is the divergence of a random flux on the links of the lattice of a field, with the
            value as the amplitude of the flux, a number or an expression of parameters.

    Raises:
        ValidationError:
            The convention is unknown, the process is not a label, a tuple of labels, or a handle, or a conserved
            noise names a process.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import NoiseLaw
        >>> price = symbol('price')
        >>> NoiseLaw(price, 0.2 * price).value
        0.2*price
        >>> NoiseLaw(price, 0.2 * price, process='market').process
        'market'
        >>> NoiseLaw(symbol('rho'), 0.1, conserved=True).conserved
        True
    """

    law_kind: ClassVar[LawKind] = LawKind.NOISE
    time_class: ClassVar[LawClass] = LawClass.CONTINUOUS
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.CONTINUOUS},
        domains={DomainKind.REAL, DomainKind.ANGLE, DomainKind.VECTOR},
    )

    target: Target
    value: object
    order: int | None = None
    convention: str = 'ito'
    process: Process = None
    conserved: bool = False

    def __post_init__(self) -> None:
        """Check the value, the convention, and the process of the law."""
        _check_value(self, 'value', f'value of the {type(self).__name__} of {self.target}')
        if self.convention not in NOISE_CONVENTIONS:
            error_msg = f'Unknown noise convention {self.convention!r}. Use one of {", ".join(NOISE_CONVENTIONS)}.'
            raise ValidationError(error_msg)
        if not _is_process(self.process):
            error_msg = (
                f'The process of the noise law of {self.target} is {self.process!r}. Give a label, a tuple of labels, '
                'or the handle `s.<role>.dW` of a Wiener element.'
            )
            raise ValidationError(error_msg)
        if self.conserved and self.process is not None:
            error_msg = (
                f'The conserved noise law of {self.target} names the process {self.process!r}, but it draws one '
                'process per link of the lattice. Remove the process.'
            )
            raise ValidationError(error_msg)


@dataclass(frozen=True)
class JumpLaw(BaseLaw):
    """Random event that changes states, on an exponential clock of a given rate or on a clock of drawn waiting times.

    With a `rate`, the event fires at the times of a Poisson process of that propensity and changes states by fixed
    increments, as in continuous-time Markov chains. With a `waiting` time instead, it fires after a waiting time drawn
    anew at every firing, and applies its `reset`, as in renewal processes such as the jumps of a random walk. Every
    member of a group then fires on its own clock, draws in the waiting time and the reset take a fresh value at every
    firing, and the trajectory records every firing and counts them in `<label>.renewals`. Jumps on such clocks
    combine with rate and noise laws, which evolve the states between firings.

    Args:
        rate:
            The propensity of the event, as a symbolic expression or a callable `f(t, values)`, or `None` with a
            waiting time.

        changes:
            A mapping from the symbol of a scalar state to its increment when the event fires. In a law of a group,
            a target may also be a member offset read `shift_member(s.n, k)`, so the event of member `i` changes
            member `i + k` with the boundary of the read. Past a fixed end that change is left out, so the quantity
            leaves the group. Offsets follow the member order, which on a lattice of more than one dimension does not
            follow lattice neighbours.

        waiting:
            The waiting time until the next firing, usually a draw from a distribution with nonnegative support,
            such as `draw('wait', 'pareto', b=alpha)`, or `None` for the exponential clock of `rate`.

        reset:
            With a waiting time, a mapping from state symbol to its new value, which may hold draws, or `None`.

        label:
            With a waiting time, the name of the firings in the trajectory, or `None` for the component's path.

    Raises:
        ValidationError:
            The jump has both or neither of a rate and a waiting time, takes an option of the other clock, or an
            increment holds a symbol.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import JumpLaw, shift_member
        >>> s, i = symbol('s'), symbol('i')
        >>> JumpLaw(0.1 * s * i, {s: -1, i: 1}).changes[i]
        1
        >>> n = symbol('n')
        >>> JumpLaw(n, {n: -1, shift_member(n, 1): 1}).changes[shift_member(n, 1)]
        1
        >>> from skcomplex.core.laws import draw
        >>> JumpLaw(waiting=draw('wait', 'expon'), reset={n: 0}).reset[n]
        0
    """

    @property
    def renewal(self) -> 'RenewalLaw | None':
        """Renewal the compiler runs for a jump on a clock of waiting times, or `None` for an exponential clock."""
        return None if self.waiting is None else RenewalLaw(self.waiting, self.reset, self.label)

    law_kind: ClassVar[LawKind] = LawKind.JUMP
    time_class: ClassVar[LawClass] = LawClass.EVENT
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.CONTINUOUS},
        domains={DomainKind.REAL, DomainKind.INTEGER, DomainKind.BOOLEAN, DomainKind.CATEGORICAL},
    )

    rate: object = None
    changes: Mapping[Expression, float] = field(default_factory=dict)
    waiting: object = None
    reset: Mapping[Any, object] | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        """Check the clock and the increments of the jump."""
        if (self.rate is None) == (self.waiting is None):
            error_msg = 'A jump fires at a rate or after a waiting time. Give exactly one of them.'
            raise ValidationError(error_msg)
        _check_clock_options(self)
        if self.waiting is not None:
            _check_value(self, 'waiting', 'waiting time of a jump')
            return
        _check_value(self, 'rate', f'rate of the jump of {", ".join(map(str, self.changes))}')
        for state, increment in self.changes.items():
            if not _is_number(increment):
                error_msg = (
                    f'The jump of {state} has the increment {increment}, which holds a symbol. Give the increment as a '
                    'number.'
                )
                raise ValidationError(error_msg)


@dataclass(frozen=True)
class UpdateLaw(BaseLaw):
    """Next value of a state after one discrete time step.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression or a callable `f(t, values)` that gives the next value.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import UpdateLaw
        >>> x = symbol('x')
        >>> UpdateLaw(x, 3.7 * x * (1 - x)).target
        x
    """

    law_kind: ClassVar[LawKind] = LawKind.UPDATE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE})

    target: Target
    value: object

    def __post_init__(self) -> None:
        """Check the value of the law."""
        _check_value(self, 'value', f'value of the {type(self).__name__} of {self.target}')


@dataclass(frozen=True)
class TransitionLaw(BaseLaw):
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
        >>> from skcomplex.core.laws import TransitionLaw
        >>> x, q = symbol('x'), symbol('q')
        >>> TransitionLaw(x, [[q, 1 - q], [0.4, 0.6]]).matrix[0]
        (q, -q + 1)
    """

    law_kind: ClassVar[LawKind] = LawKind.TRANSITION
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
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
class EmissionLaw(BaseLaw):
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
        >>> from skcomplex.core.laws import EmissionLaw
        >>> symbol_, hidden = symbol('symbol'), symbol('hidden')
        >>> len(EmissionLaw(symbol_, hidden, [[0.7, 0.3], [0.2, 0.8]]).matrix)
        2
    """

    law_kind: ClassVar[LawKind] = LawKind.TRANSITION
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
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
class BoundaryLaw(BaseLaw):
    """Value a state takes after every solver step, such as a reflection at a wall or a wrap around a periodic box.

    Boundary laws do not change the form of a system, so they combine with every other law kind.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression or a callable `f(t, values)` that gives the new value.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import BoundaryLaw, mod
        >>> x = symbol('x')
        >>> BoundaryLaw(x, mod(x, 1)).value
        mod(x, 1)
    """

    law_kind: ClassVar[LawKind] = LawKind.BOUNDARY
    time_class: ClassVar[LawClass] = LawClass.CONSTRAINT
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.FORMLESS, Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires()

    target: Target
    value: object

    def __post_init__(self) -> None:
        """Check the value of the law."""
        _check_value(self, 'value', f'value of the {type(self).__name__} of {self.target}')


@dataclass(frozen=True)
class Objective(BaseLaw):
    """Cost or gain of a system that optimal control solvers minimise or maximise over its controls.

    Under continuous time the value of a run is the integral of the running value `expression` over the run plus the
    terminal value `terminal` at its horizon, the time `until` of the run. The controls are the bounded adaptive
    parameters of the system. The trajectory records the running value under `<component path>.<label>`.

    Args:
        expression:
            The running value, a symbolic expression of states, controls, parameters, and time.

        terminal:
            The terminal value at the horizon, a symbolic expression of states and parameters.

        sense:
            `'min'` for a cost to minimise or `'max'` for a gain to maximise.

        label:
            The name of the running value in the trajectory, unique among the transfers, auxiliaries, invariants, and
            objectives of its component.

    Raises:
        ValidationError:
            The sense is neither `'min'` nor `'max'`.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import Objective
        >>> x, u = symbol('x'), symbol('u')
        >>> Objective(x**2 + u**2, terminal=x**2).terminal
        x^2
    """

    law_kind: ClassVar[LawKind] = LawKind.OBJECTIVE
    time_class: ClassVar[LawClass] = LawClass.CONSTRAINT
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.KERNEL_FREE, Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.STATIC, TimeKind.CONTINUOUS})

    expression: object
    terminal: object = 0
    sense: str = 'min'
    label: str = 'objective'

    def __post_init__(self) -> None:
        """Check the values and the sense of the objective."""
        _check_value(self, 'expression', f'running value of the objective {self.label}')
        _check_value(self, 'terminal', f'terminal value of the objective {self.label}')
        if self.sense not in SENSES:
            error_msg = f'Unknown sense {self.sense!r} of the objective {self.label}. Use one of {", ".join(SENSES)}.'
            raise ValidationError(error_msg)


@dataclass(frozen=True)
class EventLaw(BaseLaw):
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
        >>> from skcomplex.core.laws import EventLaw
        >>> height, velocity = symbol('height'), symbol('velocity')
        >>> EventLaw(height, {velocity: -0.8 * velocity}, direction=-1).direction
        -1
    """

    law_kind: ClassVar[LawKind] = LawKind.EVENT
    time_class: ClassVar[LawClass] = LawClass.EVENT
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.FORMLESS, Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(time={TimeKind.CONTINUOUS})

    guard: object
    reset: Mapping[Any, object] | None = None
    direction: int = 0
    terminal: bool = False
    label: str | None = None


@dataclass(frozen=True)
class RenewalLaw(BaseLaw):
    """Renewal of states after waiting times drawn anew at every renewal, run for a waiting `JumpLaw`.

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
    """

    law_kind: ClassVar[LawKind] = LawKind.JUMP
    time_class: ClassVar[LawClass] = LawClass.EVENT
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


@dataclass(frozen=True, init=False)
class EquationLaw(BaseLaw):
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
        >>> from skcomplex.core.laws import EquationLaw, derivative
        >>> theta, g, l = symbol('theta'), symbol('g'), symbol('l')
        >>> EquationLaw(l * derivative(theta, 2) == -g * sin(theta)).rhs
        -g*sin(theta)
    """

    law_kind: ClassVar[LawKind] = LawKind.EQUATION
    time_class: ClassVar[LawClass] = LawClass.CONTINUOUS
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires(
        time={TimeKind.CONTINUOUS, TimeKind.STATIC},
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


register_laws(
    RateLaw,
    NoiseLaw,
    JumpLaw,
    UpdateLaw,
    TransitionLaw,
    EmissionLaw,
    BoundaryLaw,
    Objective,
    EventLaw,
    RenewalLaw,
    EquationLaw,
)
