"""Define flows of conserved quantities, auxiliaries, lookup tables, and system-dynamics delays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import pickle  # nosec B403: pickles the invariant's function to check that it pickles, and never loads
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from sage.all__sagemath_symbolics import SR
from sage.symbolic.function import BuiltinFunction

from ..base import (
    ConservationError,
    DomainKind,
    Expression,
    LawKind,
    Requires,
    SymbolicVector,
    ValidationError,
    symbol,
)
from ._functions import series
from ._laws import _check_value, _is_number

DELAY_NAME = re.compile(r'^(?P<kind>material|smoothing)_delay(?P<order>[1-9][0-9]*)$')


class _DelayFunction(BuiltinFunction):
    """Output of a system-dynamics delay or smoothing of an input with a delay time, `name(input, time)`."""

    def __init__(self, name: str) -> None:
        super().__init__(name, nargs=2)

    def _eval_(self, value: Expression, time: Expression) -> Expression | None:
        """Return a constant input unchanged, since every delay of a constant is the constant, or `None` otherwise."""
        if SR(value).is_numeric():
            return SR(value)
        return None


@dataclass(frozen=True)
class Source:
    """Environment a flow draws a quantity from, outside the system, such as a bank that creates money.

    Args:
        label:
            The name of the environment in transaction-flow matrices.

    Examples:
        >>> from skcomplex.core.model import Source
        >>> Source().label
        'environment'
    """

    label: str = 'environment'


@dataclass(frozen=True)
class Sink:
    """Environment a flow releases a quantity to, outside the system, such as a drain or the atmosphere.

    Args:
        label:
            The name of the environment in transaction-flow matrices.

    Examples:
        >>> from skcomplex.core.model import Sink
        >>> Sink('drain').label
        'drain'
    """

    label: str = 'environment'


type Holder = Expression | Source | Sink


def _check_holder(name: str, holder: object, environment: type) -> None:
    """Raise an error for a holder that is neither a scalar symbol nor the environment marker of its side."""
    if isinstance(holder, environment):
        return
    if isinstance(holder, Source | Sink) or not isinstance(holder, Expression) or holder.operator() is not None:
        error_msg = (
            f'The {name} of a flow must be the symbol of a scalar state or `{environment.__name__}()`, but got '
            f'{holder!r}.'
        )
        raise ValidationError(error_msg)


@dataclass(frozen=True)
class Transfer:
    """Movement of an amount of a quantity from a source holder to a sink holder.

    A transfer is one outflow of its source and one inflow of its sink. Its amount is a rate in a system of rate laws,
    the amount moved per step in a system of update laws, and the size of every jump when a `rate` is given. The
    trajectory records the amount at every frame under `<component path>.<label>`.

    Args:
        source:
            The symbol of the state the quantity leaves, or `Source()` or `None` for the environment.

        sink:
            The symbol of the state the quantity enters, or `Sink()` or `None` for the environment.

        amount:
            The symbolic amount, a rate, a per-step amount, or a jump size.

        label:
            The name of the flow, unique among the flows and auxiliaries of its component, or `None` for
            `flow_<index>`.

        rate:
            The propensity of a jump that moves `amount` at random times, or `None` for a deterministic flow.

    Raises:
        ValidationError:
            A holder is not a scalar state symbol or the environment, both holders are the environment, or the
            amount of a jump holds a symbol.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Transfer, Source
        >>> level = symbol('level')
        >>> Transfer(Source(), level, 5, label='faucet').label
        'faucet'
    """

    law_kind: ClassVar[LawKind] = LawKind.TRANSFER
    requires: ClassVar[Requires] = Requires(domains={DomainKind.REAL})

    source: Holder
    sink: Holder
    amount: object
    label: str | None = None
    rate: object = None

    def __post_init__(self) -> None:
        """Take `None` as the environment, and check the holders and the amount of the transfer."""
        if self.source is None:
            object.__setattr__(self, 'source', Source())
        if self.sink is None:
            object.__setattr__(self, 'sink', Sink())
        _check_holder('source', self.source, Source)
        _check_holder('sink', self.sink, Sink)
        if isinstance(self.source, Source) and isinstance(self.sink, Sink):
            error_msg = 'A flow moves a quantity into or out of a state, but both of its holders are the environment.'
            raise ValidationError(error_msg)
        _check_value(self, 'amount', f'amount of the flow {self.label or ""}'.rstrip())
        if self.rate is not None and not _is_number(self.amount):
            error_msg = (
                f'The flow {self.label} has a rate, so its amount is a jump size, but {self.amount} is a symbol.'
            )
            raise ValidationError(error_msg)


@dataclass(frozen=True)
class Auxiliary:
    """Named intermediate expression that laws reuse through its symbol and that trajectories record.

    Laws use `auxiliary.symbol` where the expression belongs, and the compiler substitutes the expression. The loop
    analysis keeps auxiliaries as nodes of the feedback graph, and the trajectory records the value under
    `<component path>.<label>`.

    Args:
        label:
            The name of the auxiliary, unique among the flows and auxiliaries of its component.

        expression:
            The symbolic expression, which can use the symbols of earlier auxiliaries.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Auxiliary
        >>> n, b = symbol('n'), symbol('b')
        >>> births = Auxiliary('births', b * n)
        >>> births.symbol
        births__aux
    """

    law_kind: ClassVar[LawKind] = LawKind.EQUATION
    requires: ClassVar[Requires] = Requires()

    label: str
    expression: object

    def __post_init__(self) -> None:
        """Check the expression of the auxiliary."""
        _check_value(self, 'expression', f'expression of the auxiliary {self.label}')

    @property
    def symbol(self) -> Expression:
        """The symbol laws use in place of the expression."""
        return symbol(f'{self.label}__aux')


@dataclass(frozen=True)
class Invariant:
    """Expression of the states of a component that stays constant during a run, such as a conserved total.

    The trajectory records the expression under `<component path>.<label>`, and every run checks at every frame that it
    differs from its value at the first frame by no more than the tolerance, raising an error that names the invariant
    and the first frame that breaks it. A group of agents declares an invariant of each member as a module-level
    function of the member, which the agent stepper checks after every step against the member's first value.

    Args:
        label:
            The name of the invariant, unique among the transfers, auxiliaries, and invariants of its component.

        expression:
            The expression of states and parameters, or a module-level function of a member of an agent group.

        tolerance:
            The largest change from the first frame that the run accepts.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Invariant
        >>> a, b = symbol('a'), symbol('b')
        >>> Invariant('total', a + b).tolerance
        1e-09
    """

    law_kind: ClassVar[LawKind] = LawKind.INVARIANT
    requires: ClassVar[Requires] = Requires()

    label: str
    expression: object
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Check the expression of the invariant."""
        if not self.is_member:
            _check_value(self, 'expression', f'expression of the invariant {self.label}')
            return
        try:
            pickle.dumps(self.expression)
        except (pickle.PicklingError, AttributeError, TypeError) as error:
            error_msg = (
                f'The function of the invariant {self.label!r} cannot be pickled: {error}. Use a module-level function.'
            )
            raise ValidationError(error_msg) from error

    @property
    def is_member(self) -> bool:
        """Whether the invariant is a function of each member of an agent group."""
        return callable(self.expression) and not isinstance(self.expression, Expression | SymbolicVector)

    def check(self, member: Any, first: float, name: str) -> None:  # noqa: ANN401
        """Raise an error when the invariant of a member moved from its first value by more than the tolerance.

        Args:
            member:
                The member, as rules receive it.

            first:
                The value of the invariant when the member was first checked.

            name:
                The name of the member in the message, such as its group and id.

        Raises:
            ConservationError:
                The invariant changed by more than the tolerance.
        """
        difference = float(self.expression(member)) - first
        if abs(difference) > self.tolerance:
            error_msg = (
                f'The invariant {self.label!r} of {name} does not hold: it changed by {difference:.6g}. Change '
                'the states it reads with transfers, or change all of them in the same rule.'
            )
            raise ConservationError(error_msg)


def transfer(source: Any, sink: Any, amount: float, state: str, sink_state: str | None = None) -> None:  # noqa: ANN401
    """Move an amount of a quantity from one agent to another at once, inside a rule.

    It is the imperative form of `Transfer` for agent rules. Both holdings change together, so the total of the quantity
    over all agents is unchanged, and a negative holding is an amount the agent owes.

    Args:
        source:
            The agent the amount leaves, as the rule receives it.

        sink:
            The agent the amount enters.

        amount:
            The amount moved, negative to move it the other way.

        state:
            The state of the source that holds the quantity.

        sink_state:
            The state of the sink that holds the quantity, or `None` for `state`.

    Raises:
        ValidationError:
            An agent has no such state.

    Examples:
        >>> from types import SimpleNamespace
        >>> from skcomplex.core.model import transfer
        >>> first, second = SimpleNamespace(level=10.0), SimpleNamespace(level=0.0)
        >>> transfer(first, second, 4.0, 'level')
        >>> first.level, second.level
        (6.0, 4.0)
    """
    target = state if sink_state is None else sink_state
    for agent, name in ((source, state), (sink, target)):
        if not hasattr(agent, name):
            error_msg = f'The transfer of {amount} reads the state {name!r}, which the agent {agent!r} does not have.'
            raise ValidationError(error_msg)
    if source is sink and state == target:
        return
    source_value, sink_value = getattr(source, state), getattr(sink, target)
    setattr(source, state, source_value - amount)
    setattr(sink, target, sink_value + amount)


def lookup(value: object, points: Sequence[Sequence[float]]) -> Expression:
    """Return a table function, the piecewise-linear interpolation of a value between points, flat outside them.

    Args:
        value:
            The symbolic input of the table.

        points:
            The points `(x, y)` of the table, with increasing `x`.

    Returns:
        output:
            The symbolic output, nested conditions that every backend lowers.

    Raises:
        ValueError:
            The points are empty or their inputs do not increase.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import lookup
        >>> food = symbol('food')
        >>> fertility = lookup(food, [(0, 0.0), (1, 2.0), (2, 3.0)])
        >>> [float(fertility.subs({food: value})) for value in (-1, 0.5, 1.5, 5)]
        [0.0, 1.0, 2.5, 3.0]
    """
    table = [(float(x), float(y)) for x, y in points]
    return series(SR(value), [x for x, _ in table], [y for _, y in table])


DELAY_FUNCTIONS: dict[str, _DelayFunction] = {}


def delay_parts(name: str) -> tuple[bool, int] | None:
    """Return whether a delay function conserves its input, and its order, from its name, or `None` for no delay.

    Args:
        name:
            The name of a function.

    Returns:
        parts:
            Whether the delay is material, and its order.
    """
    match = DELAY_NAME.match(name)
    return None if match is None else (match['kind'] == 'material', int(match['order']))


def distributed_delay(value: object, time: object, order: int = 1, *, conserved: bool = True) -> Expression:
    """Return the output of a distributed delay of a value: hidden stages in series, each with a share of the time.

    A conserved delay passes its input through the stages like a material, so its output is the outflow of the last
    stage, `stage / (time / order)`, and its response to a pulse is the Erlang distribution of the order with the mean
    `time`. A delay that does not conserve its input smooths it, each stage adjusting to the previous one at
    `gap / (time / order)`, and its output starts at the value. The compiler adds the stages as hidden stocks.

    Args:
        value:
            The symbolic input.

        time:
            The mean delay time, a positive number or an expression of parameters.

        order:
            The number of stages.

        conserved:
            Whether the delay conserves its input, as a material delay, or smooths it, as an information delay.

    Returns:
        output:
            The symbolic output.

    Raises:
        ValueError:
            The order is not a positive integer, or the delay time is a number that is not positive.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import distributed_delay
        >>> distributed_delay(symbol('orders'), 4, order=3)
        material_delay3(orders, 4)
        >>> distributed_delay(symbol('sales'), 2, conserved=False)
        smoothing_delay1(sales, 2)
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        error_msg = f'A distributed delay has a positive integer order, but got {order!r}.'
        raise ValueError(error_msg)
    name = f'{"material" if conserved else "smoothing"}_delay{order}'
    if _is_number(time) and float(time) <= 0:  # type: ignore[arg-type]
        error_msg = f'The delay time of {name} must be positive, but got {time}.'
        raise ValueError(error_msg)
    if name not in DELAY_FUNCTIONS:
        DELAY_FUNCTIONS[name] = _DelayFunction(name)
    function: Any = DELAY_FUNCTIONS[name]
    return function(value, time)
