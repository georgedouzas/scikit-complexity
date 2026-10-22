"""Define flows of conserved quantities, auxiliaries, lookup tables, and system-dynamics delays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from sage.all__sagemath_symbolics import SR
from sage.symbolic.function import BuiltinFunction

from ..base import (
    DomainKind,
    Expression,
    LawKind,
    Requires,
    ValidationError,
    symbol,
)
from ._functions import series
from ._laws import _check_value, _is_number

DELAY_ORDERS = {'delay1': 1, 'delay3': 3, 'smooth': 1, 'smooth3': 3}
MATERIAL_DELAYS = frozenset({'delay1', 'delay3'})


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
    and the first frame that breaks it.

    Args:
        label:
            The name of the invariant, unique among the transfers, auxiliaries, and invariants of its component.

        expression:
            The expression of states and parameters.

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
        _check_value(self, 'expression', f'expression of the invariant {self.label}')


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


def _delay(name: str, value: object, time: object) -> Expression:
    """Return the call of a delay function after checking the delay time."""
    if _is_number(time) and float(time) <= 0:  # type: ignore[arg-type]
        error_msg = f'The delay time of {name} must be positive, but got {time}.'
        raise ValueError(error_msg)
    function: Any = DELAY_FUNCTIONS[name]
    return function(value, time)


def delay1(value: object, time: object) -> Expression:
    """Return the outflow of a first-order material delay of an inflow, one hidden stock emptied at `stock / time`.

    Args:
        value:
            The symbolic inflow.

        time:
            The average delay time, a number or an expression of parameters.

    Returns:
        output:
            The symbolic outflow, which the compiler replaces by the hidden stock over the delay time.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import delay1
        >>> delay1(symbol('orders'), 4)
        delay1(orders, 4)
    """
    return _delay('delay1', value, time)


def delay3(value: object, time: object) -> Expression:
    """Return the outflow of a third-order material delay, three hidden stocks in series each with a third of the time.

    The response to a step is the Erlang distribution of order 3 with the mean `time`.

    Args:
        value:
            The symbolic inflow.

        time:
            The average delay time, a number or an expression of parameters.

    Returns:
        output:
            The symbolic outflow.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import delay3
        >>> delay3(symbol('shipments'), 6)
        delay3(shipments, 6)
    """
    return _delay('delay3', value, time)


def smooth(value: object, time: object) -> Expression:
    """Return the first-order information smoothing of a value, a hidden stock that adjusts at `gap / time`.

    Args:
        value:
            The symbolic value.

        time:
            The smoothing time, a number or an expression of parameters.

    Returns:
        output:
            The symbolic smoothed value, which starts at the value.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import smooth
        >>> smooth(symbol('sales'), 2)
        smooth(sales, 2)
    """
    return _delay('smooth', value, time)


def smooth3(value: object, time: object) -> Expression:
    """Return the third-order information smoothing of a value, three smoothings each with a third of the time.

    Args:
        value:
            The symbolic value.

        time:
            The smoothing time, a number or an expression of parameters.

    Returns:
        output:
            The symbolic smoothed value, which starts at the value.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import smooth3
        >>> smooth3(symbol('sales'), 3)
        smooth3(sales, 3)
    """
    return _delay('smooth3', value, time)


DELAY_FUNCTIONS = {name: _DelayFunction(name) for name in DELAY_ORDERS}
