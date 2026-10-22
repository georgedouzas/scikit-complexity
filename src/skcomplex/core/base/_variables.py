"""Declare the state variables, parameters, and roles of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ._domains import AnyDomain, Domain

METADATA_KEY = 'skcomplex'


class VariableKind(Enum):
    """Role a declared field plays in a component."""

    STATE = 'state'
    PARAMETER = 'parameter'
    ROLE = 'role'


@dataclass(frozen=True)
class Variable:
    """Declaration of a field of a component.

    Args:
        kind:
            Whether the field is a state variable, a parameter, or a role.

        domain:
            The set of values the field takes.

        vector:
            Whether the field has one component per dimension of the enclosing space.

        order:
            The order of the highest time derivative that laws give for a state.

        lags:
            The number of past values a state of a discrete map keeps.

        initial:
            The default initial values of the derivatives of orders 1 to `order - 1`, or of the lags 1 to `lags`.

        derivative_of:
            The state whose derivative a companion state holds, or `None`.

        lag_of:
            The state whose past value a companion state holds, or `None`.

        index:
            The order of the derivative or the lag that a companion state holds.

        variadic:
            Whether a role binds a sequence of element paths and group member references instead of one path.

        per:
            The variadic role whose entries a parameter holds one value for, or `None`.

        quantity:
            The conserved quantity a state holds, such as `'money'` or `'mass'`, or `None`.

        history:
            The values of a state before the start, which delayed laws read, a number or a function of time, or `None`
            to keep the initial value before the start.

        position:
            Whether a vector state is the position of its component in a continuous space, on which the boundary of
            the space acts.

        velocity_of:
            The position state whose rate a vector state is, which a reflecting boundary reverses, or `None`.

        adaptive:
            Whether a state was declared as an adaptive parameter, which laws may change.
    """

    kind: VariableKind
    domain: AnyDomain = Domain.REAL
    vector: bool = False
    order: int = 1
    lags: int = 0
    initial: tuple[Any, ...] = ()
    derivative_of: str | None = None
    lag_of: str | None = None
    index: int = 0
    variadic: bool = False
    per: str | None = None
    quantity: str | None = None
    history: Any = None
    position: bool = False
    velocity_of: str | None = None
    adaptive: bool = False


def _declare_field(default: Any, declaration: Variable) -> Any:
    metadata = {METADATA_KEY: declaration}
    if default is dataclasses.MISSING:
        return dataclasses.field(metadata=metadata)
    if not isinstance(default, int | float | complex | str | bytes | Enum | np.generic | type(None)):
        return dataclasses.field(default_factory=lambda: copy.deepcopy(default), metadata=metadata)
    return dataclasses.field(default=default, metadata=metadata)


def state(
    default: Any = 0.0,
    *,
    domain: AnyDomain = Domain.REAL,
    vector: bool = False,
    order: int = 1,
    lags: int = 0,
    initial: Sequence[Any] = (),
    quantity: str | None = None,
    history: Any = None,
    position: bool = False,
    velocity_of: str | None = None,
) -> Any:
    """Declare a state variable of an element, with its initial value as default.

    A state of order `k` gets companion states `<name>_d1` to `<name>_d<k-1>` that hold its time derivatives, and laws
    on it give its `k`-th derivative. A state of a discrete map with `m` lags gets companion states `<name>_lag1` to
    `<name>_lag<m>` that hold its past values.

    Args:
        default:
            The default initial value. A scalar default of a vector state applies to every component.

        domain:
            The set of values the state takes.

        vector:
            Whether the state has one component per dimension of the enclosing space.

        order:
            The order of the highest time derivative that laws give, 1 for first-order equations.

        lags:
            The number of past values the state keeps for difference equations, 0 for none.

        initial:
            The default initial values of the derivatives of orders 1 to `order - 1`, or of the past values at steps
            -1 to `-lags`. Missing derivatives default to 0, and missing past values to `default`.

        quantity:
            The conserved quantity the state holds, such as `'money'`, or `None`. Only flows change a tagged state, so
            the total of every quantity changes only through flows with the environment.

        history:
            The values of the state before the start, which delayed laws read, a number or a function of time, or
            `None` to keep the initial value before the start.

        position:
            Whether a vector state is the position of its component in a continuous space, on which the boundary of
            the space acts.

        velocity_of:
            The position state whose rate a vector state is, which a reflecting boundary of the space reverses.

    Returns:
        field:
            A dataclass field that carries the declaration.

    Raises:
        ValueError:
            The order is below 1, the lags are negative, both are given, or there are more initial values than
            companions.

    Examples:
        >>> from skcomplex.core.base import Domain, state
        >>> from skcomplex.core.model import Element
        >>> class Population(Element):
        ...     n: float = state(1.0, domain=Domain.NONNEGATIVE)
        >>> Population('rabbits', n=40).n
        40
    """
    if not (isinstance(order, int) and order >= 1 and isinstance(lags, int) and lags >= 0):
        error_msg = f'order = {order!r} must be an integer of at least 1 and lags = {lags!r} a nonnegative integer.'
        raise ValueError(error_msg)
    if order > 1 and lags:
        error_msg = 'A state takes either an order above 1 for differential equations or lags for difference equations.'
        raise ValueError(error_msg)
    if len(initial) > max(order - 1, lags):
        error_msg = f'{len(initial)} initial values were given for {max(order - 1, lags)} companion states.'
        raise ValueError(error_msg)
    declaration = Variable(
        VariableKind.STATE,
        domain,
        vector,
        order,
        lags,
        tuple(initial),
        quantity=quantity,
        history=history,
        position=position,
        velocity_of=velocity_of,
    )
    return _declare_field(default, declaration)


def declaration_of(value: object) -> Variable | None:
    """Return the declaration a dataclass field carries, or `None` for any other value.

    Args:
        value:
            A class attribute of a component.

    Returns:
        declaration:
            The declaration of a field made by `state`, `parameter`, or `role`, or `None`.
    """
    if isinstance(value, dataclasses.Field):
        return value.metadata.get(METADATA_KEY)
    return None


def companions_of(name: str, declaration: Variable, default: Any) -> dict[str, Any]:
    """Return the companion state fields of a state of higher order or with lags, by field name.

    The derivatives of a complex state are complex, and those of any other state are real.

    Args:
        name:
            The name of the state.

        declaration:
            The declaration of the state.

        default:
            The default value of the state, which missing past values take.

    Returns:
        fields:
            The companion fields, empty for a first-order state without lags.
    """
    fields = {}
    domain = Domain.COMPLEX if declaration.domain is Domain.COMPLEX else Domain.REAL
    for index in range(1, declaration.order):
        value = declaration.initial[index - 1] if index <= len(declaration.initial) else 0.0
        companion = Variable(VariableKind.STATE, domain, declaration.vector, derivative_of=name, index=index)
        fields[f'{name}_d{index}'] = _declare_field(value, companion)
    for index in range(1, declaration.lags + 1):
        value = declaration.initial[index - 1] if index <= len(declaration.initial) else default
        companion = Variable(VariableKind.STATE, declaration.domain, declaration.vector, lag_of=name, index=index)
        fields[f'{name}_lag{index}'] = _declare_field(value, companion)
    return fields


def parameter(
    default: Any = dataclasses.MISSING,
    *,
    domain: AnyDomain = Domain.REAL,
    vector: bool = False,
    per: str | None = None,
    adaptive: bool = False,
) -> Any:
    """Declare a parameter of an element or an interaction.

    A parameter declared per entry of a variadic role holds a sequence with one value per entry of the role. Laws read
    it as a tuple of symbols, and the component `i` of the parameter `name` of the interaction at `path` is addressed
    as `<path>.<name>[<i>]`.

    Args:
        default:
            The default value. A parameter without a default must be given to the constructor.

        domain:
            The set of values the parameter takes.

        vector:
            Whether the parameter has one component per dimension of the enclosing space.

        per:
            The name of a variadic role of the interaction whose entries the parameter holds one value for, or `None`.

        adaptive:
            Whether laws may change the parameter, such as a policy rate an institution sets. An adaptive parameter of
            an element is compiled and recorded as a state whose initial value is the one the constructor gets, so
            laws target it like any state.

    Returns:
        field:
            A dataclass field that carries the declaration.

    Raises:
        ValueError:
            The parameter is both a vector and declared per entry of a role.

    Examples:
        >>> from skcomplex.core.base import Domain, parameter, role
        >>> from skcomplex.core.model import Interaction
        >>> class Growth(Interaction):
        ...     r: float = parameter(1.0, domain=Domain.POSITIVE)
        >>> Growth('growth', r=0.5).r
        0.5
        >>> class Regulation(Interaction):
        ...     inputs: list[str] = role(variadic=True)
        ...     weights: list[float] = parameter(per='inputs')
        >>> Regulation('regulation', inputs=['a', 'b'], weights=[1.0, -1.0]).weights
        [1.0, -1.0]
    """
    if adaptive:
        if per is not None:
            error_msg = f'A parameter takes either `adaptive=True` or `per={per!r}`, not both.'
            raise ValueError(error_msg)
        return _declare_field(default, Variable(VariableKind.STATE, domain, vector, adaptive=True))
    if vector and per is not None:
        error_msg = f'A parameter takes either `vector=True` or `per={per!r}`, not both.'
        raise ValueError(error_msg)
    return _declare_field(default, Variable(VariableKind.PARAMETER, domain, vector, per=per))


def role(*, variadic: bool = False) -> Any:
    """Declare a role of an interaction, bound to an element path in the constructor.

    A plain role binds one element path, one group path, or one group member reference `<group>[<index>]`, and laws
    read a bound member like a single element whose states stay in the group. A variadic role binds a sequence of
    element paths and group member references, and laws read it as a tuple of namespaces, one per entry in the given
    order.

    Args:
        variadic:
            Whether the role binds a sequence of entries instead of one path.

    Returns:
        field:
            A required dataclass field that carries the declaration.

    Examples:
        >>> from skcomplex.core.base import role
        >>> from skcomplex.core.model import Interaction
        >>> class Mortality(Interaction):
        ...     target: str = role()
        >>> Mortality('death', target='foxes').target
        'foxes'
        >>> Mortality('death', target='foxes[3]').target
        'foxes[3]'
        >>> class Sum(Interaction):
        ...     target: str = role()
        ...     inputs: list[str] = role(variadic=True)
        >>> Sum('sum', target='total', inputs=['a', 'nodes[3]']).inputs
        ['a', 'nodes[3]']
    """
    return _declare_field(dataclasses.MISSING, Variable(VariableKind.ROLE, variadic=variadic))


def declarations_of(component: Any, kind: VariableKind) -> dict[str, Variable]:
    """Return the declared fields of one kind of a component, in declaration order.

    Args:
        component:
            A component class or instance.

        kind:
            The kind of field to return.

    Returns:
        declarations:
            A mapping from field name to declaration.
    """
    return {
        field.name: field.metadata[METADATA_KEY]
        for field in dataclasses.fields(component)
        if METADATA_KEY in field.metadata and field.metadata[METADATA_KEY].kind is kind
    }
