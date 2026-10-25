"""Declare the state variables, parameters, and roles of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import dataclasses
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np

from ._domains import BaseDomain, NumericDomain
from ._errors import ArgumentError, LabelError

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
            The order of the highest time derivative that laws give for a state, an integer or a real number in
            `(0, 1)` for the Caputo derivative of a fractional order.

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

        noise:
            Whether a role binds a set of Wiener processes, which the noise laws of the interaction draw, instead of
            a path.

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

        bounds:
            The lower and upper bound of an adaptive parameter that is a control input, or `None` for an unbounded one.
    """

    kind: VariableKind
    domain: BaseDomain = NumericDomain.REAL
    vector: bool = False
    order: float = 1
    lags: int = 0
    initial: tuple[Any, ...] = ()
    derivative_of: str | None = None
    lag_of: str | None = None
    index: int = 0
    variadic: bool = False
    noise: bool = False
    per: str | None = None
    quantity: str | None = None
    history: Any = None
    position: bool = False
    velocity_of: str | None = None
    adaptive: bool = False
    bounds: tuple[float, float] | None = None

    @property
    def is_fractional(self) -> bool:
        """Whether the order of the state is not an integer, so its rate laws give a Caputo derivative."""
        return not isinstance(self.order, int)

    @property
    def derivatives(self) -> int:
        """Number of companion states that hold the derivatives of orders 1 to `order - 1`."""
        return 0 if self.is_fractional else int(self.order) - 1


def _normalized_order(order: object) -> float:
    """Return an order as an integer when it has an integer value, or raise an error when it is not positive."""
    if isinstance(order, bool) or not isinstance(order, int | float | np.integer | np.floating):
        error_msg = f'order = {order!r} must be a number: an integer of at least 1 or a real number in (0, 1).'
        raise TypeError(error_msg)
    if float(order).is_integer():
        return int(order)
    if not (math.isfinite(order) and order > 0):
        error_msg = f'order = {order!r} must be an integer of at least 1 or a real number in (0, 1).'
        raise ValueError(error_msg)
    return float(order)


def _declare_field(default: Any, declaration: Variable) -> Any:
    metadata = {METADATA_KEY: declaration}
    if default is dataclasses.MISSING:
        return dataclasses.field(metadata=metadata)
    if not isinstance(default, int | float | complex | str | bytes | Enum | np.generic | type(None)):
        return dataclasses.field(default_factory=lambda: copy.deepcopy(default), metadata=metadata)
    return dataclasses.field(default=default, metadata=metadata)


def _check_bounds(bounds: tuple[float, float], *, adaptive: bool) -> None:
    """Raise an error for bounds of a parameter that is not adaptive, or whose lower bound is above the upper one."""
    if not adaptive:
        error_msg = f'Only an adaptive parameter takes bounds, but bounds={bounds!r} were given. Add `adaptive=True`.'
        raise ValueError(error_msg)
    lower, upper = (float(value) for value in bounds)
    if not lower <= upper:
        error_msg = f'bounds={bounds!r} must be a pair (lower, upper) with lower at most upper.'
        raise ValueError(error_msg)


def _companions_of(name: str, declaration: Variable, default: Any) -> dict[str, Any]:
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
    domain = NumericDomain.COMPLEX if declaration.domain is NumericDomain.COMPLEX else NumericDomain.REAL
    for index in range(1, declaration.derivatives + 1):
        value = declaration.initial[index - 1] if index <= len(declaration.initial) else 0.0
        companion = Variable(VariableKind.STATE, domain, declaration.vector, derivative_of=name, index=index)
        fields[f'{name}_d{index}'] = _declare_field(value, companion)
    for index in range(1, declaration.lags + 1):
        value = declaration.initial[index - 1] if index <= len(declaration.initial) else default
        companion = Variable(VariableKind.STATE, declaration.domain, declaration.vector, lag_of=name, index=index)
        fields[f'{name}_lag{index}'] = _declare_field(value, companion)
    return fields


def _default_of(field: Any) -> Any:
    """Return the default value a declared field carries, a fresh copy for a mutable default."""
    factory = field.default_factory
    return factory() if callable(factory) else field.default


def _is_declaration(value: object) -> bool:
    """Check whether a value is a field declared with `state`, `parameter`, or `role`."""
    return isinstance(value, dataclasses.Field)


def _laws_method(laws: Callable[..., list[Any]]) -> Callable[..., list[Any]]:
    """Return a `laws` method from a function of the namespace, or of the instance and the namespace."""
    if len(inspect.signature(laws).parameters) >= 2:  # noqa: PLR2004
        return laws

    def method(self: object, s: Any) -> list[Any]:
        """Return the laws of the function."""
        return laws(s)

    method.__doc__ = laws.__doc__ or 'Return the laws of the type.'
    return method


def _caller_module() -> str:
    """Return the module of the code that called `define`, three calls above this one, so the type pickles there."""
    frame = inspect.currentframe()
    for _ in range(3):
        frame = frame.f_back if frame is not None else None
    return frame.f_globals.get('__name__', __name__) if frame is not None else __name__


def _fields(kind: type) -> dict[str, Any]:
    """Return the declared fields of a type, after rejecting the names it reserves."""
    fields = declared_fields(kind)
    reserved = sorted(set(fields) & kind.reserved_names)  # type: ignore[attr-defined]
    if reserved:
        error_msg = (
            f'{kind.__name__} declares the variable {reserved[0]!r}, whose name is reserved. Rename it, since '
            f'{", ".join(sorted(kind.reserved_names))} are reserved names.'  # type: ignore[attr-defined]
        )
        raise LabelError(error_msg)
    return fields


def _settings(kind: type, root: type) -> dict[str, inspect.Parameter]:
    """Return the keyword arguments of the constructors of a type up to its base, without `**values`."""
    found: dict[str, inspect.Parameter] = {}
    for klass in kind.__mro__:
        init = vars(klass).get('__init__')
        if init is None:
            continue
        parameters = list(inspect.signature(init).parameters.values())[1:]
        for setting in parameters:
            if setting.kind not in {setting.VAR_KEYWORD, setting.VAR_POSITIONAL}:
                found.setdefault(setting.name, setting)
        if klass is root or not any(item.kind == item.VAR_KEYWORD for item in parameters):
            break
    return found


def _declare(instance: Any, label: str, values: Mapping[str, Any]) -> None:
    """Store the label and every given value of an instance, the default for a declared name not given.

    A declared name without a default that is not given is stored as `dataclasses.MISSING`, and every given name is
    stored, also one the type does not declare, so that `_check_declared` reports both.
    """
    instance.label = label
    for name, field in declared_fields(type(instance)).items():
        setattr(instance, name, values[name] if name in values else _default_of(field))
    for name, value in values.items():
        setattr(instance, name, value)


def _constructor_names(kind: type, root: type) -> set[str]:
    """Return the names the constructors of a type up to its base take, without `**values`."""
    names: set[str] = set()
    for klass in kind.__mro__:
        init = vars(klass).get('__init__')
        if init is not None:
            parameters = list(inspect.signature(init).parameters.values())[1:]
            names |= {item.name for item in parameters if item.kind not in {item.VAR_KEYWORD, item.VAR_POSITIONAL}}
        if klass is root:
            break
    return names


def _check_declared(instance: Any, root: type) -> None:
    """Raise an error for a stored value whose name the type does not declare, or a declared name not given."""
    kind = type(instance)
    fields = _fields(kind)
    known = {'label', *fields, *_constructor_names(kind, root)}
    unknown = sorted(name for name in vars(instance) if name not in known and not name.endswith('_'))
    if unknown:
        declared = ', '.join(fields) or 'none'
        error_msg = (
            f'{kind.__name__} got {", ".join(map(repr, unknown))}, which it does not declare. Its declared '
            f'variables are {declared}.'
        )
        raise ArgumentError(error_msg)
    missing = [name for name in fields if getattr(instance, name, dataclasses.MISSING) is dataclasses.MISSING]
    if missing:
        error_msg = f'{kind.__name__} needs the arguments {", ".join(map(repr, missing))}.'
        raise ArgumentError(error_msg)


def _define_type(
    kind: type,
    name: str,
    module: str,
    *,
    roles: Sequence[str] = (),
    states: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    laws: Callable[..., list[Any]] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> type:
    """Return a subclass of a type built from declarations, a laws function, and class attributes, in a module.

    A state or a parameter given as a plain value is declared with that value as its default.
    """
    fields = {
        **{key: role() for key in roles},
        **{key: value if _is_declaration(value) else state(value) for key, value in (states or {}).items()},
        **{key: value if _is_declaration(value) else parameter(value) for key, value in (parameters or {}).items()},
    }
    namespace: dict[str, Any] = {'__module__': module, **(attributes or {}), **fields}
    namespace['__qualname__'] = name
    if laws is not None:
        namespace['laws'] = _laws_method(laws)
    new_type = type(name, (kind,), namespace)
    new_type._declarations()  # type: ignore[attr-defined]
    return new_type


def _declarations(kind: type) -> dict[str, 'Variable']:
    """Return the declarations of the states, parameters, and roles of a type, with the derived companions."""
    return {name: cast('Variable', declaration_of(field)) for name, field in _fields(kind).items()}


def _get_default(kind: type, root: type, name: str) -> Any:
    """Return the default value of a declared name or a setting of a type."""
    fields = _fields(kind)
    if name in fields:
        return _default_of(fields[name])
    setting = _settings(kind, root).get(name)
    if setting is None or setting.default is inspect.Parameter.empty:
        error_msg = f'{kind.__name__} has no default for {name!r}.'
        raise KeyError(error_msg)
    return setting.default


def _argument_names(kind: type, root: type) -> list[str]:
    """Return the names the constructor of a type takes besides the label: its settings and the declared names."""
    return [*(name for name in _settings(kind, root) if name != 'label'), *_fields(kind)]


def _get_params(instance: Any, root: type, *, deep: bool) -> dict[str, Any]:
    """Return the label, the settings, and the values of every declared name of an instance."""
    params: dict[str, Any] = {}
    for name in sorted(['label', *_argument_names(type(instance), root)]):
        value = getattr(instance, name)
        if deep and hasattr(value, 'get_params') and not isinstance(value, type):
            params.update({f'{name}__{key}': item for key, item in value.get_params().items()})
        params[name] = value
    return params


def state(
    default: Any = 0.0,
    *,
    domain: BaseDomain = NumericDomain.REAL,
    vector: bool = False,
    order: float = 1,
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
            The order of the highest time derivative that laws give, 1 for first-order equations, or a real number in
            `(0, 1)` for the Caputo derivative of a fractional order. An order that is neither is reported by
            validation.

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
        TypeError:
            The order is not a number.

        ValueError:
            The order is not positive, the lags are negative, an order other than 1 and lags are both given, or there
            are more initial values than companions.

    Examples:
        >>> from skcomplex.core.base import NumericDomain, state
        >>> from skcomplex.core.components import ElementComponent
        >>> class PopulationComponent(ElementComponent):
        ...     n: float = state(1.0, domain=NumericDomain.NONNEGATIVE)
        >>> PopulationComponent('rabbits', n=40).n
        40
    """
    order = _normalized_order(order)
    if not (order > 0 and isinstance(lags, int) and lags >= 0):
        error_msg = f'order = {order!r} must be an integer of at least 1 and lags = {lags!r} a nonnegative integer.'
        raise ValueError(error_msg)
    if order != 1 and lags:
        error_msg = (
            'A state takes either an order other than 1 for differential equations or lags for difference equations.'
        )
        raise ValueError(error_msg)
    companions = max(0 if isinstance(order, float) else order - 1, lags)
    if len(initial) > companions:
        error_msg = f'{len(initial)} initial values were given for {companions} companion states.'
        raise ValueError(error_msg)
    declaration = Variable(
        VariableKind.STATE,
        domain.check(),
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


def parameter(
    default: Any = dataclasses.MISSING,
    *,
    domain: BaseDomain = NumericDomain.REAL,
    vector: bool = False,
    per: str | None = None,
    adaptive: bool = False,
    bounds: tuple[float, float] | None = None,
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

        bounds:
            The lower and upper bound of an adaptive parameter, such as `(0, 1)`, which makes it a control input that a
            schedule, an optimal control solver, or a controller supplies, or `None` for no bounds. Validation reports
            an initial value outside the bounds.

    Returns:
        field:
            A dataclass field that carries the declaration.

    Raises:
        ValueError:
            The parameter is both a vector and declared per entry of a role, or has bounds without being adaptive or
            with a lower bound above the upper one.

    Examples:
        >>> from skcomplex.core.base import NumericDomain, parameter, role
        >>> from skcomplex.core.base import BaseInteraction
        >>> class GrowthInteraction(BaseInteraction):
        ...     r: float = parameter(1.0, domain=NumericDomain.POSITIVE)
        >>> GrowthInteraction('growth', r=0.5).r
        0.5
        >>> class RegulationInteraction(BaseInteraction):
        ...     inputs: list[str] = role(variadic=True)
        ...     weights: list[float] = parameter(per='inputs')
        >>> RegulationInteraction('regulation', inputs=['a', 'b'], weights=[1.0, -1.0]).weights
        [1.0, -1.0]
        >>> from skcomplex.core.components import ElementComponent
        >>> class FisheryComponent(ElementComponent):
        ...     effort: float = parameter(0.5, adaptive=True, bounds=(0.0, 1.0))
        >>> FisheryComponent('fishery').effort
        0.5
    """
    if bounds is not None:
        _check_bounds(bounds, adaptive=adaptive)
    if adaptive:
        if per is not None:
            error_msg = f'A parameter takes either `adaptive=True` or `per={per!r}`, not both.'
            raise ValueError(error_msg)
        checked = None if bounds is None else (float(bounds[0]), float(bounds[1]))
        declaration = Variable(VariableKind.STATE, domain.check(), vector, adaptive=True, bounds=checked)
        return _declare_field(default, declaration)
    if vector and per is not None:
        error_msg = f'A parameter takes either `vector=True` or `per={per!r}`, not both.'
        raise ValueError(error_msg)
    return _declare_field(default, Variable(VariableKind.PARAMETER, domain.check(), vector, per=per))


def role(*, variadic: bool = False, noise: bool = False) -> Any:
    """Declare a role of an interaction, bound to an element path or to a set of Wiener processes in the constructor.

    A plain role binds one element path, one group path, or one group member reference `<group>[<index>]`, and laws
    read a bound member like a single element whose states stay in the group. A variadic role binds a sequence of
    element paths and group member references, and laws read it as a tuple of namespaces, one per entry in the given
    order. A noise role binds a `WienerProcess`, whose processes the noise laws of the interaction name through the
    handle `s.<role>.dW`. Interactions of one system that bind processes of one label share them and their
    correlations.

    Args:
        variadic:
            Whether the role binds a sequence of entries instead of one path.

        noise:
            Whether the role binds a set of Wiener processes instead of a path.

    Returns:
        field:
            A required dataclass field that carries the declaration.

    Raises:
        ValueError:
            The role is both variadic and a noise role.

    Examples:
        >>> from skcomplex.core.base import role
        >>> from skcomplex.core.base import BaseInteraction
        >>> class MortalityInteraction(BaseInteraction):
        ...     target: str = role()
        >>> MortalityInteraction('death', target='foxes').target
        'foxes'
        >>> MortalityInteraction('death', target='foxes[3]').target
        'foxes[3]'
        >>> class SumInteraction(BaseInteraction):
        ...     target: str = role()
        ...     inputs: list[str] = role(variadic=True)
        >>> SumInteraction('sum', target='total', inputs=['a', 'nodes[3]']).inputs
        ['a', 'nodes[3]']
        >>> from skcomplex.core.components import WienerProcess
        >>> class ShockInteraction(BaseInteraction):
        ...     target: str = role()
        ...     market: WienerProcess = role(noise=True)
        >>> ShockInteraction('shock', target='stock', market=WienerProcess('market', dimension=2)).market.dimension
        2
    """
    if variadic and noise:
        error_msg = 'A role takes either `variadic=True` or `noise=True`, not both.'
        raise ValueError(error_msg)
    return _declare_field(dataclasses.MISSING, Variable(VariableKind.ROLE, variadic=variadic, noise=noise))


def declared_fields(component: Any) -> dict[str, Any]:
    """Return the fields a component class declares with `state`, `parameter`, and `role`, with their companions.

    The fields come in the order of the classes from the base to the class, each class in definition order followed by
    the companion states of its states of higher order or with lags, `<name>_d1` to `<name>_d<k-1>` and
    `<name>_lag1` to `<name>_lag<m>`. A field declared again in a subclass keeps its place.

    Args:
        component:
            A component class or instance.

    Returns:
        fields:
            A mapping from name to the field that carries the declaration and the default.

    Examples:
        >>> from skcomplex.core.base import declared_fields, state
        >>> class PendulumComponent:
        ...     theta = state(0.1, order=2)
        >>> list(declared_fields(PendulumComponent))
        ['theta', 'theta_d1']
    """
    cls = component if isinstance(component, type) else type(component)
    fields: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        own = {name: value for name, value in vars(klass).items() if declaration_of(value) is not None}
        fields.update(own)
        for name, value in own.items():
            declaration = cast('Variable', declaration_of(value))
            default = value.default if value.default is not dataclasses.MISSING else 0.0
            companions = _companions_of(name, declaration, default)
            fields.update({key: item for key, item in companions.items() if key not in vars(klass)})
    return fields


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
        name: field.metadata[METADATA_KEY]
        for name, field in declared_fields(component).items()
        if field.metadata[METADATA_KEY].kind == kind
    }
