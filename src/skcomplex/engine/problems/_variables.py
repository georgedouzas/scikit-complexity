"""Read the states, parameters, and roles that parts declare, with the companion states the compiler adds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
from dataclasses import dataclass
from enum import Enum

from sage.all__sagemath_symbolics import CC, RR

from ...core.specs import ParameterSpec, RoleSpec, StateSpec


class VariableKind(Enum):
    """Role a declared argument plays in a part."""

    STATE = 'state'
    PARAMETER = 'parameter'
    ROLE = 'role'


@dataclass(frozen=True)
class Variable:
    """Declaration of a state, a parameter, or a role of a part, as the compiler reads it.

    A state of order `k` adds the companion states `<name>_d1` to `<name>_d<k-1>` that hold its derivatives, and a
    state with `m` lags adds `<name>_lag1` to `<name>_lag<m>` that hold its past values.

    Args:
        kind:
            Whether the argument is a state, a parameter, or a role.

        domain:
            The Sage set of the values, or `None` for a structure rewritten as a whole.

        vector:
            Whether the variable has one component per dimension of the enclosing space.

        order:
            The order of the highest time derivative that laws give for a state.

        lags:
            The number of past values a state keeps.

        initial:
            The initial values of the derivatives of orders 1 to `order - 1`, or of the lags 1 to `lags`.

        period:
            The period that values of the state wrap around, or `None`.

        derivative_of:
            The state whose derivative a companion state holds, or `None`.

        lag_of:
            The state whose past value a companion state holds, or `None`.

        index:
            The order of the derivative or the lag that a companion state holds.

        variadic:
            Whether a role binds a sequence of entries instead of one path.

        noise:
            Whether a role binds a set of Wiener processes.

        per:
            The variadic role whose entries a parameter holds one value for, or `None`.

        quantity:
            The conserved quantity a state holds, or `None`.

        history:
            The values of a state before the start, or `None`.

        position:
            Whether a vector state is the position of its element in a continuous space.

        velocity_of:
            The position state whose rate a vector state is, or `None`.

        adaptive:
            Whether a parameter is adaptive, which laws may change.

        bounds:
            The lower and upper bound of an adaptive parameter, or `None`.
    """

    kind: VariableKind
    domain: object = RR
    vector: bool = False
    order: float = 1
    lags: int = 0
    initial: tuple[object, ...] = ()
    period: float | None = None
    derivative_of: str | None = None
    lag_of: str | None = None
    index: int = 0
    variadic: bool = False
    noise: bool = False
    per: str | None = None
    quantity: str | None = None
    history: object = None
    position: bool = False
    velocity_of: str | None = None
    adaptive: bool = False
    bounds: tuple[float, float] | None = None

    @property
    def is_fractional(self) -> bool:
        """Whether the order of the state is not an integer, so its rate laws give a Caputo derivative."""
        return not float(self.order).is_integer()

    @property
    def derivatives(self) -> int:
        """Number of companion states that hold the derivatives of orders 1 to `order - 1`."""
        return 0 if self.is_fractional else int(self.order) - 1


def _variable_of(spec: StateSpec | ParameterSpec | RoleSpec) -> Variable:
    """Return the declaration of a spec."""
    if isinstance(spec, RoleSpec):
        return Variable(VariableKind.ROLE, variadic=spec.variadic, noise=spec.noise)
    if isinstance(spec, ParameterSpec):
        return Variable(
            VariableKind.STATE if spec.adaptive else VariableKind.PARAMETER,
            spec.domain,
            spec.vector,
            per=spec.per,
            adaptive=spec.adaptive,
            bounds=spec.bounds,
        )
    order = int(spec.order) if float(spec.order).is_integer() else float(spec.order)
    return Variable(
        VariableKind.STATE,
        spec.domain,
        spec.vector,
        order,
        spec.lags,
        tuple(spec.initial),
        spec.period,
        quantity=spec.quantity,
        history=spec.history,
        position=spec.position,
        velocity_of=spec.velocity_of,
    )


def _companions(name: str, variable: Variable, default: object) -> dict[str, tuple[Variable, object]]:
    """Return the companion states of a state of higher order or with lags, with their defaults, by name."""
    companions: dict[str, tuple[Variable, object]] = {}
    domain = CC if variable.domain is CC else RR
    for index in range(1, variable.derivatives + 1):
        value = variable.initial[index - 1] if index <= len(variable.initial) else 0.0
        companion = Variable(VariableKind.STATE, domain, variable.vector, derivative_of=name, index=index)
        companions[f'{name}_d{index}'] = (companion, value)
    for index in range(1, variable.lags + 1):
        value = variable.initial[index - 1] if index <= len(variable.initial) else default
        companion = Variable(VariableKind.STATE, variable.domain, variable.vector, lag_of=name, index=index)
        companions[f'{name}_lag{index}'] = (companion, value)
    return companions


def declared_variables(part: object) -> dict[str, tuple[Variable, object]]:
    """Return the declaration and the default of every state, parameter, and role of a part, with companion states.

    A companion state follows the state it belongs to.

    Args:
        part:
            A part class or instance.

    Returns:
        variables:
            A mapping from name to the declaration and the default.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.engine.problems import declared_variables
        >>> class PendulumElement(Element):
        ...     def __init__(self, label, theta: Annotated[float, StateSpec(order=2)] = 0.1):
        ...         super().__init__(label)
        ...         self.theta = theta
        >>> list(declared_variables(PendulumElement))
        ['theta', 'theta_d1']
    """
    cls = part if isinstance(part, type) else type(part)
    parameters = inspect.signature(cls).parameters
    variables: dict[str, tuple[Variable, object]] = {}
    for name, spec in cls._get_specs().items():  # type: ignore[attr-defined]
        variable = _variable_of(spec)
        default = parameters[name].default
        default = 0.0 if default is inspect.Parameter.empty else default
        variables[name] = (variable, default)
        if variable.kind == VariableKind.STATE:
            variables.update(_companions(name, variable, default))
    return variables


def declarations_of(part: object, kind: VariableKind) -> dict[str, Variable]:
    """Return the declarations of one kind of a part, with companion states for states.

    Args:
        part:
            A part class or instance.

        kind:
            The kind of declaration to return.

    Returns:
        declarations:
            A mapping from name to declaration.
    """
    return {name: variable for name, (variable, _) in declared_variables(part).items() if variable.kind == kind}


def default_of(part: object, name: str) -> object:
    """Return the default of a declared state, parameter, or role of a part, or of one of its companion states.

    Args:
        part:
            A part class or instance.

        name:
            The name of the variable.

    Returns:
        default:
            The default value.
    """
    return declared_variables(part)[name][1]
