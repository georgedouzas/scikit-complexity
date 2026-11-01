"""Define the helpers every engine package shares: symbols of compiled laws and declarations of parts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import inspect
from collections.abc import Iterable, Mapping

from sage.all__sagemath_symbolics import CC, RR, SR, ZZ, Expression
from sage.all__sagemath_symbolics import vector as sage_vector
from sage.modules.free_module_element import FreeModuleElement

from ..core.base import BasePart
from ..specs import EntrySpec, ParameterSpec, RoleSpec, StateSpec, VariableSpec
from ..types import VariableKind


def symbol(name: str, domain: object = RR) -> Expression:
    """Return the Sage symbol of a name, declared real or integer, or left complex, for the values of a Sage set.

    The symbol is declared without Maxima, which Sage would otherwise load to record the domain, so compiling stays
    fast. Sage keeps one symbol per name, so a name declared before with another domain is declared again.

    Args:
        name:
            The name of the symbol.

        domain:
            The Sage set of the values the symbol stands for.

    Returns:
        symbol:
            The Sage symbol.

    Examples:
        >>> from sage.all__sagemath_symbolics import CC
        >>> from skcomplex.engine._utils import symbol
        >>> symbol('z').is_real(), symbol('z', CC).is_real()
        (True, False)
    """
    item = SR.symbol(name)
    for previous in ('integer', 'real'):
        item.decl_forget(previous)
    assumption = EntrySpec._sage_assumption(domain)
    if assumption != 'complex':
        item.decl_assume('integer' if assumption == 'integer' else 'real')
    return item


def domain_of(item: Expression) -> object:
    """Return the Sage set a symbol was declared for: the integers, the reals, or the complex numbers.

    Args:
        item:
            The symbol.

    Returns:
        domain:
            `ZZ`, `RR`, or `CC`.

    Examples:
        >>> from sage.all__sagemath_symbolics import ZZ
        >>> from skcomplex.engine._utils import domain_of, symbol
        >>> domain_of(symbol('k', ZZ))
        Integer Ring
    """
    if item.is_integer():
        return ZZ
    return RR if item.is_real() else CC


def vector(components: Iterable[object]) -> FreeModuleElement:
    """Return a vector over the symbolic ring.

    Args:
        components:
            The components.

    Returns:
        vector:
            The vector.
    """
    return sage_vector(SR, list(components))


def components_of(value: object) -> list[object]:
    """Return the components of a vector, or a scalar as its only component.

    Args:
        value:
            A vector, an expression, or a number.

    Returns:
        components:
            The components.
    """
    return list(value) if isinstance(value, FreeModuleElement) else [value]


def symbols_of(expressions: Iterable[object]) -> set[Expression]:
    """Return the symbols that expressions read.

    Args:
        expressions:
            The expressions.

    Returns:
        symbols:
            The symbols.
    """
    return {item for expression in expressions for item in SR(expression).variables()}


def bound_labels(value: object) -> list[str]:
    """Return the labels of the elements and groups a role binds.

    A role binds one label, a sequence of labels, or the entries of a variadic role. An entry is a label, or a pair of
    a label and the name of a state, and the entries may be the keys of a mapping, such as the coefficients of a
    reaction. A key of a mapping that is a name alone is a state of the element another role binds, not a label.

    Args:
        value:
            The value of the role.

    Returns:
        labels:
            The labels, in the order the role binds them.

    Examples:
        >>> from skcomplex.engine._utils import bound_labels
        >>> bound_labels({('cell', 'u'): 2.0, 'v': 1.0})
        ['cell']
        >>> bound_labels(['cell', 'tank'])
        ['cell', 'tank']
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [entry[0] for entry in value if isinstance(entry, tuple) and entry and isinstance(entry[0], str)]
    entries = list(value) if isinstance(value, list | tuple) else []
    labels = [entry[0] if isinstance(entry, tuple) and entry else entry for entry in entries]
    return [label for label in labels if isinstance(label, str)]


def _variable_of(spec: StateSpec | ParameterSpec | RoleSpec) -> VariableSpec:
    """Return the declaration of a spec."""
    if isinstance(spec, RoleSpec):
        return VariableSpec(VariableKind.ROLE, variadic=spec.variadic, noise=spec.noise, optional=spec.optional)
    if isinstance(spec, ParameterSpec):
        return VariableSpec(
            VariableKind.STATE if spec.adaptive else VariableKind.PARAMETER,
            spec.domain,
            spec.vector,
            per=spec.per,
            adaptive=spec.adaptive,
            bounds=spec.bounds,
            shape=spec.shape,
        )
    order = int(spec.order) if float(spec.order).is_integer() else float(spec.order)
    return VariableSpec(
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
        reference=spec.reference,
    )


def _companions(name: str, variable: VariableSpec, default: object) -> dict[str, tuple[VariableSpec, object]]:
    """Return the companion states of a state of higher order or with lags, with their defaults, by name."""
    companions: dict[str, tuple[VariableSpec, object]] = {}
    domain = CC if variable.domain is CC else RR
    for index in range(1, variable.derivatives + 1):
        value = variable.initial[index - 1] if index <= len(variable.initial) else 0.0
        companion = VariableSpec(VariableKind.STATE, domain, variable.vector, derivative_of=name, index=index)
        companions[f'{name}_d{index}'] = (companion, value)
    for index in range(1, variable.lags + 1):
        value = variable.initial[index - 1] if index <= len(variable.initial) else default
        companion = VariableSpec(VariableKind.STATE, variable.domain, variable.vector, lag_of=name, index=index)
        companions[f'{name}_lag{index}'] = (companion, value)
    return companions


def declared_variables(part: object) -> dict[str, tuple[VariableSpec, object]]:
    """Return the declaration and the default of every state, parameter, and role of a part, with companion states.

    A companion state follows the state it belongs to.

    Args:
        part:
            A part class or instance.

    Returns:
        variables:
            A mapping from name to the declaration and the default.

    Raises:
        TypeError:
            The part is not a part class or instance.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.specs import StateSpec
        >>> from skcomplex.engine._utils import declared_variables
        >>> class PendulumElement(Element):
        ...     def __init__(self, label, theta: Annotated[float, StateSpec(order=2)] = 0.1):
        ...         super().__init__(label)
        ...         self.theta = theta
        >>> list(declared_variables(PendulumElement))
        ['theta', 'theta_d1']
    """
    cls = part if isinstance(part, type) else type(part)
    if not issubclass(cls, BasePart):
        error_msg = f'{cls.__name__} is not a part, so it declares no variables.'
        raise TypeError(error_msg)
    return dict(_class_variables(cls))


@functools.cache
def _class_variables(cls: type[BasePart]) -> dict[str, tuple[VariableSpec, object]]:
    """Return the declaration and the default of every variable of a part class, read once per class.

    Rules read the declarations of their agents every step, such as the group a reference state names, so the
    signature and the annotations of a class are parsed once. Callers receive a copy from `declared_variables`.
    """
    parameters = inspect.signature(cls).parameters
    variables: dict[str, tuple[VariableSpec, object]] = {}
    for name, spec in cls._get_specs().items():
        variable = _variable_of(spec)
        default = parameters[name].default
        default = 0.0 if default is inspect.Parameter.empty else default
        variables[name] = (variable, default)
        if variable.kind == VariableKind.STATE:
            variables.update(_companions(name, variable, default))
    return variables


def declarations_of(part: object, kind: VariableKind) -> dict[str, VariableSpec]:
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
