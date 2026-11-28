"""Define the helpers every engine package shares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import inspect
import weakref
from collections.abc import Iterable, Mapping
from types import SimpleNamespace

from sage.all__sagemath_symbolics import CC, RR, SR, ZZ, Expression, var
from sage.all__sagemath_symbolics import vector as sage_vector
from sage.modules.free_module_element import FreeModuleElement

from ..components import EntrySpec, RoleSpec
from ..components._declared_variable_spec import _DeclaredVariableSpec
from ..components._parameter_spec import _ParameterSpec
from ..components._state_spec import _StateSpec
from ..components.base import BaseComponent
from ..components.elements import BaseElement
from ..components.interactions import BaseInteraction
from ..enums import VariableKind
from ..errors import SkcomplexError
from .problems._compiled_element_spec import _Variable

INSTANCE_VARIABLES: weakref.WeakKeyDictionary[
    BaseComponent,
    tuple[Mapping[str, _DeclaredVariableSpec], dict[str, tuple[_Variable, object]]],
] = weakref.WeakKeyDictionary()


def symbol(name: str, domain: object = RR) -> Expression:
    """Return the Sage symbol of a name, declared real, integer, or complex.

    The engine writes every value of a system as a Sage symbol, and the domain decides what Sage may assume when it
    simplifies, such as a real `x` whose conjugate is `x` itself. The symbol is declared without Maxima, which Sage
    would otherwise load, so compiling stays fast, and a name declared before with another domain is declared again. It
    names the size of a population, the phase of a wave, and the count of molecules.

    Read more in the user guide, under Extend the library.

    Args:
        name:
            The name of the symbol.

        domain:
            The Sage set of the values the symbol stands for.

    Returns:
        symbol:
            The Sage symbol.

    See Also:
        domain_of: Returns the domain a symbol was declared for.

    Examples:
        >>> from sage.all__sagemath_symbolics import CC
        >>> from skcomplex.core.engine._utils import symbol
        >>> symbol('z').is_real(), symbol('z', CC).is_real()
        (True, False)
    """
    item = SR.symbol(name)
    for previous in ('integer', 'real'):
        item.decl_forget(previous)
    assumption = EntrySpec._get_sage_assumption(domain)
    if assumption != 'complex':
        item.decl_assume('integer' if assumption == 'integer' else 'real')
    return item


def domain_of(item: Expression) -> object:
    """Return the Sage set a symbol was declared for: integers, reals, or complexes.

    It reads back what `symbol` declared, so the engine knows which laws a value takes, such as `ZZ` for a count that a
    jump raises by one. A function of time that equals its own conjugate, as the value of a real state that an
    interaction receives, is real. It tells counts of molecules, real prices, and complex amplitudes apart.

    Read more in the user guide, under Extend the library.

    Args:
        item:
            The symbol.

    Returns:
        domain:
            `ZZ`, `RR`, or `CC`.

    See Also:
        symbol: Declares a symbol for a domain.

    Examples:
        >>> from sage.all__sagemath_symbolics import ZZ
        >>> from skcomplex.core.engine._utils import domain_of, symbol
        >>> domain_of(symbol('k', ZZ))
        Integer Ring
    """
    if item.is_integer():
        return ZZ
    return RR if item.is_real() or (item.conjugate() - item).is_trivial_zero() else CC


def vector(components: Iterable[object]) -> FreeModuleElement:
    """Return a vector of Sage expressions from its components.

    The engine writes a vector value, such as the position of a particle in a plane, as a vector over the symbolic ring
    of Sage, so that laws can add, scale, and differentiate it component by component. It holds the position of a
    walker, the velocity of a bird, and the prices of a basket of goods.

    Read more in the user guide, under Extend the library.

    Args:
        components:
            The components of the vector.

    Returns:
        vector:
            The vector.

    See Also:
        components_of: Returns the components of a vector.

    Examples:
        >>> from skcomplex.core.engine._utils import symbol, vector
        >>> vector([symbol('x'), 2 * symbol('y')])
        (x, 2*y)
    """
    return sage_vector(SR, list(components))


def components_of(value: object) -> list[object]:
    """Return the components of a vector, or a single value as its only component.

    It lets the engine treat a number and a vector alike, such as `[x]` for a number `x` and `[x, y]` for a position in
    a plane. It splits the position of a walker, the velocity of a bird, and the prices of a basket of goods.

    Read more in the user guide, under Extend the library.

    Args:
        value:
            A vector, an expression, or a number.

    Returns:
        components:
            The components of the vector.

    See Also:
        vector: Builds a vector from its components.

    Examples:
        >>> from skcomplex.core.engine._utils import components_of, symbol, vector
        >>> components_of(vector([symbol('x'), symbol('y')])), components_of(3)
        ([x, y], [3])
    """
    return list(value) if isinstance(value, FreeModuleElement) else [value]


def symbols_of(expressions: Iterable[object]) -> set[Expression]:
    """Return the set of symbols that some expressions read.

    The engine reads it to learn which values a law depends on, such as `x` and `y` in `x + y`, so it gathers only those
    values at run time. It finds what the growth of a population, the force on a particle, and the price of a stock
    depend on.

    Read more in the user guide, under Extend the library.

    Args:
        expressions:
            The expressions.

    Returns:
        symbols:
            The symbols.

    See Also:
        symbol: Declares a symbol.

    Examples:
        >>> from skcomplex.core.engine._utils import symbol, symbols_of
        >>> sorted(map(str, symbols_of([symbol('x') + symbol('y'), 2 * symbol('x')])))
        ['x', 'y']
    """
    return {item for expression in expressions for item in SR(expression).variables()}


def bound_labels(value: object) -> list[str]:
    """Return the labels of the elements and groups that the value of a role binds.

    A role binds one label, a sequence of labels, or the entries of a variadic role, where an entry is a label or a pair
    of a label and the name of a state, such as `('cell', 'u')`. The entries may be the keys of a mapping, such as the
    coefficients of a reaction, and a key that is a name alone is a state of the element another role binds, not a
    label. It finds the species of a reaction, the tanks of a plant, and the accounts of a payment.

    Read more in the user guide, under Interactions.

    Args:
        value:
            The value of the role.

    Returns:
        labels:
            The labels, in the order the role binds them.

    See Also:
        binding_of: Returns the value a role of a component binds.

    Examples:
        >>> from skcomplex.core.engine._utils import bound_labels
        >>> bound_labels({('cell', 'u'): 2.0, 'v': 1.0}), bound_labels(['cell', 'tank'])
        (['cell'], ['cell', 'tank'])
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [entry[0] for entry in value if isinstance(entry, tuple) and entry and isinstance(entry[0], str)]
    entries = list(value) if isinstance(value, list | tuple) else []
    labels = [entry[0] if isinstance(entry, tuple) and entry else entry for entry in entries]
    return [label for label in labels if isinstance(label, str)]


def binding_of(component: object, role: str) -> object:
    """Return what a role of a component binds, such as the label of an element.

    An interaction gives it as its argument of the same name, such as `target='ball'`, or in `roles`, such as
    `roles={'target': 'ball'}` of a rule written without a class, and this reads it from either. It finds the ball that
    gravity pulls, the prey that a predator eats, and the account that a payment fills.

    Read more in the user guide, under Interactions.

    Args:
        component:
            The component, usually an interaction.

        role:
            The name of the role.

    Returns:
        binding:
            The label, labels, or entries the role binds, or `None` when it binds nothing.

    See Also:
        bound_labels: Returns the labels a role binds.

    Examples:
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.engine._utils import binding_of
        >>> binding_of(Interaction('gravity', roles={'target': 'ball'}), 'target')
        'ball'
    """
    if isinstance(component, BaseInteraction):
        return component._get_binding(role)
    return getattr(component, role, None)


def _variable_of(spec: _StateSpec | _ParameterSpec | RoleSpec) -> _Variable:
    """Return the declaration of a spec."""
    if isinstance(spec, RoleSpec):
        return _Variable(VariableKind.ROLE, variadic=spec.variadic, noise=spec.noise, optional=spec.optional)
    if isinstance(spec, _ParameterSpec):
        return _Variable(
            VariableKind.STATE if spec.adaptive else VariableKind.PARAMETER,
            spec.domain,
            spec.vector,
            per=spec.per,
            adaptive=spec.adaptive,
            bounds=spec.bounds,
            shape=spec.shape,
            tensor=spec.tensor,
            n_dim=spec.n_dim,
        )
    order = int(spec.order) if float(spec.order).is_integer() else float(spec.order)
    return _Variable(
        VariableKind.STATE,
        spec.domain,
        spec.vector,
        order,
        spec.lags,
        tuple(spec.initial),
        spec.period,
        quantity=spec.quantity,
        history=spec.history,
        reference=spec.reference,
        role=spec.role,
        tensor=spec.tensor,
        algebraic=spec.algebraic,
        group=spec.group,
        n_dim=spec.n_dim,
    )


def _companions(name: str, variable: _Variable, default: object) -> dict[str, tuple[_Variable, object]]:
    """Return the companion states of a state of higher order or with lags.

    They are given by name, with their defaults.
    """
    companions: dict[str, tuple[_Variable, object]] = {}
    domain = CC if variable.domain is CC else RR
    for index in range(1, variable.derivatives + 1):
        value = variable.initial[index - 1] if index <= len(variable.initial) else 0.0
        companion = _Variable(
            VariableKind.STATE,
            domain,
            variable.vector,
            derivative_of=name,
            index=index,
            tensor=variable.tensor,
            n_dim=variable.n_dim,
        )
        companions[f'{name}_d{index}'] = (companion, value)
    for index in range(1, variable.lags + 1):
        value = variable.initial[index - 1] if index <= len(variable.initial) else default
        companion = _Variable(
            VariableKind.STATE,
            variable.domain,
            variable.vector,
            lag_of=name,
            index=index,
            tensor=variable.tensor,
            n_dim=variable.n_dim,
        )
        companions[f'{name}_lag{index}'] = (companion, value)
    return companions


def _instance_variables(component: BaseComponent) -> dict[str, tuple[_Variable, object]]:
    """Return the variables of a component instance.

    They are those of its class and those it declares, read once per declaration.
    """
    declared = component._get_declared()
    if not declared:
        return _class_variables(type(component))
    known = INSTANCE_VARIABLES.get(component)
    if known is not None and known[0] is declared:
        return known[1]
    variables = dict(_class_variables(type(component)))
    for name, item in declared.items():
        variable = _variable_of(item.spec)
        variables[name] = (variable, item.value)
        if variable.kind == VariableKind.STATE:
            variables.update(_companions(name, variable, item.value))
    INSTANCE_VARIABLES[component] = (declared, variables)
    return variables


def declared_variables(component: object) -> dict[str, tuple[_Variable, object]]:
    """Return the declaration and the default of every value a component declares.

    A component declares its states, parameters, and roles in the annotations of the arguments of its `__init__`, such
    as a learning rate between 0 and 1. The engine reads them with their defaults, and the past values and derivatives
    that a state keeps follow the state they belong to. An instance adds the values it declares from data, with their
    values as defaults. It reads the rates of a reaction, the gains of a controller, and the step of a training.

    Read more in the user guide, under Extend the library.

    Args:
        component:
            A component class or instance.

    Returns:
        variables:
            A mapping from name to the declaration and the default.

    Raises:
        TypeError:
            The component is not a component class or instance.

    See Also:
        declarations_of: Returns the declarations of one kind.
        default_of: Returns the default of one value.

    Examples:
        >>> from skcomplex.core.components.interactions import TrainingInteraction
        >>> from skcomplex.core.engine._utils import declared_variables
        >>> declared_variables(TrainingInteraction)['learning_rate'][1]
        0.001
    """
    cls = component if isinstance(component, type) else type(component)
    if not issubclass(cls, BaseComponent):
        error_msg = f'{cls.__name__} is not a component, so it declares no variables.'
        raise TypeError(error_msg)
    return dict(_instance_variables(component)) if isinstance(component, BaseComponent) else dict(_class_variables(cls))


@functools.cache
def _class_variables(cls: type[BaseComponent]) -> dict[str, tuple[_Variable, object]]:
    """Return the declaration and the default of every variable of a component class.

    They are read once per class. Rules read the declarations of their agents every step, such as the group a reference
    state names, so the signature and the annotations of a class are parsed once. Callers receive a copy from
    `declared_variables`.
    """
    parameters = inspect.signature(cls).parameters
    variables: dict[str, tuple[_Variable, object]] = {}
    for name, spec in cls._get_specs().items():
        variable = _variable_of(spec)
        default = parameters[name].default
        default = 0.0 if default is inspect.Parameter.empty else default
        variables[name] = (variable, default)
        if variable.kind == VariableKind.STATE:
            variables.update(_companions(name, variable, default))
    return variables


def declarations_of(component: object, kind: VariableKind) -> dict[str, _Variable]:
    """Return the declarations of one kind of a component, such as its parameters.

    It keeps the values of `declared_variables` of one kind, states, parameters, or roles, with the past values and
    derivatives of the states. It lists the rates of a reaction, the gains of a controller, and the settings of a
    training.

    Read more in the user guide, under Extend the library.

    Args:
        component:
            A component class or instance.

        kind:
            The kind of declaration to return.

    Returns:
        declarations:
            A mapping from name to declaration.

    See Also:
        declared_variables: Returns every declaration with its default.

    Examples:
        >>> from skcomplex.core.components.interactions import TrainingInteraction
        >>> from skcomplex.core.engine._utils import declarations_of
        >>> from skcomplex.core.enums import VariableKind
        >>> list(declarations_of(TrainingInteraction, VariableKind.PARAMETER))
        ['learning_rate', 'momentum', 'beta1', 'beta2']
    """
    return {name: variable for name, (variable, _) in declared_variables(component).items() if variable.kind == kind}


def default_of(component: object, name: str) -> object:
    """Return the default of one value a component declares.

    The value is a state, a parameter, or a role, or a past value or derivative of a state, and its default is the
    default of its argument, such as `0.001` for a learning rate. It reads the initial rate of a reaction, the default
    gain of a controller, and the step of a training.

    Read more in the user guide, under Extend the library.

    Args:
        component:
            A component class or instance.

        name:
            The name of the variable.

    Returns:
        default:
            The default value.

    See Also:
        declared_variables: Returns every declaration with its default.

    Examples:
        >>> from skcomplex.core.components.interactions import TrainingInteraction
        >>> from skcomplex.core.engine._utils import default_of
        >>> default_of(TrainingInteraction, 'beta2')
        0.999
    """
    return declared_variables(component)[name][1]


def _probe_namespace(component: BaseElement | BaseInteraction) -> SimpleNamespace:
    """Return the namespace of free symbols of the declared names of a component.

    Probing the laws of the component reads this namespace. An element gives its states and its parameters, with
    `other`, the neighbour of a member, holding the same names. An interaction gives its parameters and an empty
    namespace per role, or an empty tuple for a variadic role, since the names of the bound elements are not known
    before compiling. Both give the time `t`.
    """
    specs = component._get_component_specs()
    names = {name: var(name) for name, spec in specs.items() if not isinstance(spec, RoleSpec)}
    roles = {
        name: () if spec.variadic else SimpleNamespace() for name, spec in specs.items() if isinstance(spec, RoleSpec)
    }
    other = {} if roles else {'other': SimpleNamespace(**names)}
    return SimpleNamespace(**names, **roles, **other, t=var('t'))


def probe_laws(component: object) -> list[object] | None:
    """Return the laws of a component before a system is built, when it can tell them.

    The engine asks an interaction for its laws with a free symbol for every value it declares, the time `t`, and an
    empty namespace for every role, so it can learn their kinds early, such as a rate law for a decay. An element states
    no laws and gives an empty list. Laws that read the values of the elements a role binds cannot be listed this way
    and give `None`. It finds early whether a model of growth, a market, or a circuit has rate, jump, or update laws.

    Read more in the user guide, under Extend the library.

    Args:
        component:
            The element, group, or interaction.

    Returns:
        laws:
            The laws, or `None` when they cannot be listed.

    See Also:
        declared_variables: Returns the values a component declares.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.engine._utils import probe_laws
        >>> probe_laws(Element('ball'))
        []
    """
    if isinstance(component, BaseElement):
        return []
    if not isinstance(component, BaseInteraction):
        return None
    try:
        return list(component._get_laws_of(_probe_namespace(component)))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ArithmeticError, RuntimeError, SkcomplexError):
        return None
