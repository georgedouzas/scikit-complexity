"""Compile a system into a problem of one mathematical form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from dataclasses import dataclass, field, replace

import numpy as np
import sympy

from ..core import (
    Form,
    IncompatibleLawsError,
    LabelError,
    RoleError,
    ValidationError,
    Variable,
    VariableKind,
    declarations_of,
)
from ..model import Element, Jump, Law, Noise, Rate, Symbols, System, Update
from ._layout import Entry, Layout
from ._lowering import Context, Slots, build_function, symbolic_components
from ._problem import CompiledElement, Problem, Vector, VectorFunction
from ._tree import Node, coerce_value, flatten_system, shape_of

FORMS: dict[frozenset[type], Form] = {
    frozenset(): Form.ODE,
    frozenset({Rate}): Form.ODE,
    frozenset({Noise}): Form.SDE,
    frozenset({Rate, Noise}): Form.SDE,
    frozenset({Jump}): Form.JUMP,
    frozenset({Update}): Form.DISCRETE,
}


@dataclass
class _Declarations:
    """Entries, initial values, and symbols collected from the components."""

    time: sympy.Symbol
    states: list[Entry] = field(default_factory=list)
    parameters: list[Entry] = field(default_factory=list)
    initial_states: list[Vector] = field(default_factory=list)
    initial_parameters: list[Vector] = field(default_factory=list)
    symbols: dict[str, sympy.Symbol | sympy.MatrixBase] = field(default_factory=dict)

    def _declare(
        self,
        node: Node,
        variable: str,
        declaration: Variable,
        entries: list[Entry],
        values: list[Vector],
    ) -> sympy.Symbol | sympy.MatrixBase:
        """Add one variable of a component and return its symbol."""
        name = f'{node.path}.{variable}'
        shape = shape_of(name, declaration, node.space)
        value = coerce_value(name, getattr(node.component, variable), shape, declaration.domain)
        start = sum(entry.size for entry in entries)
        entries.append(Entry(name, start, shape, declaration.domain))
        values.append(value)
        base = f'{variable}_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}'
        assumptions = declaration.domain.assumptions
        symbol: sympy.Symbol | sympy.MatrixBase = (
            sympy.Matrix([sympy.Symbol(f'{base}_{index}', **assumptions) for index in range(shape[0])])
            if shape
            else sympy.Symbol(base, **assumptions)
        )
        self.symbols[name] = symbol
        return symbol

    def declare_component(self, node: Node) -> dict[str, sympy.Symbol | sympy.MatrixBase]:
        """Add the states and parameters of a component and return their symbols by variable name.

        Args:
            node:
                The component with its place in the tree.

        Returns:
            symbols:
                A mapping from variable name to symbol.
        """
        symbols = {
            variable: self._declare(node, variable, declaration, self.states, self.initial_states)
            for variable, declaration in declarations_of(node.component, VariableKind.STATE).items()
        }
        symbols.update(
            {
                variable: self._declare(node, variable, declaration, self.parameters, self.initial_parameters)
                for variable, declaration in declarations_of(node.component, VariableKind.PARAMETER).items()
            },
        )
        return symbols


def _check_symbol_names(problem_symbols: dict[str, sympy.Symbol | sympy.MatrixBase]) -> None:
    """Raise an error when two variables produce the same symbol name."""
    seen: dict[str, str] = {}
    for name, symbol in problem_symbols.items():
        for component in symbol if isinstance(symbol, sympy.MatrixBase) else [symbol]:
            if component.name in seen:
                error_msg = (
                    f'{name} and {seen[component.name]} produce the same symbol {component.name}. Rename one of '
                    'their labels or variables.'
                )
                raise LabelError(error_msg)
            seen[component.name] = name


def _resolve_roles(node: Node, namespaces: dict[str, Symbols]) -> dict[str, Symbols]:
    """Return the namespace of the element bound to each role of an interaction."""
    roles = {}
    for role in declarations_of(node.component, VariableKind.ROLE):
        label = getattr(node.component, role)
        path = f'{node.scope}{label}' if isinstance(label, str) else None
        if path not in namespaces:
            where = node.scope.rstrip('/') or 'the root system'
            error_msg = (
                f'Interaction {node.path} binds the role {role!r} to {label!r}, which is not an element path '
                f'relative to {where}.'
            )
            raise RoleError(error_msg)
        roles[role] = namespaces[path]
    return roles


def _collect_laws(nodes: list[Node], declarations: _Declarations) -> list[tuple[str, Law]]:
    """Return the laws of every component with the path of the component that returned it."""
    namespaces: dict[str, Symbols] = {}
    component_symbols: dict[str, dict[str, sympy.Symbol | sympy.MatrixBase]] = {}
    for node in nodes:
        component_symbols[node.path] = declarations.declare_component(node)
        if isinstance(node.component, Element):
            namespaces[node.path] = Symbols(t=declarations.time, **component_symbols[node.path])
    laws: list[tuple[str, Law]] = []
    for node in nodes:
        if isinstance(node.component, Element):
            namespace = namespaces[node.path]
        else:
            roles = _resolve_roles(node, namespaces)
            namespace = Symbols(t=declarations.time, **component_symbols[node.path], **roles)
        component_laws = node.component.laws(namespace)
        if not isinstance(component_laws, list | tuple) or not all(
            isinstance(law, Rate | Noise | Jump | Update) for law in component_laws
        ):
            error_msg = f'`laws` of {node.path} must return a list of Rate, Noise, Jump, or Update laws.'
            raise ValidationError(error_msg)
        laws.extend((node.path, law) for law in component_laws)
    return laws


def _infer_form(laws: list[tuple[str, Law]]) -> Form:
    """Return the form that the law kinds of a system determine."""
    kinds = frozenset(type(law) for _, law in laws)
    if kinds not in FORMS:
        found = ', '.join(sorted(kind.__name__ for kind in kinds))
        supported = '; '.join(
            ', '.join(sorted(kind.__name__ for kind in combination)) for combination in FORMS if combination
        )
        error_msg = f'The system mixes the law kinds {found}. Supported combinations are: {supported}.'
        raise IncompatibleLawsError(error_msg)
    return FORMS[kinds]


def _target_indices(target: object, index_of: dict[sympy.Symbol, int], owner: str) -> list[int]:
    """Return the state vector indices of a law target."""
    components = list(target) if isinstance(target, sympy.MatrixBase) else [target]
    if not all(isinstance(component, sympy.Symbol) and component in index_of for component in components):
        error_msg = f'A law of {owner} targets {target!r}, which is not a state symbol of the namespace `s`.'
        raise ValidationError(error_msg)
    return [index_of[component] for component in components]


def _fill_slots(slots: Slots, indices: list[int], value: object, context: Context, owner: str) -> None:
    """Add a law value to the slots at the given indices."""
    expressions = symbolic_components(value, len(indices), context, owner)
    if expressions is None:
        slots.add_callable(indices, value)  # type: ignore[arg-type]
        return
    for index, expression in zip(indices, expressions, strict=True):
        slots.add_expression(index, expression)


def _lower_continuous(
    laws: list[tuple[str, Law]],
    context: Context,
    index_of: dict[sympy.Symbol, int],
) -> tuple[Slots, Slots, VectorFunction, VectorFunction]:
    """Return the drift and noise slots and the drift and diffusion functions of rate and noise laws."""
    size = len(context.state_symbols)
    drift = Slots(size)
    rows: list[int] = []
    noise = Slots(0)
    for owner, law in laws:
        indices = _target_indices(law.target, index_of, owner)  # type: ignore[union-attr]
        if isinstance(law, Rate):
            _fill_slots(drift, indices, law.value, context, owner)
            continue
        columns = list(range(len(rows), len(rows) + len(indices)))
        rows.extend(indices)
        noise.size = len(rows)
        _fill_slots(noise, columns, law.value, context, owner)  # type: ignore[union-attr]
    amplitudes = build_function(noise, context)
    row_indices = np.asarray(rows, dtype=np.intp)

    def diffusion(t: float, x: Vector, p: Vector) -> Vector:
        """Return one column per independent Wiener process."""
        matrix = np.zeros((size, len(row_indices)))
        matrix[row_indices, np.arange(len(row_indices))] = amplitudes(t, x, p)
        return matrix

    return drift, noise, build_function(drift, context), diffusion


def _lower_jumps(
    laws: list[tuple[str, Law]],
    context: Context,
    index_of: dict[sympy.Symbol, int],
) -> tuple[Slots, VectorFunction, Vector]:
    """Return the rate slots, the propensity function, and the stoichiometry matrix of jump laws."""
    rates = Slots(len(laws))
    stoichiometry = np.zeros((len(laws), len(context.state_symbols)))
    for row, (owner, law) in enumerate(laws):
        _fill_slots(rates, [row], law.rate, context, owner)  # type: ignore[union-attr]
        for target, increment in law.changes.items():  # type: ignore[union-attr]
            (index,) = _target_indices(target, index_of, owner)
            try:
                stoichiometry[row, index] = float(increment)
            except (TypeError, ValueError) as error:
                error_msg = f'A jump of {owner} changes {target} by {increment!r}, which is not a number.'
                raise ValidationError(error_msg) from error
    return rates, build_function(rates, context), stoichiometry


def _lower_updates(
    laws: list[tuple[str, Law]],
    context: Context,
    index_of: dict[sympy.Symbol, int],
) -> tuple[Slots, VectorFunction]:
    """Return the update slots and the function that gives the next state vector."""
    updates = Slots(len(context.state_symbols))
    assigned: set[int] = set()
    for owner, law in laws:
        indices = _target_indices(law.target, index_of, owner)  # type: ignore[union-attr]
        if assigned.intersection(indices):
            error_msg = f'A law of {owner} updates {indices} a second time. Each state takes one update law.'
            raise ValidationError(error_msg)
        assigned.update(indices)
        _fill_slots(updates, indices, law.value, context, owner)  # type: ignore[union-attr]
    for index, symbol in enumerate(context.state_symbols):
        if index not in assigned:
            updates.add_expression(index, symbol)
    return updates, build_function(updates, context)


def _equations_of(slots: Slots, context: Context) -> dict[str, sympy.Expr]:
    """Return the symbolic expression of every state component without a callable law."""
    excluded = slots.callable_indices
    names = context.states.component_names
    return {
        names[index]: slots.expressions.get(index, sympy.Integer(0))
        for index in range(len(names))
        if index not in excluded
    }


def _lower(
    problem: Problem,
    laws: list[tuple[str, Law]],
    context: Context,
    index_of: dict[sympy.Symbol, int],
) -> Problem:
    """Return the problem with the numeric functions and equations of its form."""
    if problem.form in (Form.ODE, Form.SDE):
        slots, noise, drift, diffusion = _lower_continuous(laws, context, index_of)
        return replace(
            problem,
            drift=drift,
            diffusion=diffusion if problem.form is Form.SDE else None,
            equations=_equations_of(slots, context),
            has_callables=bool(slots.callables or noise.callables),
        )
    if problem.form is Form.JUMP:
        rates, propensities, stoichiometry = _lower_jumps(laws, context, index_of)
        return replace(
            problem,
            propensities=propensities,
            stoichiometry=stoichiometry,
            has_callables=bool(rates.callables),
        )
    updates, update = _lower_updates(laws, context, index_of)
    return replace(
        problem,
        update=update,
        equations=_equations_of(updates, context),
        has_callables=bool(updates.callables),
    )


def compile_system(system: System) -> Problem:
    """Compile a system into a problem of the form its law kinds determine.

    Args:
        system:
            The system to compile.

    Returns:
        problem:
            The compiled problem.

    Raises:
        ValidationError:
            A child has the wrong type, a value is outside its domain, or a law is malformed.

        LabelError:
            A label is not valid, duplicated, or produces a symbol collision.

        RoleError:
            A role refers to an element that does not exist.

        SpaceError:
            A vector variable has no enclosing space.

        IncompatibleLawsError:
            The system mixes law kinds that no form supports.

    Examples:
        >>> from skcomplex.compile import compile_system
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> problem = compile_system(
        ...     System(
        ...         elements=[Population('rabbits', n=10)],
        ...         interactions=[ExponentialGrowth('growth', target='rabbits', r=0.5)],
        ...     ),
        ... )
        >>> problem.form.name
        'ODE'
        >>> problem.equations
        {'rabbits.n': n_rabbits*r_growth}
    """
    nodes = flatten_system(system)
    declarations = _Declarations(time=sympy.Symbol('t', real=True))
    laws = _collect_laws(nodes, declarations)
    _check_symbol_names(declarations.symbols)
    states, parameters = Layout(declarations.states), Layout(declarations.parameters)
    problem = Problem(
        form=_infer_form(laws),
        states=states,
        parameters=parameters,
        initial_state=np.concatenate([np.zeros(0), *declarations.initial_states]),
        initial_parameters=np.concatenate([np.zeros(0), *declarations.initial_parameters]),
        time=declarations.time,
        symbols=declarations.symbols,
        equations={},
        has_callables=False,
        elements=tuple(
            CompiledElement(
                path=node.path,
                element=node.component,
                states={name: f'{node.path}.{name}' for name in declarations_of(node.component, VariableKind.STATE)},
                parameters={
                    name: f'{node.path}.{name}' for name in declarations_of(node.component, VariableKind.PARAMETER)
                },
            )
            for node in nodes
            if isinstance(node.component, Element)
        ),
    )
    context = Context(problem.time, problem.state_symbols, problem.parameter_symbols, states, parameters)
    index_of = {symbol: index for index, symbol in enumerate(context.state_symbols)}
    return _lower(problem, laws, context, index_of)
