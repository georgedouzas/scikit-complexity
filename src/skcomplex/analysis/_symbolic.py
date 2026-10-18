"""Derive the equations of systems and solve them exactly with Sage."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from dataclasses import dataclass

from sage.all__sagemath_symbolics import SR, desolve, desolve_laplace, desolve_system, diff, function

from ..compile import Problem, compile_system
from ..core import AnalysisError, Expression, Form, components_of, declare_domains
from ..model import System

UNEVALUATED = frozenset({'ilt', 'laplace', 'integrate', 'integral'})


@dataclass(frozen=True)
class _Unknown:
    """State component whose equation the analysis writes in its own order, with its function of time."""

    name: str
    function: Expression
    order: int
    indices: tuple[int, ...]


def _components(problem: Problem) -> dict[str, tuple[str, int, int]]:
    """Return every equation name with its entry name, column, and flat index."""
    found = {}
    for entry in problem.states:
        for column, name in enumerate(entry.generic_names):
            found[name] = (entry.name, column, entry.start + column)
    return found


def _written(problem: Problem) -> tuple[dict[str, Expression], list[_Unknown], dict[Expression, Expression]]:
    """Return the equations in their written order, the unknown functions, and the substitution of symbols."""
    components = _components(problem)
    by_entry = {(entry, column): name for name, (entry, column, _) in components.items()}
    orders: dict[str, int] = {}
    for kind, base, index in problem.companions.values():
        if kind == 'derivative':
            orders[base] = max(orders.get(base, 1), index + 1)
    unknowns: list[_Unknown] = []
    substitution: dict[Expression, Expression] = {}
    for name, item in problem.equation_symbols.items():
        entry, column, index = components[name]
        if entry in problem.companions:
            continue
        order = orders.get(entry, 1)
        unknown_function = function(str(item))(problem.time)
        indices = [index]
        substitution[item] = unknown_function
        for degree in range(1, order):
            companion = by_entry[(f'{entry}_d{degree}', column)]
            substitution[problem.equation_symbols[companion]] = diff(unknown_function, problem.time, degree)
            indices.append(components[companion][2])
        for companion_name, (kind, base, steps) in problem.companions.items():
            if kind == 'lag' and base == entry:
                lagged = by_entry[(companion_name, column)]
                substitution[problem.equation_symbols[lagged]] = function(str(item))(problem.time - steps)
        unknowns.append(_Unknown(name, unknown_function, order, tuple(indices)))
    equations = {}
    for unknown in unknowns:
        entry, column, _ = components[unknown.name]
        source = by_entry[(f'{entry}_d{unknown.order - 1}', column)] if unknown.order > 1 else unknown.name
        right = SR(problem.equations[source]).subs(substitution)
        if problem.form is Form.DISCRETE:
            left = function(str(problem.equation_symbols[unknown.name]))(problem.time + 1)
        else:
            left = diff(unknown.function, problem.time, unknown.order)
        equations[unknown.name] = left == right
    return equations, unknowns, substitution


def _equations_of_problem(problem: Problem) -> dict[str, Expression]:
    """Return the equations of a compiled problem as relations between functions of time, in their written order."""
    if problem.form is Form.JUMP:
        error_msg = 'A jump process has no symbolic equations. Use a deterministic version of its interactions.'
        raise AnalysisError(error_msg)
    return _written(problem)[0]


def _first_order(problem: Problem) -> tuple[list[Expression], list[Expression]]:
    """Return the first-order equations of every state component, companions included, and their functions."""
    functions = {item: function(str(item))(problem.time) for item in problem.equation_symbols.values()}
    equations = [
        diff(functions[item], problem.time) == SR(problem.equations[name]).subs(functions)
        for name, item in problem.equation_symbols.items()
    ]
    return equations, list(functions.values())


def _forget_initial_values(functions: list[Expression], time: Expression) -> None:
    """Reset the values at `t = 0` that `desolve_system` leaves in Maxima for every function but the first."""
    for item in functions:
        maxima_function = item._maxima_()
        maxima_function.atvalue(time == 0, maxima_function)


def _is_closed(expression: Expression, functions: list[Expression], time: Expression) -> bool:
    """Check that an expression has no unevaluated transform or integral and no unknown function of time.

    Values of the unknown functions at fixed times, such as `x(0)`, are integration constants and are allowed.
    """
    names = {str(item.operator()) for item in functions}
    stack = [SR(expression)]
    while stack:
        item = stack.pop()
        operator = item.operator()
        if operator is not None:
            name = operator.name() if hasattr(operator, 'name') else str(operator)
            if name in UNEVALUATED or (str(operator) in names and time in item.variables()):
                return False
            stack.extend(item.operands())
    return True


def _parameter_values(problem: Problem) -> dict[Expression, float]:
    """Return the current value of every parameter component of single elements and interactions by symbol."""
    values = {}
    for entry in problem.parameters:
        if entry.members is not None or entry.name not in problem.symbols:
            continue
        for column, item in enumerate(components_of(problem.symbols[entry.name])):
            values[item] = float(problem.initial_parameters[entry.start + column])
    return values


def _sign_error(error: Exception) -> AnalysisError:
    """Return the error of a sign question that Maxima asks, naming the parameters it involves."""
    match = re.search(r'Is (.+?) (positive|negative|zero)', str(error))
    expression = match.group(1) if match else 'an expression'
    names = sorted(set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', expression))) if match else []
    error_msg = (
        f'Maxima needs the sign of {expression} to solve the equations. Declare the parameters '
        f'{", ".join(names) or "involved"} with `parameter(..., domain=Domain.POSITIVE)` or another domain that gives '
        'their sign.'
    )
    return AnalysisError(error_msg)


def equations_of(system: System) -> dict[str, Expression]:
    """Return the symbolic equations of a system, one per state component.

    Each state component is a function of time `t`. Continuous forms give its time derivative, the drift for
    stochastic systems, and discrete maps give its value at `t + 1`. States of higher order give their highest
    derivative, `diff(x(t), t, k)`, with their companions written as derivatives, and lagged states read their past
    values as `x(t - j)`. Every equation is a Sage relation. The equations
    of a group hold for every member and are named with the member index `i`. Components driven by a callable law or by
    a sum over members or pairs are left out.

    Args:
        system:
            The system.

    Returns:
        equations:
            A mapping from state component name to its equation.

    Raises:
        AnalysisError:
            The system is a jump process.

    Examples:
        >>> from skcomplex.analysis import equations_of
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import LogisticGrowth, Population
        >>> system = System(
        ...     elements=[Population('rabbits', n=10)],
        ...     interactions=[LogisticGrowth('growth', target='rabbits', r=1.0, K=100)],
        ... )
        >>> equations_of(system)['rabbits.n']
        diff(n_rabbits(t), t) == -r_growth*(n_rabbits(t)/K_growth - 1)*n_rabbits(t)
    """
    return _equations_of_problem(compile_system(system))


def solve_symbolically(system: System, *, use_initial_state: bool = False) -> dict[str, Expression]:
    """Solve the equations of an ODE system exactly as functions of time, with Maxima through Sage.

    One equation of order 1 or 2 is solved with `desolve`, one of higher order with `desolve_laplace`, and several with
    `desolve_system` after reduction to first order, which solves linear systems with constant coefficients. A result
    that still contains unevaluated transforms, integrals, or the unknown functions is not a solution, and raises the
    error. Parameters give Maxima their sign from their domain, or from their value when their domain is real.

    Args:
        system:
            The system.

        use_initial_state:
            Whether to use the initial state of the system, with the initial values of every derivative of states of
            higher order, as initial conditions at `t = 0`. Without them, the solution keeps its integration
            constants.

    Returns:
        solutions:
            A mapping from state component name to its solution, a relation `f(t) == solution`. States of higher order
            appear once, without their derivatives.

    Raises:
        AnalysisError:
            The system is not an ODE, has a callable law or groups, has equations Maxima cannot solve, or needs the
            sign of a parameter.

    Examples:
        >>> from skcomplex.analysis import solve_symbolically
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> system = System(
        ...     elements=[Population('a', n=2)],
        ...     interactions=[ExponentialGrowth('g', target='a', r=0.5)],
        ... )
        >>> solve_symbolically(system, use_initial_state=True)['a.n']
        n_a(t) == 2*e^(r_g*t)
    """
    problem = compile_system(system)
    if problem.form is not Form.ODE:
        error_msg = f'Only ODE systems can be solved symbolically, but the system is a {problem.form.value}.'
        raise AnalysisError(error_msg)
    if problem.has_callables:
        error_msg = 'The system has a callable law, which has no symbolic form.'
        raise AnalysisError(error_msg)
    if problem.has_group_sums or any(element.members is not None for element in problem.elements):
        error_msg = 'The system has element groups, which are not solved symbolically. Solve one member as an element.'
        raise AnalysisError(error_msg)
    equations, unknowns, _ = _written(problem)
    initial = problem.initial_state
    try:
        declare_domains([*problem.state_symbols, *problem.parameter_symbols, problem.time], _parameter_values(problem))
        if len(unknowns) == 1:
            unknown = unknowns[0]
            conditions = [0, *(float(initial[index]) for index in unknown.indices)] if use_initial_state else None
            equation = next(iter(equations.values()))
            if unknown.order <= 2:  # noqa: PLR2004
                solution = desolve(equation, unknown.function, ivar=problem.time, ics=conditions)
            else:
                solution = desolve_laplace(equation, unknown.function, ivar=problem.time, ics=conditions)
            solutions = {unknown.name: unknown.function == solution}
        else:
            first_order, functions = _first_order(problem)
            conditions = [0, *initial.tolist()] if use_initial_state else None
            try:
                found = desolve_system(first_order, functions, ics=conditions, ivar=problem.time)
            finally:
                _forget_initial_values(functions, problem.time)
            by_function = {str(item.lhs()): item for item in found}
            solutions = {unknown.name: by_function[str(unknown.function)] for unknown in unknowns}
    except TypeError as error:
        if 'positive, negative' in str(error):
            raise _sign_error(error) from error
        error_msg = (
            f'No closed form was found for the system, Maxima reports: {error}. Simulate it numerically instead.'
        )
        raise AnalysisError(error_msg) from error
    except (NotImplementedError, ValueError, RuntimeError) as error:
        error_msg = (
            f'No closed form was found for the system, Maxima reports: {error}. Simulate it numerically instead.'
        )
        raise AnalysisError(error_msg) from error
    functions = [unknown.function for unknown in unknowns]
    unsolved = [name for name, solution in solutions.items() if not _is_closed(solution.rhs(), functions, problem.time)]
    if unsolved:
        error_msg = (
            f'No closed form was found for {", ".join(unsolved)}: the solution contains unevaluated transforms, '
            'integrals, or the unknown functions themselves. Simulate the system numerically instead.'
        )
        raise AnalysisError(error_msg)
    return solutions
