"""Derive the equations of systems and solve them exactly with Sage."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import desolve, desolve_system, diff, function

from ..compile import Problem, compile_system
from ..core import AnalysisError, Expression, Form, declare_domains
from ..model import System


def _functions_of_time(problem: Problem) -> dict[Expression, Expression]:
    """Return one function of time per state component symbol."""
    return {item: function(str(item))(problem.time) for item in problem.equation_symbols.values()}


def _equations_of_problem(problem: Problem) -> dict[str, Expression]:
    """Return the equations of a compiled problem as relations between functions of time."""
    if problem.form is Form.JUMP:
        error_msg = 'A jump process has no symbolic equations. Use a deterministic version of its interactions.'
        raise AnalysisError(error_msg)
    functions = _functions_of_time(problem)
    equations = {}
    for name, item in problem.equation_symbols.items():
        right = problem.equations[name].subs(functions)
        left = (
            function(str(item))(problem.time + 1)
            if problem.form is Form.DISCRETE
            else diff(functions[item], problem.time)
        )
        equations[name] = left == right
    return equations


def _forget_initial_values(functions: list[Expression], time: Expression) -> None:
    """Reset the values at `t = 0` that `desolve_system` leaves in Maxima for every function but the first."""
    for item in functions:
        maxima_function = item._maxima_()
        maxima_function.atvalue(time == 0, maxima_function)


def equations_of(system: System) -> dict[str, Expression]:
    """Return the symbolic equations of a system, one per state component.

    Each state component is a function of time `t`. Continuous forms give its time derivative, the drift for
    stochastic systems, and discrete maps give its value at `t + 1`. Every equation is a Sage relation. The equations
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

    One equation is solved with `desolve`, and several with `desolve_system`, which solves linear systems with
    constant coefficients.

    Args:
        system:
            The system.

        use_initial_state:
            Whether to use the initial state of the system as initial conditions at `t = 0`. Without them, the
            solution keeps its integration constants, which are the values of the functions at `t = 0`, such as
            `n_a(0)`, or a constant `_C` for one equation.

    Returns:
        solutions:
            A mapping from state component name to its solution, a relation `f(t) == solution`.

    Raises:
        AnalysisError:
            The system is not an ODE, has a callable law or groups, or has equations Maxima cannot solve.

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
    equations = _equations_of_problem(problem)
    functions = list(_functions_of_time(problem).values())
    initial_values = problem.initial_state.tolist() if use_initial_state else None
    try:
        declare_domains([*problem.state_symbols, *problem.parameter_symbols, problem.time])
        if len(functions) == 1:
            initial_conditions = [0, initial_values[0]] if initial_values is not None else None
            solution = desolve(next(iter(equations.values())), functions[0], ivar=problem.time, ics=initial_conditions)
            solutions = [functions[0] == solution]
        else:
            initial_conditions = [0, *initial_values] if initial_values is not None else None
            try:
                solutions = desolve_system(
                    list(equations.values()),
                    functions,
                    ics=initial_conditions,
                    ivar=problem.time,
                )
            finally:
                _forget_initial_values(functions, problem.time)
    except (NotImplementedError, ValueError, TypeError, RuntimeError) as error:
        error_msg = (
            f'Maxima cannot solve the equations of the system: {error}. Simulate the system numerically instead.'
        )
        raise AnalysisError(error_msg) from error
    by_function = {str(solution.lhs()): solution for solution in solutions}
    return {name: by_function[str(item)] for name, item in zip(equations, functions, strict=True)}
