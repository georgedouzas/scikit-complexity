"""Derive the equations of systems and solve them exactly with SymPy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sympy

from ..compile import Problem, compile_system
from ..core import AnalysisError, Form
from ..model import System


def _functions_of_time(problem: Problem) -> dict[sympy.Symbol, sympy.Expr]:
    """Return one function of time per state component symbol."""
    return {symbol: sympy.Function(symbol.name)(problem.time) for symbol in problem.equation_symbols.values()}


def _equations_of_problem(problem: Problem) -> dict[str, sympy.Eq]:
    """Return the equations of a compiled problem as functions of time."""
    if problem.form is Form.JUMP:
        error_msg = 'A jump process has no symbolic equations. Use a deterministic version of its interactions.'
        raise AnalysisError(error_msg)
    functions = _functions_of_time(problem)
    equations = {}
    for name, symbol in problem.equation_symbols.items():
        right = problem.equations[name].subs(functions)
        function = functions[symbol]
        left = (
            function.subs(problem.time, problem.time + 1)
            if problem.form is Form.DISCRETE
            else sympy.Derivative(function, problem.time)
        )
        equations[name] = sympy.Eq(left, right)
    return equations


def equations_of(system: System) -> dict[str, sympy.Eq]:
    """Return the symbolic equations of a system, one per state component.

    Each state component is a function of time `t`. Continuous forms give its time derivative, the drift for
    stochastic systems, and discrete maps give its value at `t + 1`. The equations of a group hold for every member and
    are named with the member index `i`. Components driven by a callable law or by a sum over members or pairs are left
    out.

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
        Eq(Derivative(n_rabbits(t), t), r_growth*(1 - n_rabbits(t)/K_growth)*n_rabbits(t))
    """
    return _equations_of_problem(compile_system(system))


def solve_symbolically(system: System, *, use_initial_state: bool = False) -> dict[str, sympy.Eq]:
    """Solve the equations of an ODE system exactly as functions of time.

    Args:
        system:
            The system.

        use_initial_state:
            Whether to use the initial state of the system as initial conditions at `t = 0`. Without them, the
            solution keeps its integration constants.

    Returns:
        solutions:
            A mapping from state component name to its solution.

    Raises:
        AnalysisError:
            The system is not an ODE, has a callable law or groups, or has equations SymPy cannot solve.

    Examples:
        >>> from skcomplex.analysis import solve_symbolically
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> system = System(
        ...     elements=[Population('a', n=2)],
        ...     interactions=[ExponentialGrowth('g', target='a', r=0.5)],
        ... )
        >>> solve_symbolically(system, use_initial_state=True)['a.n']
        Eq(n_a(t), 2.0*exp(r_g*t))
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
    initial_conditions = (
        {
            function.subs(problem.time, 0): value
            for function, value in zip(functions, problem.initial_state.tolist(), strict=True)
        }
        if use_initial_state
        else None
    )
    try:
        solutions = sympy.dsolve(list(equations.values()), functions, ics=initial_conditions)
    except (NotImplementedError, ValueError) as error:
        error_msg = f'SymPy cannot solve the equations of the system: {error}'
        raise AnalysisError(error_msg) from error
    solutions = solutions if isinstance(solutions, list) else [solutions]
    by_function = {solution.lhs: solution for solution in solutions}
    return {name: by_function[function] for name, function in zip(equations, functions, strict=True)}
