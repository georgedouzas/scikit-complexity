"""Derive the equations of systems and solve them exactly with Sage."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Self

from sage.all__sagemath_symbolics import SR, Expression, desolve, desolve_laplace, desolve_system, diff, function
from sage.symbolic.function_factory import function as formal_function

from ..core.errors import AnalysisError, ConservationError
from ..engine.compile import compile_system, compiled_problem
from ..engine.problems import BaseProblem, Form, Problem, components_of
from ._assumptions import declare_domains
from ._base import BaseAnalysis, Source
from ._fractional import _mittag_leffler
from ._kinds import DataKind
from ._loops import _stock_flow
from ._quantum import _quantum_problem

KINDS = ('equations', 'solve', 'hamiltonian', 'conservation')
SETTINGS: dict[str, set[str]] = {
    'equations': set(),
    'solve': {'use_initial_state'},
    'hamiltonian': set(),
    'conservation': set(),
}
"""Settings every kind reads, besides `kind`."""
PURPOSES = {
    'equations': 'The equations of a system',
    'solve': 'Symbolic solving',
    'conservation': 'The conservation analysis',
}
UNEVALUATED = frozenset({'ilt', 'laplace', 'integrate', 'integral'})
CAPUTO = formal_function('caputo', nargs=3)


def _evaluate_mittag_leffler(
    self: Any,  # noqa: ANN401
    alpha: Any,  # noqa: ANN401
    beta: Any,  # noqa: ANN401
    z: Any,  # noqa: ANN401
    *,
    parent: Any = None,  # noqa: ANN401
    algorithm: Any = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Return the numeric value of the symbolic Mittag-Leffler function in the field of its arguments."""
    del self, algorithm
    if any(isinstance(item, Expression) and not item.is_numeric() for item in (alpha, beta, z)):
        error_msg = 'The Mittag-Leffler function has a numeric value only at numeric arguments.'
        raise TypeError(error_msg)
    value = complex(_mittag_leffler(complex(alpha).real, complex(beta).real, complex(z)))
    result = value.real if value.imag == 0 else value
    return result if parent is None else parent(result)


MITTAG_LEFFLER = formal_function('mittag_leffler', nargs=3, evalf_func=_evaluate_mittag_leffler)


def _exact(value: float) -> Expression:
    """Return a number as the exact rational of its shortest decimal form."""
    fraction = Fraction(repr(value))
    return SR(fraction.numerator) / fraction.denominator


def _components(problem: Problem) -> dict[str, tuple[str, int, int]]:
    """Return every equation name with its entry name, column, and flat index."""
    found = {}
    for entry in problem.states.entries:
        for column, name in enumerate(entry.generic_names):
            found[name] = (entry.name, column, entry.start + column)
    return found


def _delayed_functions(problem: Problem, substitution: dict[Expression, Expression]) -> dict[Expression, Expression]:
    """Return the substitution of every delayed state by its unknown function at the delayed time."""
    delayed = {}
    for delay in problem.delays:
        lag = SR(delay.delay).subs(substitution)
        delayed[delay.symbol] = SR(delay.state).subs(substitution).subs({problem.time: problem.time - lag})
    return delayed


@dataclass(frozen=True)
class _Unknown:
    """State component whose equation the analysis writes in its own order, with its function of time."""

    name: str
    function: Expression
    order: int
    indices: tuple[int, ...]


def _left_side(problem: Problem, unknown: _Unknown) -> Expression:
    """Return the left side of the equation of an unknown: its next value, its highest derivative, or its Caputo one."""
    if problem.form == Form.DISCRETE:
        return function(str(problem.equation_symbols[unknown.name]))(problem.time + 1)
    order = float(problem.orders[unknown.indices[0]]) if problem.orders.size else 1.0
    if order != 1:
        return CAPUTO(unknown.function, problem.time, _exact(order))
    return diff(unknown.function, problem.time, unknown.order)


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
    substitution.update(_delayed_functions(problem, substitution))
    equations = {}
    for unknown in unknowns:
        entry, column, _ = components[unknown.name]
        source = by_entry[(f'{entry}_d{unknown.order - 1}', column)] if unknown.order > 1 else unknown.name
        right = SR(problem.equations[source]).subs(substitution)
        equations[unknown.name] = _left_side(problem, unknown) == right
    return equations, unknowns, substitution


def _equations_of_problem(problem: Problem) -> dict[str, Expression]:
    """Return the equations of a compiled problem as relations between functions of time, in their written order."""
    if problem.form == Form.JUMP:
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
    for entry in problem.parameters.entries:
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
        f'{", ".join(names) or "involved"} with `ParameterSpec(domain=RealSet.open(0, oo))` or another domain that '
        'gives their sign.'
    )
    return AnalysisError(error_msg)


def _check_solvable(problem: Problem) -> None:
    """Raise an error when a problem is not of the ODE or fractional form, or has callable laws or groups."""
    if problem.form not in {Form.ODE, Form.FRACTIONAL}:
        error_msg = f'Only ODE systems can be solved symbolically, but the system is a {problem.form.value}.'
        raise AnalysisError(error_msg)
    if problem.has_callables:
        error_msg = 'The system has a callable law, which has no symbolic form.'
        raise AnalysisError(error_msg)
    if problem.has_group_sums or any(element.members is not None for element in problem.elements):
        error_msg = 'The system has element groups, which are not solved symbolically. Solve one member as an element.'
        raise AnalysisError(error_msg)


def _fractional_solution(problem: Problem, *, use_initial_state: bool) -> dict[str, Expression]:
    """Return the solution `x(t) = x(0) E_alpha(c t^alpha)` of one linear fractional relaxation `D^alpha x = c x`."""
    _, unknowns, _ = _written(problem)
    unknown = unknowns[0]
    item = problem.equation_symbols[unknown.name]
    right = SR(problem.equations[unknown.name])
    rate = right.diff(item)
    linear = len(unknowns) == 1 and not rate.has(item) and not rate.has(problem.time)
    if not (linear and bool((right - rate * item).expand() == 0)):
        error_msg = (
            'Only one linear fractional relaxation D^alpha x = c x, with c free of x and t, is solved exactly. '
            'Simulate the system with a fractional solver instead.'
        )
        raise AnalysisError(error_msg)
    index = unknown.indices[0]
    order = _exact(float(problem.orders[index]))
    start = (
        _exact(float(problem.initial_state[index])) if use_initial_state else unknown.function.subs({problem.time: 0})
    )
    return {unknown.name: unknown.function == start * MITTAG_LEFFLER(order, 1, rate * problem.time**order)}


def _solve(problem: Problem, *, use_initial_state: bool) -> dict[str, Expression]:
    """Return the exact solution of the equations of a problem as functions of time, with Maxima through Sage."""
    _check_solvable(problem)
    if problem.form == Form.FRACTIONAL:
        return _fractional_solution(problem, use_initial_state=use_initial_state)
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
            if solution.is_relational():
                error_msg = (
                    f'Maxima found only the implicit solution {solution} for {unknown.name}. Simulate the system '
                    'numerically instead.'
                )
                raise AnalysisError(error_msg)
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


def _conservation(problem: Problem) -> dict[str, Expression]:
    """Return the net inflow from the environment of every quantity, after checking that internal flows cancel."""
    stock_flow = _stock_flow(problem)
    amounts = {record.path: record.expression for record in stock_flow.recorded}
    net: dict[str, Expression] = defaultdict(lambda: SR(0))
    boundary: dict[str, Expression] = defaultdict(lambda: SR(0))
    for flow in stock_flow.flows:
        if flow.quantity is None:
            continue
        amount = SR(amounts[flow.path])
        net[flow.quantity] += (flow.sink is not None) * amount - (flow.source is not None) * amount
        boundary[flow.quantity] += (flow.source is None) * amount - (flow.sink is None) * amount
    for quantity, value in net.items():
        if not bool((value - boundary[quantity]).simplify_full().is_trivial_zero()):
            error_msg = f'The internal flows of {quantity!r} do not cancel: the total changes by {value}.'
            raise ConservationError(error_msg)
    return {quantity: boundary[quantity] for quantity in stock_flow.quantities if quantity in boundary} | {
        quantity: SR(0) for quantity in stock_flow.quantities if quantity not in boundary
    }


class SymbolicAnalysis(BaseAnalysis):
    """Symbolic form of a system: its equations, their exact solution, its Hamiltonian, or its boundary flows.

    The kinds read the equations of a system:

    - `'equations'`: one equation per state component, a Sage relation. Each state component is a function of time
      `t`. Continuous forms give its time derivative, the drift for stochastic systems, and discrete maps give its
      value at `t + 1`. States of higher order give their highest derivative, `diff(x(t), t, k)`, with their
      companions written as derivatives, and lagged states read their past values as `x(t - j)`. The equations of a
      group hold for every member and are named with the member index `i`. Components driven by a callable law or by
      a sum over members or pairs are left out.
    - `'solve'`: the exact solution of the equations of an ODE system as functions of time, with Maxima through Sage.
      One equation of order 1 or 2 is solved with `desolve`, one of higher order with `desolve_laplace`, and several
      with `desolve_system` after reduction to first order, which solves linear systems with constant coefficients.
      A result that still contains unevaluated transforms, integrals, or the unknown functions is not a solution, and
      raises the error. Parameters give Maxima their sign from their domain, or from their value when their domain is
      real. One linear fractional relaxation `D^alpha x = c x` has the solution `x(0) E_alpha(c t^alpha)`, written
      with the symbolic function `mittag_leffler(alpha, beta, z)`.
    - `'hamiltonian'`: the Hamiltonian of a quantum system, the sum of the operators of its generator laws, whose
      subsystems are the symbols of the states of the elements, named `<state>_<path>`, and whose parameters are
      named `<parameter>_<path>`.
    - `'conservation'`: the net inflow from the environment of every quantity of a system of stocks and flows, after
      checking that its internal flows cancel, so the total changes only through the boundary.

    Args:
        kind:
            The symbolic form to derive, one of the kinds above.

        use_initial_state:
            Whether `'solve'` uses the initial state of the system, with the initial values of every derivative of
            states of higher order, as initial conditions at `t = 0`. Without them, the solution keeps its integration
            constants.

    Attributes:
        problem_:
            The compiled problem.

        equations_:
            A mapping from state component name to its equation.

        solutions_:
            A mapping from state component name to its solution, a relation `f(t) == solution`. States of higher order
            appear once, without their derivatives.

        hamiltonian_:
            The Hamiltonian, a Sage expression of local operators.

        inflows_:
            The symbolic rate, or amount per step, at which each quantity enters the system from the environment.

    Raises:
        AnalysisError:
            The system has no form of the kind, Maxima cannot solve its equations, or needs the sign of a parameter.

        ConservationError:
            The internal flows of a quantity do not cancel.

    Examples:
        >>> from skcomplex.analysis import SymbolicAnalysis
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationElement
        >>> system = System(
        ...     elements=[PopulationElement('a', n=2)],
        ...     interactions=[ExponentialGrowthInteraction('g', target='a', r=0.5)],
        ... )
        >>> SymbolicAnalysis().check().prepare(system).run().equations_['a.n']
        diff(n_a(t), t) == r_g*n_a(t)
        >>> SymbolicAnalysis('solve', use_initial_state=True).check().prepare(system).run().solutions_['a.n']
        n_a(t) == 2*e^(r_g*t)
    """

    def __init__(self, kind: str = 'equations', *, use_initial_state: bool = False) -> None:
        self.kind = kind
        self.use_initial_state = use_initial_state

    def _check(self) -> Self:
        """Validate the kind and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind])
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the system kind."""
        return DataKind.SYSTEM

    def _prepare(self, source: Source) -> Self:
        """Compile the system of the source, checking that it has the form of the kind."""
        system = self._read_system(source)
        problem = system if isinstance(system, BaseProblem) else compile_system(system)
        if self.kind == 'hamiltonian':
            self.problem_ = _quantum_problem(problem)
            return self
        if self.kind == 'equations' and problem.form == Form.QUANTUM:
            error_msg = (
                'A quantum system has an operator Hamiltonian, not state equations. Use '
                "`SymbolicAnalysis('hamiltonian')`."
            )
            raise AnalysisError(error_msg)
        self.problem_ = compiled_problem(problem, PURPOSES[self.kind])
        return self

    def _run(self) -> Self:
        """Derive the equations, their solution, the Hamiltonian, or the boundary flows."""
        if self.kind == 'equations':
            self.equations_ = _equations_of_problem(self.problem_)
        elif self.kind == 'solve':
            self.solutions_ = _solve(self.problem_, use_initial_state=self.use_initial_state)
        elif self.kind == 'hamiltonian':
            self.hamiltonian_ = self.problem_.hamiltonian_operator
        else:
            self.inflows_ = _conservation(self.problem_)
        return self
