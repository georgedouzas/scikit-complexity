"""Derive the equations of systems and solve them exactly with Sage."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cache

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import (
    RR,
    SR,
    ZZ,
    ComplexField,
    Expression,
    RealField,
    desolve,
    desolve_laplace,
    desolve_system,
    diff,
    function,
)
from sage.rings.real_mpfr import RealNumber
from sage.structure.element import Element
from sage.structure.parent import Parent
from sage.symbolic.assumptions import assume, forget
from sage.symbolic.function import Function
from sage.symbolic.function_factory import function as formal_function

from ..engine._utils import components_of, domain_of
from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems import BaseProblemSpec, ProblemSpec
from ..engine.problems._operator_problem_spec import OperatorProblemSpec
from ..engine.problems._transfer_network_spec import TransferNetworkSpec
from ..errors import AnalysisError, ConservationError
from ..specs import AnalysisTags
from ..specs._unknown_spec import _UnknownSpec
from ..types import Array, ComplexArray, DataKind, FormKind
from ._base import _check_choice, _check_settings, _compiled_problem, _composite, _read_system
from ._operator import _operator_problem
from .types import Source

ASYMPTOTIC_ORDER = 2
DESOLVE_ORDER = 2

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
ASYMPTOTIC_SCALE = 50.0
LARGEST_SCALE = 2000.0
GUARD_BITS = 64
DECLARED: set[str] = set()
SIGNS: dict[str, Expression] = {}

TAGS = AnalysisTags(DataKind.SYSTEM)


def _evaluate_mittag_leffler(
    self: Function,
    alpha: Element | complex,
    beta: Element | complex,
    z: Element | complex,
    *,
    parent: Parent | None = None,
    algorithm: str | None = None,
) -> Element | complex | float:
    """Return the numeric value of the symbolic Mittag-Leffler function in the field of its arguments."""
    del self, algorithm
    if any(isinstance(item, Expression) and not item.is_numeric() for item in (alpha, beta, z)):
        error_msg = 'The Mittag-Leffler function has a numeric value only at numeric arguments.'
        raise TypeError(error_msg)
    value = complex(_mittag_leffler(complex(alpha).real, complex(beta).real, complex(z)))
    result = value.real if value.imag == 0 else value
    return result if parent is None else parent(result)


MITTAG_LEFFLER = formal_function('mittag_leffler', nargs=3, evalf_func=_evaluate_mittag_leffler)


@dataclass(frozen=True)
class SymbolicResult:
    """Symbolic form of a system: its equations, their exact solution, its Hamiltonian, or its boundary flows.

    Args:
        problem:
            The compiled problem.

        equations:
            A mapping from state component name to its equation, set for `'equations'`.

        solutions:
            A mapping from state component name to its solution, a relation `f(t) == solution`, set for `'solve'`.
            States of higher order appear once, without their derivatives.

        hamiltonian:
            The Hamiltonian, a Sage expression of local operators, set for `'hamiltonian'`.

        inflows:
            The symbolic rate, or amount per step, at which each quantity enters the system from the environment, set
            for `'conservation'`.
    """

    problem: ProblemSpec | OperatorProblemSpec
    equations: dict[str, Expression] | None = None
    solutions: dict[str, Expression] | None = None
    hamiltonian: Expression | None = None
    inflows: dict[str, Expression] | None = None


def symbolic(source: Source, *, kind: str = 'equations', use_initial_state: bool = False) -> SymbolicResult:
    """Return the symbolic form of a system: its equations, their solution, its Hamiltonian, or its boundary flows.

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
    - `'hamiltonian'`: the Hamiltonian of an operator system, the sum of the operators of its generator laws, whose
      subsystems are the symbols of the states of the elements, named `<state>_<path>`, and whose parameters are
      named `<parameter>_<path>`.
    - `'conservation'`: the net inflow from the environment of every quantity of a system with a transfer network, after
      checking that its internal flows cancel, so the total changes only through the boundary.

    Args:
        source:
            The system, its compiled problem, or a trajectory of it.

        kind:
            The symbolic form to derive, one of the kinds above.

        use_initial_state:
            Whether `'solve'` uses the initial state of the system, with the initial values of every derivative of
            states of higher order, as initial conditions at `t = 0`. Without them, the solution keeps its integration
            constants.

    Returns:
        result:
            The compiled problem and the symbolic form of the kind.

    Raises:
        AnalysisError:
            The system has no form of the kind, Maxima cannot solve its equations, or needs the sign of a parameter.

        ConservationError:
            The internal flows of a quantity do not cancel.

    Examples:
        >>> from skcomplex.core.analysis import symbolic
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
        >>> system = System(
        ...     elements=[PopulationElement('a', n=2)],
        ...     interactions=[ExponentialGrowthInteraction('g', target='a', r=0.5)],
        ... )
        >>> symbolic(system).equations['a.n']
        diff(n_a(t), t) == r_g*n_a(t)
        >>> symbolic(system, kind='solve', use_initial_state=True).solutions['a.n']
        n_a(t) == 2*e^(r_g*t)
    """
    source = _composite(source)
    _check_choice('symbolic', 'kind', kind, KINDS)
    _check_settings(symbolic, 'kind', SETTINGS[kind], kind=kind, use_initial_state=use_initial_state)
    system = _read_system(source)
    compiled = system if isinstance(system, BaseProblemSpec) else PassCompiler().prepare(system).run().problem_
    if kind == 'hamiltonian':
        operators = _operator_problem(compiled)
        return SymbolicResult(operators, hamiltonian=operators.generator_operator)
    if kind == 'equations' and compiled.form == FormKind.OPERATOR:
        error_msg = (
            'An operator system has an operator Hamiltonian, not state equations. Use '
            "`symbolic(system, kind='hamiltonian')`."
        )
        raise AnalysisError(error_msg)
    problem = _compiled_problem(compiled, PURPOSES[kind])
    if kind == 'equations':
        return SymbolicResult(problem, equations=_equations_of_problem(problem))
    if kind == 'solve':
        return SymbolicResult(problem, solutions=_solve(problem, use_initial_state=use_initial_state))
    return SymbolicResult(problem, inflows=_conservation(problem))


def _declare_domains(items: Iterable[Expression], values: Mapping[Expression, float] | None = None) -> None:
    """Give Maxima the domains of symbols, once per symbol, so exact solutions use them.

    A symbol of a real domain with a nonzero value also gets the sign of its value, which replaces the sign of an
    earlier value.

    Args:
        items:
            The symbols.

        values:
            Current values of symbols whose sign Maxima takes when their domain is real.

    Raises:
        ValueError:
            A domain contradicts an assumption Maxima already holds.
    """
    for item in items:
        name = str(item)
        domain = domain_of(item)
        if name not in DECLARED:
            if domain is ZZ:
                assume(item, 'integer')
            elif domain is RR:
                assume(item, 'real')
            DECLARED.add(name)
        value = (values or {}).get(item)
        if domain is not RR or not value:
            continue
        sign = item > 0 if value > 0 else item < 0
        previous = SIGNS.get(name)
        if previous is not None and str(previous) == str(sign):
            continue
        if previous is not None:
            forget(previous)
        assume(sign)
        SIGNS[name] = sign


def _reciprocal_gamma(argument: RealNumber) -> RealNumber:
    """Return `1 / Gamma(argument)` of a real number of a Sage field, which is 0 at the poles of gamma."""
    if argument <= 0 and argument == argument.round():
        return argument.parent()(0)
    return 1 / argument.gamma()


@cache
def _mittag_leffler_coefficients(alpha: float, beta: float, bits: int, count: int) -> tuple[RealNumber, ...]:
    """Return the coefficients `1 / Gamma(alpha k + beta)` of the power series for `k < count` in a Sage field."""
    reals = RealField(bits)
    a, b = reals(alpha), reals(beta)
    return tuple(_reciprocal_gamma(a * index + b) for index in range(count))


def _mittag_leffler_series(alpha: float, beta: float, z: complex, scale: float) -> complex:
    """Return the Mittag-Leffler function by its power series, summed precisely enough for the cancellations."""
    extra = math.ceil(scale / math.log(2) + 2 * math.log2(1 + abs(z)))
    bits = 53 + GUARD_BITS * (1 + extra // GUARD_BITS)
    complexes = ComplexField(bits)
    w = complexes(z.real, z.imag)
    total, power = complexes(0), complexes(1)
    tolerance = RealField(bits)(2) ** -(53 + 10)
    count = 2 ** math.ceil(math.log2(scale / alpha + 16))
    coefficients = _mittag_leffler_coefficients(alpha, beta, bits, count)
    index = 0
    while True:
        if index == len(coefficients):
            coefficients = _mittag_leffler_coefficients(alpha, beta, bits, 2 * len(coefficients))
        term = power * coefficients[index]
        total += term
        if index > scale / alpha + 2 and abs(term) <= tolerance * abs(total):
            return complex(total)
        power *= w
        index += 1


def _mittag_leffler_asymptotic(alpha: float, beta: float, z: complex, scale: float) -> complex:
    """Return the Mittag-Leffler function by its asymptotic expansion for large `|z|`, with `0 < alpha < 2`.

    The expansion `E(z) = (1 / alpha) z^((1 - beta) / alpha) exp(z^(1 / alpha)) - sum_k z^(-k) / Gamma(beta - alpha
    k)` holds for `|arg z| < alpha pi`, and the exponential term drops out of it for larger arguments. The sum stops
    where the bound `|z|^(-k) Gamma(1 + alpha k - beta) / pi` of its terms falls below the precision or starts to
    grow.
    """
    reals, complexes = RealField(53 + GUARD_BITS), ComplexField(53 + GUARD_BITS)
    a, b, w = reals(alpha), reals(beta), complexes(z.real, z.imag)
    tolerance = reals(2) ** -(53 + 10)
    total = complexes(0)
    if abs(w.arg()) < a * reals.pi():
        total += w ** ((1 - b) / a) * (w ** (1 / a)).exp() / a
    for index in range(1, math.floor(scale) + 1):
        total -= w ** (-index) * _reciprocal_gamma(b - a * index)
        bound = abs(w) ** (-index) * (1 + a * index - b).gamma() / reals.pi()
        if bound <= tolerance * abs(total):
            break
    return complex(total)


def _mittag_leffler_scalar(alpha: float, beta: float, z: complex) -> complex:
    """Return the Mittag-Leffler function at one point by its series or its asymptotic expansion."""
    if alpha == 1 and beta == 1:
        return complex(np.exp(z))
    scale = abs(z) ** (1 / alpha)
    if scale <= ASYMPTOTIC_SCALE:
        return _mittag_leffler_series(alpha, beta, z, scale)
    if alpha < ASYMPTOTIC_ORDER:
        return _mittag_leffler_asymptotic(alpha, beta, z, scale)
    if scale <= LARGEST_SCALE:
        return _mittag_leffler_series(alpha, beta, z, scale)
    error_msg = (
        f'E_(alpha, beta)(z) with alpha = {alpha} >= 2 is evaluated for |z|^(1/alpha) <= {LARGEST_SCALE}, but got '
        f'|z| = {abs(z)}. Rescale the argument.'
    )
    raise AnalysisError(error_msg)


def _mittag_leffler(alpha: float, beta: float, z: npt.ArrayLike) -> Array | ComplexArray:
    """Return the two-parameter Mittag-Leffler function `E_(alpha, beta)(z) = sum_k z^k / Gamma(alpha k + beta)`.

    `E_(1, 1)(z)` is `exp(z)`, and `x_0 E_(alpha, 1)(-lambda t^alpha)` solves the fractional relaxation `D^alpha x =
    -lambda x` of a Caputo derivative of order `alpha`, which decays as the power law `t^(-alpha)` at long times.
    The function sums its power series in arbitrary precision with Sage for `|z|^(1/alpha) <= 50`, and takes its
    asymptotic expansion beyond.

    Args:
        alpha:
            The positive first parameter, the order of the derivative.

        beta:
            The real second parameter.

        z:
            The real or complex arguments.

    Returns:
        values:
            The values with the shape of `z`, real for real arguments.

    Raises:
        AnalysisError:
            `alpha` is not positive, or the argument is too large for `alpha >= 2`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis._symbolic import _mittag_leffler
        >>> bool(np.isclose(_mittag_leffler(1.0, 1.0, -2.0), np.exp(-2.0)))
        True
        >>> round(float(_mittag_leffler(0.5, 1.0, -1.0)), 6)
        0.427584
    """
    if not alpha > 0:
        error_msg = f'alpha = {alpha!r} must be positive.'
        raise AnalysisError(error_msg)
    values = np.asarray(z)
    flat = [_mittag_leffler_scalar(float(alpha), float(beta), complex(item)) for item in values.ravel()]
    result = np.asarray(flat, dtype=complex).reshape(values.shape)
    return result if np.iscomplexobj(values) else result.real


def _exact(value: float) -> Expression:
    """Return a number as the exact rational of its shortest decimal form."""
    fraction = Fraction(repr(value))
    return SR(fraction.numerator) / fraction.denominator


def _components(problem: ProblemSpec) -> dict[str, tuple[str, int, int]]:
    """Return every equation name with its entry name, column, and flat index."""
    found = {}
    for entry in problem.states.entries:
        for column, name in enumerate(entry.generic_names):
            found[name] = (entry.name, column, entry.start + column)
    return found


def _delayed_functions(
    problem: ProblemSpec,
    substitution: dict[Expression, Expression],
) -> dict[Expression, Expression]:
    """Return the substitution of every delayed state by its unknown function at the delayed time."""
    delayed = {}
    for delay in problem.delays:
        lag = SR(delay.delay).subs(substitution)
        delayed[delay.symbol] = SR(delay.state).subs(substitution).subs({problem.time: problem.time - lag})
    return delayed


def _left_side(problem: ProblemSpec, unknown: _UnknownSpec) -> Expression:
    """Return the left side of the equation of an unknown: its next value, highest derivative, or Caputo one."""
    if problem.form == FormKind.DISCRETE:
        return function(str(problem.equation_symbols[unknown.name]))(problem.time + 1)
    order = float(problem.orders[unknown.indices[0]]) if problem.orders.size else 1.0
    if order != 1:
        return CAPUTO(unknown.function, problem.time, _exact(order))
    return diff(unknown.function, problem.time, unknown.order)


def _written(
    problem: ProblemSpec,
) -> tuple[dict[str, Expression], list[_UnknownSpec], dict[Expression, Expression]]:
    """Return the equations in their written order, the unknown functions, and the substitution of symbols."""
    components = _components(problem)
    by_entry = {(entry, column): name for name, (entry, column, _) in components.items()}
    orders: dict[str, int] = {}
    for kind, base, index in problem.companions.values():
        if kind == 'derivative':
            orders[base] = max(orders.get(base, 1), index + 1)
    unknowns: list[_UnknownSpec] = []
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
            if companion not in problem.equation_symbols:
                error_msg = f'{entry} has its derivatives computed numerically, so it has no symbolic equations.'
                raise AnalysisError(error_msg)
            substitution[problem.equation_symbols[companion]] = diff(unknown_function, problem.time, degree)
            indices.append(components[companion][2])
        for companion_name, (kind, base, steps) in problem.companions.items():
            if kind == 'lag' and base == entry:
                lagged = by_entry[(companion_name, column)]
                substitution[problem.equation_symbols[lagged]] = function(str(item))(problem.time - steps)
        unknowns.append(_UnknownSpec(name, unknown_function, order, tuple(indices)))
    substitution.update(_delayed_functions(problem, substitution))
    equations = {}
    for unknown in unknowns:
        entry, column, _ = components[unknown.name]
        source = by_entry[(f'{entry}_d{unknown.order - 1}', column)] if unknown.order > 1 else unknown.name
        right = SR(problem.equations[source]).subs(substitution)
        equations[unknown.name] = _left_side(problem, unknown) == right
    return equations, unknowns, substitution


def _equations_of_problem(problem: ProblemSpec) -> dict[str, Expression]:
    """Return the equations of a compiled problem as relations between functions of time, in their written order."""
    if problem.form == FormKind.JUMP:
        error_msg = 'A jump process has no symbolic equations. Use a deterministic version of its interactions.'
        raise AnalysisError(error_msg)
    return _written(problem)[0]


def _first_order(problem: ProblemSpec) -> tuple[list[Expression], list[Expression]]:
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


def _parameter_values(problem: ProblemSpec) -> dict[Expression, float]:
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


def _check_solvable(problem: ProblemSpec) -> None:
    """Raise an error when a problem is not of the ODE or fractional form, or has callable laws or groups."""
    if problem.form not in {FormKind.ODE, FormKind.FRACTIONAL}:
        error_msg = f'Only ODE systems can be solved symbolically, but the system is a {problem.form}.'
        raise AnalysisError(error_msg)
    if problem.has_callables:
        error_msg = 'The system has a callable law, which has no symbolic form.'
        raise AnalysisError(error_msg)
    if problem.has_group_sums or any(element.members is not None for element in problem.elements):
        error_msg = 'The system has element groups, which are not solved symbolically. Solve one member as an element.'
        raise AnalysisError(error_msg)


def _fractional_solution(problem: ProblemSpec, *, use_initial_state: bool) -> dict[str, Expression]:
    """Return the solution `x(t) = x(0) E_alpha(c t^alpha)` of a linear fractional relaxation `D^alpha x = c x`."""
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


def _solve(problem: ProblemSpec, *, use_initial_state: bool) -> dict[str, Expression]:
    """Return the exact solution of the equations of a problem as functions of time, with Maxima through Sage."""
    _check_solvable(problem)
    if problem.form == FormKind.FRACTIONAL:
        return _fractional_solution(problem, use_initial_state=use_initial_state)
    equations, unknowns, _ = _written(problem)
    initial = problem.initial_state
    try:
        _declare_domains(
            [*problem.state_symbols, *problem.parameter_symbols, problem.time],
            _parameter_values(problem),
        )
        if len(unknowns) == 1:
            unknown = unknowns[0]
            conditions = [0, *(float(initial[index]) for index in unknown.indices)] if use_initial_state else None
            equation = next(iter(equations.values()))
            if unknown.order <= DESOLVE_ORDER:
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


def _transfer_network(problem: ProblemSpec) -> TransferNetworkSpec:
    """Return the transfer network of a problem, or raise an error for problems without one."""
    if problem.transfer_network is None:
        error_msg = 'The problem has no transfer network. Compile the system as one partition with `PassCompiler`.'
        raise AnalysisError(error_msg)
    return problem.transfer_network


def _conservation(problem: ProblemSpec) -> dict[str, Expression]:
    """Return the net inflow from the environment of every quantity, after checking that internal flows cancel."""
    transfer_network = _transfer_network(problem)
    amounts = {record.path: record.expression for record in transfer_network.recorded}
    net: dict[str, Expression] = defaultdict(lambda: SR(0))
    boundary: dict[str, Expression] = defaultdict(lambda: SR(0))
    for flow in transfer_network.flows:
        if flow.quantity is None:
            continue
        amount = SR(amounts[flow.path])
        net[flow.quantity] += (flow.sink is not None) * amount - (flow.source is not None) * amount
        boundary[flow.quantity] += (flow.source is None) * amount - (flow.sink is None) * amount
    for quantity, value in net.items():
        if not bool((value - boundary[quantity]).simplify_full().is_trivial_zero()):
            error_msg = f'The internal flows of {quantity!r} do not cancel: the total changes by {value}.'
            raise ConservationError(error_msg)
    return {quantity: boundary[quantity] for quantity in transfer_network.quantities if quantity in boundary} | {
        quantity: SR(0) for quantity in transfer_network.quantities if quantity not in boundary
    }
