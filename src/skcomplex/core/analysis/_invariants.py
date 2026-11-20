"""Derive the conserved quantities of a system with Sage."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ..components import Observable
from ..engine.compile._observables import ObservableEvaluator
from ..engine.problems import ProblemSpec
from ..engine.problems._composite_system import _CompositeSystem
from ..engine.problems._dynamics_spec import _DynamicsSpec
from ..enums import DataKind
from ..errors import AnalysisError
from ..systems import System
from ._analysis_tags import AnalysisTags
from ._base import _get_system, _read_system
from ._stability import StabilityResult
from .types import DataSource

TOLERANCE = 1e-9
TAGS = AnalysisTags(DataKind.SYSTEM)


@dataclass(frozen=True)
class InvariantsResult:
    """A result of `invariants`, the rates of expressions and if they are conserved.

    Args:
        problem:
            The compiled problem.

        system:
            The `System` whose expressions were checked, the one a ready-made system builds.

        dynamics:
            The first-order equations of the system, with the values of its parameters.

        namespace:
            The namespace of the symbols of the states and parameters of the system, where `s.<element>.<state>` is
            the symbol of a state.

        rates:
            A mapping from every label to the rate of change of its expression.

        conserved:
            A mapping from every label to whether its expression is conserved.

        methods:
            A mapping from every label to the method that decided, `'symbolic'` or `'numeric'`.
    """

    problem: ProblemSpec
    system: System
    dynamics: _DynamicsSpec
    namespace: SimpleNamespace
    rates: dict[str, Expression]
    conserved: dict[str, bool]
    methods: dict[str, str]


def invariants(
    source: DataSource,
    *,
    expressions: Sequence[Observable] = (),
    samples: int = 16,
    random_state: int | np.random.Generator | None = None,
) -> InvariantsResult:
    """Return the rates of expressions of a system and whether they are conserved.

    The rate `df/dt` of an expression `f` is its derivative along the equations of the system, `diff(f, t) + sum
    df/dx x'`. An expression is conserved when its rate simplifies to zero, or, when simplification does not decide,
    when the rate vanishes within a tolerance at random states with the parameters at their values. The expressions
    are `Observable`s whose expression is a function of the namespace of the system, where `s.<element>.<state>` is the
    symbol of a state, or a symbolic expression of those symbols.

    It appears in many fields. It finds the conserved energy of a pendulum, the conserved total of an epidemic, and the
    conserved money of an exchange economy.

    Args:
        source:
            The system, or a trajectory of it.

        expressions:
            The expressions to check.

        samples:
            The number of random states of the numeric check.

        random_state:
            The seed or generator of the random states.

    Returns:
        result:
            The rates of the expressions and whether they are conserved.

    Raises:
        AnalysisError:
            The source is not a system, the system is not an ODE, or an expression or the number of samples is
            invalid.

    Examples:
        >>> from skcomplex.core.analysis import invariants
        >>> from skcomplex.core.components import Observable
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.systems import System
        >>> def swing(s):
        ...     return [RateLaw(s.target.q, s.target.p), RateLaw(s.target.p, -s.target.q)]
        >>> spring = Interaction('spring', roles={'target': 'mass'}, laws=swing)
        >>> system = System(elements=[Element('mass', values={'q': 1.0, 'p': 0.0})], interactions=[spring])
        >>> expressions = [
        ...     Observable('energy', lambda s: s.mass.q**2 + s.mass.p**2),
        ...     Observable('position', lambda s: s.mass.q),
        ... ]
        >>> result = invariants(system, expressions=expressions)
        >>> result.conserved
        {'energy': True, 'position': False}
        >>> result.rates['position']
        p_mass
    """
    _check_inputs(expressions, samples)
    system = _get_system(source)
    declared = _read_system(system)
    if not isinstance(system, System) or not isinstance(declared, _CompositeSystem):
        error_msg = (
            f'Invariants are derived from the interactions of a system, not from {type(declared).__name__}. Give the '
            'system.'
        )
        raise AnalysisError(error_msg)
    dynamics = StabilityResult._get_dynamics_of(declared)
    namespace = ObservableEvaluator._get_namespace_of(dynamics.problem)
    rng = np.random.default_rng(random_state)
    rates, conserved, methods = {}, {}, {}
    for expression in expressions:
        value = _evaluate(namespace, expression)
        terms = [value.diff(SR(dynamics.problem.time))]
        terms += [value.diff(item) * rate for item, rate in zip(dynamics.symbols, dynamics.rates, strict=True)]
        rate, is_conserved, method = _decide(dynamics, terms, rng, samples)
        rates[expression.label] = rate
        conserved[expression.label], methods[expression.label] = is_conserved, method
    return InvariantsResult(dynamics.problem, system, dynamics, namespace, rates, conserved, methods)


def _check_inputs(expressions: Sequence[Observable], samples: int) -> None:
    """Raise an error for a foreign expression or a number of samples below one."""
    for expression in expressions:
        if not isinstance(expression, Observable) or expression.group is not None:
            error_msg = f'{expression!r} is not an expression of the system. Give an Observable without group.'
            raise AnalysisError(error_msg)
    if not isinstance(samples, int) or samples < 1:
        error_msg = f'samples = {samples!r} must be a positive integer.'
        raise AnalysisError(error_msg)


def _resolve(namespace: SimpleNamespace, path: str) -> object:
    """Return the namespace of the element or state at a path of the system.

    An error is raised when there is none.
    """
    value: object = namespace
    for part in path.replace('/', '.').split('.'):
        if not hasattr(value, part):
            error_msg = f'The system has no element or state at {path!r}. Give the path of a state, such as `a.x`.'
            raise AnalysisError(error_msg)
        value = getattr(value, part)
    return value


def _evaluate(namespace: SimpleNamespace, expression: Observable) -> Expression:
    """Return the expression of an observable in the symbols of the system."""
    value = expression.expression
    scope = namespace if expression.element is None else _resolve(namespace, expression.element)
    return SR(value(scope) if callable(value) else value)


def _vanishes(dynamics: _DynamicsSpec, terms: list[Expression], rng: np.random.Generator, samples: int) -> bool:
    """Check whether a sum of terms vanishes.

    It vanishes relative to the size of its terms at random states.
    """
    found = False
    for _ in range(samples):
        draws = rng.uniform(-1, 1, len(dynamics.symbols))
        point = {item: SR(float(value)) for item, value in zip(dynamics.symbols, draws, strict=True)}
        point |= {**dynamics.parameters, SR(dynamics.problem.time): SR(float(rng.uniform()))}
        values = np.array([complex(SR(term).subs(point)) for term in terms])
        if not np.isfinite(values).all():
            continue
        if abs(values.sum()) > TOLERANCE * max(1.0, float(np.abs(values).sum())):
            return False
        found = True
    return found


def _decide(
    dynamics: _DynamicsSpec,
    terms: list[Expression],
    rng: np.random.Generator,
    samples: int,
) -> tuple[Expression, bool, str]:
    """Return the simplified sum of terms, whether it is zero, and the deciding method."""
    total = sum(terms, SR(0)).simplify_full()
    if total.is_zero():
        return total, True, 'symbolic'
    if total.is_numeric():
        return total, False, 'symbolic'
    return total, _vanishes(dynamics, terms, rng, samples), 'numeric'
