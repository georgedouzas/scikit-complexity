"""Derive the Poisson brackets, conserved quantities, symmetries, and Noether charges of a system with Sage."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ..components.interactions import BaseInteraction, EulerLagrangeInteraction, HamiltonianInteraction
from ..components.laws.utils import derivative
from ..components.system import System
from ..engine._utils import declarations_of
from ..engine.compile import ObservableEvaluator
from ..engine.problems import ProblemSpec
from ..engine.problems._dynamics_spec import _DynamicsSpec
from ..errors import AnalysisError
from ..specs import ObservableSpec, SymmetrySpec
from ..types import VariableKind
from ._base import _check_choice, _check_settings, _read_system
from ._stability import _dynamics_of
from .types import Source

KINDS = ('brackets', 'conservation', 'symmetry')
SETTINGS = {
    'brackets': {'expressions'},
    'conservation': {'expressions', 'samples', 'random_state'},
    'symmetry': {'symmetry'},
}
"""Settings every kind reads, besides `kind`."""

TOLERANCE = 1e-9


@dataclass(frozen=True)
class InvariantsResult:
    """Poisson brackets, conserved quantities, or a symmetry with its Noether charge, of a system.

    Args:
        problem:
            The compiled problem.

        system:
            The system.

        dynamics:
            The first-order equations of the system, with the values of its parameters.

        namespace:
            The namespace of the symbols of the states and parameters of the system.

        brackets:
            A mapping from every pair of labels to the Poisson bracket of their expressions, set for `'brackets'`.

        rates:
            A mapping from every label to the rate of change of its expression, set for `'conservation'`.

        conserved:
            A mapping from every label to whether its expression is conserved, set for `'conservation'`.

        methods:
            A mapping from every label to the method that decided, `'symbolic'` or `'numeric'`, set for
            `'conservation'`.

        invariant:
            Whether the Hamiltonian or the Lagrangian is unchanged to first order by the symmetry, set for
            `'symmetry'`.

        remainder:
            The first-order change of the Hamiltonian or the Lagrangian under the symmetry, zero when it is invariant,
            set for `'symmetry'`.

        charge:
            The Noether charge of the symmetry, set for `'symmetry'`.
    """

    problem: ProblemSpec
    system: System
    dynamics: _DynamicsSpec
    namespace: SimpleNamespace
    brackets: dict[tuple[str, str], Expression] | None = None
    rates: dict[str, Expression] | None = None
    conserved: dict[str, bool] | None = None
    methods: dict[str, str] | None = None
    invariant: bool | None = None
    remainder: Expression | None = None
    charge: Expression | None = None


def invariants(
    source: Source,
    *,
    kind: str = 'conservation',
    expressions: Sequence[ObservableSpec] = (),
    symmetry: SymmetrySpec | None = None,
    samples: int = 16,
    random_state: int | np.random.Generator | None = None,
) -> InvariantsResult:
    """Return the Poisson brackets, conserved quantities, or a symmetry with its Noether charge, of a system.

    The canonical pairs and the Hamiltonian `H` are those of the system's `HamiltonianInteraction`s, whose
    Hamiltonians sum, and the Lagrangian `L` is the sum of those of its `EulerLagrangeInteraction`s. The expressions are
    `ObservableSpec`s whose expression is a function of the namespace of the system, where `s.<element>.<state>` is the
    symbol of a state, or a symbolic expression of those symbols. The kinds:

    - `'brackets'`: the Poisson bracket `{f, g} = sum df/dq dg/dp - df/dp dg/dq` of every pair of expressions, summed
      over the canonical pairs.
    - `'conservation'`: the rate of change `df/dt` of every expression, which is `diff(f, t) + {f, H}` for a system
      whose dynamics all come from Hamiltonian interactions, and the Lie derivative along the vector field of its
      equations otherwise. An expression is conserved when its rate simplifies to zero, or, when simplification does
      not decide, when the rate vanishes within a tolerance at random states with the parameters at their values.
    - `'symmetry'`: whether the Hamiltonian, or else the Lagrangian, is unchanged to first order by the variation of a
      `SymmetrySpec`, and the Noether charge of the symmetry, `sum p dq` for a Hamiltonian and `sum dL/dq' dq` for a
      Lagrangian, with the variation of the velocities `dq' = sum_j d(dq)/dq_j q_j'`.

    Args:
        source:
            The system, or a trajectory of it.

        kind:
            The invariants to derive, one of the kinds above.

        expressions:
            The expressions to bracket or to check.

        symmetry:
            The symmetry to check.

        samples:
            The number of random states of the numeric check.

        random_state:
            The seed or generator of the random states.

    Returns:
        result:
            The brackets, the rates and conservation of the expressions, or the symmetry and its charge.

    Raises:
        AnalysisError:
            The source is not a system, the system is not an ODE, an expression or the symmetry is missing or invalid,
            or the system has no Hamiltonian interaction for brackets, or neither a Hamiltonian nor an Euler-Lagrange
            interaction for a symmetry.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.analysis import invariants
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import HamiltonianInteraction
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.specs import ObservableSpec, StateSpec
        >>> class OscillatorElement(Element):
        ...     def __init__(self, label, q: Annotated[float, StateSpec()] = 1.0, p: Annotated[float, StateSpec()] = 0):
        ...         super().__init__(label)
        ...         self.q = q
        ...         self.p = p
        >>> energy = lambda s: (s.target.p**2 + s.target.q**2) / 2
        >>> system = System(
        ...     elements=[OscillatorElement('mass')],
        ...     interactions=[HamiltonianInteraction('spring', target='mass', hamiltonian=energy)],
        ... )
        >>> expressions = [
        ...     ObservableSpec('energy', lambda s: s.mass.q**2 + s.mass.p**2),
        ...     ObservableSpec('position', lambda s: s.mass.q),
        ... ]
        >>> result = invariants(system, expressions=expressions)
        >>> result.conserved
        {'energy': True, 'position': False}
        >>> result.rates['position']
        p_mass
    """
    _check_choice('invariants', 'kind', kind, KINDS)
    _check_settings(
        invariants,
        'kind',
        SETTINGS[kind],
        kind=kind,
        expressions=expressions,
        symmetry=symmetry,
        samples=samples,
        random_state=random_state,
    )
    _check_inputs(kind, expressions, symmetry, samples)
    system = _read_system(source)
    if not isinstance(system, System):
        error_msg = (
            f'Invariants are derived from the interactions of a system, not from {type(system).__name__}. Give the '
            'system.'
        )
        raise AnalysisError(error_msg)
    dynamics = _dynamics_of(system)
    namespace = ObservableEvaluator._namespace_of(dynamics.problem)
    result = InvariantsResult(dynamics.problem, system, dynamics, namespace)
    if kind == 'brackets':
        pairs = _get_pairs(system, namespace, dynamics)
        values = {expression.label: _evaluate(namespace, expression) for expression in expressions}
        brackets = {
            (first, second): sum(_get_bracket_terms(values[first], values[second], pairs), SR(0)).simplify_full()
            for first, second in itertools.combinations(values, 2)
        }
        return replace(result, brackets=brackets)
    rng = np.random.default_rng(random_state)
    if kind == 'conservation':
        hamiltonian = _is_hamiltonian(system, namespace, dynamics)
        rates, conserved, methods = {}, {}, {}
        for expression in expressions:
            terms = _get_rate_terms(
                system,
                namespace,
                dynamics,
                _evaluate(namespace, expression),
                hamiltonian=hamiltonian,
            )
            rate, is_conserved, method = _decide(dynamics, terms, rng, samples)
            rates[expression.label] = rate
            conserved[expression.label], methods[expression.label] = is_conserved, method
        return replace(result, rates=rates, conserved=conserved, methods=methods)
    assert symmetry is not None
    variations = _get_variations(namespace, symmetry)
    if _get_hamiltonians(system):
        hamiltonian_value = _get_hamiltonian(system, namespace, dynamics)
        terms = [hamiltonian_value.diff(item) * dq for item, dq in variations.items()]
        charge = sum(
            (p * variations[q] for q, p in _get_pairs(system, namespace, dynamics) if q in variations),
            SR(0),
        )
    else:
        terms, charge = _get_lagrangian_change(system, namespace, dynamics, variations)
    remainder, invariant, _ = _decide(dynamics, terms, rng, samples)
    return replace(
        result,
        invariant=invariant,
        remainder=SR(0) if invariant else remainder,
        charge=SR(charge).simplify_full(),
    )


def _check_inputs(
    kind: str,
    expressions: Sequence[ObservableSpec],
    symmetry: SymmetrySpec | None,
    samples: int,
) -> None:
    """Raise an error for an expression that is not one of the system, a missing symmetry, or bad samples."""
    for expression in expressions:
        if not isinstance(expression, ObservableSpec) or expression.group is not None:
            error_msg = f'{expression!r} is not an expression of the system. Give an ObservableSpec without group.'
            raise AnalysisError(error_msg)
    if kind == 'symmetry' and not isinstance(symmetry, SymmetrySpec):
        error_msg = f'The symmetry is {symmetry!r}. Give a SymmetrySpec to check.'
        raise AnalysisError(error_msg)
    if isinstance(symmetry, SymmetrySpec) and kind == 'symmetry':
        symmetry._validated()
    if not isinstance(samples, int) or samples < 1:
        error_msg = f'samples = {samples!r} must be a positive integer.'
        raise AnalysisError(error_msg)


def _resolve(namespace: SimpleNamespace, path: str) -> object:
    """Return the namespace of the element or state at a path of the system, or raise an error."""
    value: object = namespace
    for part in path.replace('/', '.').split('.'):
        if not hasattr(value, part):
            error_msg = f'The system has no element or state at {path!r}. Give the path of a state, such as `a.x`.'
            raise AnalysisError(error_msg)
        value = getattr(value, part)
    return value


def _get_interaction_namespace(
    namespace: SimpleNamespace,
    dynamics: _DynamicsSpec,
    interaction: BaseInteraction,
) -> SimpleNamespace:
    """Return the namespace an interaction reads, with the namespaces of its roles and its parameters."""
    roles: dict[str, object] = {}
    for name in declarations_of(type(interaction), VariableKind.ROLE):
        value = getattr(interaction, name)
        if isinstance(value, str):
            roles[name] = _resolve(namespace, value)
        elif value:
            roles[name] = tuple(_resolve(namespace, item) for item in value)
    own = getattr(namespace, interaction.label, SimpleNamespace())
    return SimpleNamespace(**vars(own), **roles, t=dynamics.problem.time)


def _get_hamiltonians(system: System) -> list[HamiltonianInteraction]:
    """Return the Hamiltonian interactions of the system."""
    return [item for item in system.interactions if isinstance(item, HamiltonianInteraction)]


def _get_pairs(
    system: System,
    namespace: SimpleNamespace,
    dynamics: _DynamicsSpec,
) -> list[tuple[Expression, Expression]]:
    """Return the canonical pairs of the components of the system's Hamiltonian interactions, each once."""
    pairs: list[tuple[Expression, Expression]] = []
    for interaction in _get_hamiltonians(system):
        for q, p in interaction._get_pairs(_get_interaction_namespace(namespace, dynamics, interaction)):
            for pair in zip(_components(q), _components(p), strict=True):
                if pair not in pairs:
                    pairs.append(pair)
    if not pairs:
        error_msg = (
            'The system has no canonical pairs, which come from a HamiltonianInteraction. Write its dynamics with a '
            'HamiltonianInteraction.'
        )
        raise AnalysisError(error_msg)
    return pairs


def _get_hamiltonian(system: System, namespace: SimpleNamespace, dynamics: _DynamicsSpec) -> Expression:
    """Return the Hamiltonian of the system, the sum of those of its Hamiltonian interactions."""
    return sum(
        (
            SR(interaction._get_hamiltonian(_get_interaction_namespace(namespace, dynamics, interaction)))
            for interaction in _get_hamiltonians(system)
        ),
        SR(0),
    )


def _get_lagrangian(
    system: System,
    namespace: SimpleNamespace,
    dynamics: _DynamicsSpec,
) -> tuple[Expression, dict[Expression, Expression]]:
    """Return the Lagrangian of the system in the symbols of its states, and the velocity of every coordinate."""
    lagrangian, velocities = SR(0), {}
    for interaction in system.interactions:
        if not isinstance(interaction, EulerLagrangeInteraction):
            continue
        own = _get_interaction_namespace(namespace, dynamics, interaction)
        for name in declarations_of(type(interaction), VariableKind.ROLE):
            member = getattr(own, name)
            if not isinstance(member, SimpleNamespace):
                continue
            for state in interaction._get_names():
                positions = _components(getattr(member, state))
                rates = _components(getattr(member, f'{state}_d1'))
                velocities.update(dict(zip(positions, rates, strict=True)))
        lagrangian += SR(interaction._get_lagrangian(own))
    if not velocities:
        error_msg = (
            'The system has neither a HamiltonianInteraction nor an EulerLagrangeInteraction, so it has no Hamiltonian '
            'or Lagrangian to check a symmetry against.'
        )
        raise AnalysisError(error_msg)
    lagrangian = lagrangian.subs({derivative(q, 1): v for q, v in velocities.items()})
    return lagrangian, velocities


def _components(value: object) -> list[Expression]:
    """Return the components of a scalar or vector value as expressions."""
    return [SR(item) for item in value] if isinstance(value, FreeModuleElement | list | tuple) else [SR(value)]


def _evaluate(namespace: SimpleNamespace, expression: ObservableSpec) -> Expression:
    """Return the expression of an observable in the symbols of the system."""
    value = expression.expression
    scope = namespace if expression.element is None else _resolve(namespace, expression.element)
    return SR(value(scope) if callable(value) else value)


def _get_bracket_terms(
    first: Expression,
    second: Expression,
    pairs: list[tuple[Expression, Expression]],
) -> list[Expression]:
    """Return the terms of the Poisson bracket of two expressions over canonical pairs."""
    return [term for q, p in pairs for term in (first.diff(q) * second.diff(p), -first.diff(p) * second.diff(q))]


def _is_hamiltonian(system: System, namespace: SimpleNamespace, dynamics: _DynamicsSpec) -> bool:
    """Check whether the vector field of the system is that of Hamilton's equations of its Hamiltonian."""
    if not _get_hamiltonians(system):
        return False
    pairs = _get_pairs(system, namespace, dynamics)
    hamiltonian = _get_hamiltonian(system, namespace, dynamics)
    rates = {q: hamiltonian.diff(p) for q, p in pairs} | {p: -hamiltonian.diff(q) for q, p in pairs}
    return set(dynamics.symbols) == set(rates) and all(
        (rate - rates[item]).simplify_full().is_zero()
        for item, rate in zip(dynamics.symbols, dynamics.rates, strict=True)
    )


def _get_rate_terms(
    system: System,
    namespace: SimpleNamespace,
    dynamics: _DynamicsSpec,
    expression: Expression,
    *,
    hamiltonian: bool,
) -> list[Expression]:
    """Return the terms of the rate of an expression, of its bracket with the Hamiltonian or its Lie derivative."""
    terms = [expression.diff(SR(dynamics.problem.time))]
    if hamiltonian:
        return terms + _get_bracket_terms(
            expression,
            _get_hamiltonian(system, namespace, dynamics),
            _get_pairs(system, namespace, dynamics),
        )
    return terms + [expression.diff(item) * rate for item, rate in zip(dynamics.symbols, dynamics.rates, strict=True)]


def _vanishes(dynamics: _DynamicsSpec, terms: list[Expression], rng: np.random.Generator, samples: int) -> bool:
    """Check whether a sum of terms vanishes relative to the size of its terms at random states."""
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
    """Return the simplified sum of terms, whether it is zero, and the method that decided."""
    total = sum(terms, SR(0)).simplify_full()
    if total.is_zero():
        return total, True, 'symbolic'
    if total.is_numeric():
        return total, False, 'symbolic'
    return total, _vanishes(dynamics, terms, rng, samples), 'numeric'


def _get_variations(namespace: SimpleNamespace, symmetry: SymmetrySpec) -> dict[Expression, Expression]:
    """Return the variation of every component the symmetry moves."""
    variations: dict[Expression, Expression] = {}
    for path, variation in symmetry.generator(namespace).items():
        states, changes = _components(_resolve(namespace, path)), _components(variation)
        if len(states) != len(changes):
            error_msg = (
                f'The symmetry {symmetry.label!r} varies {path} by {len(changes)} components for {len(states)}. Give '
                'one variation per component.'
            )
            raise AnalysisError(error_msg)
        variations.update(dict(zip(states, changes, strict=True)))
    return variations


def _get_lagrangian_change(
    system: System,
    namespace: SimpleNamespace,
    dynamics: _DynamicsSpec,
    variations: dict[Expression, Expression],
) -> tuple[list[Expression], Expression]:
    """Return the terms of the first-order change of the Lagrangian under variations, and the Noether charge."""
    lagrangian, velocities = _get_lagrangian(system, namespace, dynamics)
    terms, charge = [], SR(0)
    for item, dq in variations.items():
        terms.append(lagrangian.diff(item) * dq)
        if item not in velocities:
            continue
        momentum = lagrangian.diff(velocities[item])
        terms += [momentum * dq.diff(other) * velocity for other, velocity in velocities.items()]
        charge += momentum * dq
    return terms, charge
