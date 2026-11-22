"""Derive the Poisson brackets and the Noether charges of a mechanical system with Sage."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from .....core.analysis import InvariantsResult, invariants
from .....core.analysis.types import DataSource
from .....core.components import Observable
from .....core.components.interactions import BaseInteraction
from .....core.errors import AnalysisError
from .....core.systems import System
from ....general.mechanics import EulerLagrangeInteraction
from ._hamiltonian import HamiltonianInteraction
from ._symmetry_spec import SymmetrySpec

TOLERANCE = 1e-9


@dataclass(frozen=True)
class NoetherResult:
    """A result of `noether`, whether a symmetry leaves a system unchanged and its conserved charge.

    Args:
        invariant:
            Whether the Hamiltonian, or else the Lagrangian, is unchanged to first order by the symmetry.

        remainder:
            The first-order change of the Hamiltonian or the Lagrangian under the symmetry, zero when it is invariant.

        charge:
            The Noether charge of the symmetry, conserved when the symmetry leaves the system unchanged.
    """

    invariant: bool
    remainder: Expression
    charge: Expression


def poisson_brackets(source: DataSource, expressions: Sequence[Observable]) -> dict[tuple[str, str], Expression]:
    """Return the Poisson bracket of every pair of expressions of a mechanical system.

    The bracket is `{f, g} = sum df/dq dg/dp - df/dp dg/dq`, summed over the canonical pairs of the system's
    `HamiltonianInteraction`s. The expressions are `Observable`s whose expression is a function of the namespace of the
    system, where `s.<element>.<state>` is the symbol of a state.

    Args:
        source:
            The system, or a trajectory of it.

        expressions:
            The expressions to bracket.

    Returns:
        brackets:
            A mapping from every pair of labels to the bracket of their expressions.

    Raises:
        AnalysisError:
            The system has no Hamiltonian interaction, or an expression is not one of the system.

    Examples:
        >>> from skcomplex.core.components import Observable
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.mechanics import HamiltonianInteraction, poisson_brackets
        >>> energy = lambda s: (s.target.p**2 + s.target.q**2) / 2
        >>> system = System(
        ...     elements=[Element('mass', values={'q': 1.0, 'p': 0.0})],
        ...     interactions=[HamiltonianInteraction('spring', target='mass', hamiltonian=energy)],
        ... )
        >>> expressions = [Observable('position', lambda s: s.mass.q), Observable('momentum', lambda s: s.mass.p)]
        >>> poisson_brackets(system, expressions)
        {('position', 'momentum'): 1}
    """
    base = invariants(source, expressions=expressions)
    pairs = _get_pairs(base)
    values = {expression.label: _evaluate(base.namespace, expression) for expression in expressions}
    return {
        (first, second): sum(_get_bracket_terms(values[first], values[second], pairs), SR(0)).simplify_full()
        for first, second in itertools.combinations(values, 2)
    }


def noether(
    source: DataSource,
    symmetry: SymmetrySpec,
    *,
    samples: int = 16,
    random_state: int | np.random.Generator | None = None,
) -> NoetherResult:
    """Return whether a symmetry leaves a mechanical system unchanged, and its Noether charge.

    The Hamiltonian is the sum of those of the system's `HamiltonianInteraction`s, and the Lagrangian the sum of those
    of its `EulerLagrangeInteraction`s. The symmetry leaves the system unchanged when the first-order change of the
    Hamiltonian, or else of the Lagrangian, under its variation simplifies to zero, or vanishes within a tolerance at
    random states when simplification does not decide. The charge is `sum p dq` for a Hamiltonian and
    `sum dL/dq' dq` for a Lagrangian, with the variation of the velocities `dq' = sum_j d(dq)/dq_j q_j'`.

    Args:
        source:
            The system, or a trajectory of it.

        symmetry:
            The symmetry to check.

        samples:
            The number of random states of the numeric check.

        random_state:
            The seed or generator of the random states.

    Returns:
        result:
            Whether the symmetry leaves the system unchanged, the first-order change, and the Noether charge.

    Raises:
        AnalysisError:
            The symmetry is not a `SymmetrySpec`, or the system has neither a Hamiltonian nor an Euler-Lagrange
            interaction.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.mechanics import HamiltonianInteraction, SymmetrySpec, noether
        >>> free = lambda s: s.target.p**2 / 2
        >>> system = System(
        ...     elements=[Element('ball', values={'q': 0.0, 'p': 1.0})],
        ...     interactions=[HamiltonianInteraction('motion', target='ball', hamiltonian=free)],
        ... )
        >>> result = noether(system, SymmetrySpec('translation', lambda s: {'ball.q': 1}))
        >>> result.invariant, result.charge
        (True, p_ball)
    """
    if not isinstance(symmetry, SymmetrySpec):
        error_msg = f'The symmetry is {symmetry!r}. Give a SymmetrySpec to check.'
        raise AnalysisError(error_msg)
    symmetry._check()
    base = invariants(source)
    variations = _get_variations(base.namespace, symmetry)
    if _get_hamiltonians(base.system):
        hamiltonian = _get_hamiltonian(base)
        terms = [hamiltonian.diff(item) * dq for item, dq in variations.items()]
        charge = sum((p * variations[q] for q, p in _get_pairs(base) if q in variations), SR(0))
    else:
        terms, charge = _get_lagrangian_change(base, variations)
    remainder = sum(terms, SR(0)).simplify_full()
    invariant = bool(remainder.is_zero()) or (
        not remainder.is_numeric() and _vanishes(base, terms, np.random.default_rng(random_state), samples)
    )
    return NoetherResult(invariant, SR(0) if invariant else remainder, SR(charge).simplify_full())


def _resolve(namespace: SimpleNamespace, path: str) -> object:
    """Return the namespace of the element or state at a path of the system."""
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


def _components(value: object) -> list[Expression]:
    """Return the components of a scalar or vector value as expressions."""
    return [SR(item) for item in value] if isinstance(value, FreeModuleElement | list | tuple) else [SR(value)]


def _get_interaction_namespace(base: InvariantsResult, interaction: BaseInteraction) -> SimpleNamespace:
    """Return the namespace an interaction reads, with the namespaces of its roles and its parameters."""
    roles: dict[str, object] = {}
    for name in interaction.get_requirements().roles:
        value = getattr(interaction, name)
        if isinstance(value, str):
            roles[name] = _resolve(base.namespace, value)
        elif value:
            roles[name] = tuple(_resolve(base.namespace, item) for item in value)
    own = getattr(base.namespace, interaction.label, SimpleNamespace())
    return SimpleNamespace(**vars(own), **roles, t=base.dynamics.problem.time)


def _get_hamiltonians(system: System) -> list[HamiltonianInteraction]:
    """Return the Hamiltonian interactions of the system."""
    return [item for item in system.interactions if isinstance(item, HamiltonianInteraction)]


def _get_pairs(base: InvariantsResult) -> list[tuple[Expression, Expression]]:
    """Return the canonical pairs of the Hamiltonian interactions of the system, every pair once."""
    pairs: list[tuple[Expression, Expression]] = []
    for interaction in _get_hamiltonians(base.system):
        for q, p in interaction._get_pairs(_get_interaction_namespace(base, interaction)):
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


def _get_hamiltonian(base: InvariantsResult) -> Expression:
    """Return the Hamiltonian of the system, the sum of those of its Hamiltonian interactions."""
    return sum(
        (
            SR(interaction._get_hamiltonian(_get_interaction_namespace(base, interaction)))
            for interaction in _get_hamiltonians(base.system)
        ),
        SR(0),
    )


def _get_bracket_terms(
    first: Expression,
    second: Expression,
    pairs: list[tuple[Expression, Expression]],
) -> list[Expression]:
    """Return the terms of the Poisson bracket of two expressions over canonical pairs."""
    return [term for q, p in pairs for term in (first.diff(q) * second.diff(p), -first.diff(p) * second.diff(q))]


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


def _get_lagrangian(base: InvariantsResult) -> tuple[Expression, dict[Expression, Expression]]:
    """Return the Lagrangian of the system in the symbols of its states, and the velocity of every coordinate.

    The velocity of a coordinate is its rate in the first-order equations of the system.
    """
    lagrangian, velocities = SR(0), {}
    first_order = dict(zip(base.dynamics.symbols, base.dynamics.rates, strict=True))
    for interaction in base.system.interactions:
        if not isinstance(interaction, EulerLagrangeInteraction):
            continue
        own = _get_interaction_namespace(base, interaction)
        for name in interaction.get_requirements().roles:
            member = getattr(own, name)
            if not isinstance(member, SimpleNamespace):
                continue
            for state in interaction._get_names():
                positions = _components(getattr(member, state))
                velocities.update({position: first_order[position] for position in positions})
        lagrangian += interaction._get_lagrangian_of(own, velocities)
    if not velocities:
        error_msg = (
            'The system has neither a HamiltonianInteraction nor an EulerLagrangeInteraction, so it has no Hamiltonian '
            'or Lagrangian to check a symmetry against.'
        )
        raise AnalysisError(error_msg)
    return lagrangian, velocities


def _get_lagrangian_change(
    base: InvariantsResult,
    variations: dict[Expression, Expression],
) -> tuple[list[Expression], Expression]:
    """Return the terms of the first-order change of the Lagrangian under variations, and the Noether charge."""
    lagrangian, velocities = _get_lagrangian(base)
    terms, charge = [], SR(0)
    for item, dq in variations.items():
        terms.append(lagrangian.diff(item) * dq)
        if item not in velocities:
            continue
        momentum = lagrangian.diff(velocities[item])
        terms += [momentum * dq.diff(other) * velocity for other, velocity in velocities.items()]
        charge += momentum * dq
    return terms, charge


def _vanishes(base: InvariantsResult, terms: list[Expression], rng: np.random.Generator, samples: int) -> bool:
    """Return whether a sum of terms vanishes, relative to the size of its terms, at random states."""
    dynamics = base.dynamics
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
