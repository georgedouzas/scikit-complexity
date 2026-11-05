"""Derive the Poisson brackets, conserved quantities, symmetries, and Noether charges of a system with Sage."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ..core.interactions import BaseInteraction, EulerLagrangeInteraction, HamiltonianInteraction
from ..core.laws.utils import derivative
from ..core.system import System
from ..engine._utils import declarations_of
from ..engine.compile import ObservableEvaluator
from ..engine.problems._dynamics_spec import _DynamicsSpec
from ..errors import AnalysisError
from ..specs import ObservableSpec, SymmetrySpec
from ..types import DataKind, VariableKind
from ._base import BaseAnalysis
from ._stability import StabilityAnalysis
from .types import Source

KINDS = ('brackets', 'conservation', 'symmetry')
TOLERANCE = 1e-9
SETTINGS = {
    'brackets': {'expressions'},
    'conservation': {'expressions', 'samples', 'random_state'},
    'symmetry': {'symmetry'},
}
"""Settings every kind reads, besides `kind`."""


class InvariantAnalysis(BaseAnalysis):
    """Poisson brackets, conserved quantities, and symmetries of a system, with the Noether charge of a symmetry.

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
        kind:
            The analysis to run, one of the kinds above.

        expressions:
            The expressions to bracket or to check.

        symmetry:
            The symmetry to check.

        samples:
            The number of random states of the numeric check.

        random_state:
            The seed or generator of the random states.

    Attributes:
        problem_:
            The compiled problem.

        system_:
            The system.

        dynamics_:
            The first-order equations of the system, with the values of its parameters.

        namespace_:
            The namespace of the symbols of the states and parameters of the system.

        brackets_:
            A mapping from every pair of labels to the Poisson bracket of their expressions.

        rates_:
            A mapping from every label to the rate of change of its expression.

        conserved_:
            A mapping from every label to whether its expression is conserved.

        methods_:
            A mapping from every label to the method that decided, `'symbolic'` or `'numeric'`.

        invariant_:
            Whether the Hamiltonian or the Lagrangian is unchanged to first order by the symmetry.

        remainder_:
            The first-order change of the Hamiltonian or the Lagrangian under the symmetry, zero when it is invariant.

        charge_:
            The Noether charge of the symmetry.

    Raises:
        AnalysisError:
            The source is not a system, the system is not an ODE, an expression or the symmetry is missing or invalid,
            or the system has no Hamiltonian interaction for brackets, or neither a Hamiltonian nor an Euler-Lagrange
            interaction for a symmetry.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.analysis import InvariantAnalysis
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.interactions import HamiltonianInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.specs import ObservableSpec, StateSpec
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
        >>> analysis = InvariantAnalysis(expressions=expressions).prepare(system).run()
        >>> analysis.conserved_
        {'energy': True, 'position': False}
        >>> analysis.rates_['position']
        p_mass
    """

    def __init__(
        self,
        kind: str = 'conservation',
        *,
        expressions: Sequence[ObservableSpec] = (),
        symmetry: SymmetrySpec | None = None,
        samples: int = 16,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.kind = kind
        self.expressions = expressions
        self.symmetry = symmetry
        self.samples = samples
        self.random_state = random_state

    def _check(self) -> None:
        """Validate the kind, the settings it reads, the expressions, the symmetry, and the samples."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind])
        for expression in self.expressions:
            if not isinstance(expression, ObservableSpec) or expression.group is not None:
                error_msg = f'{expression!r} is not an expression of the system. Give an ObservableSpec without group.'
                raise AnalysisError(error_msg)
        if self.kind == 'symmetry' and not isinstance(self.symmetry, SymmetrySpec):
            error_msg = f'The symmetry is {self.symmetry!r}. Give a SymmetrySpec to check.'
            raise AnalysisError(error_msg)
        if self.kind == 'symmetry' and isinstance(self.symmetry, SymmetrySpec):
            self.symmetry._validated()
        if not isinstance(self.samples, int) or self.samples < 1:
            error_msg = f'samples = {self.samples!r} must be a positive integer.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the system kind."""
        return DataKind.SYSTEM

    def _prepare(self, source: Source) -> None:
        """Compile the system of the source and derive the namespace of its symbols."""
        system = self._read_system(source)
        if not isinstance(system, System):
            error_msg = (
                f'Invariants are derived from the interactions of a system, not from {type(system).__name__}. Give '
                'the system.'
            )
            raise AnalysisError(error_msg)
        self.system_ = system
        self.dynamics_ = StabilityAnalysis._dynamics_of(system)
        self.problem_ = self.dynamics_.problem
        self.namespace_ = ObservableEvaluator._namespace_of(self.problem_)

    def _resolve(self, path: str) -> object:
        """Return the namespace of the element or state at a path of the system, or raise an error."""
        value: object = self.namespace_
        for part in path.replace('/', '.').split('.'):
            if not hasattr(value, part):
                error_msg = f'The system has no element or state at {path!r}. Give the path of a state, such as `a.x`.'
                raise AnalysisError(error_msg)
            value = getattr(value, part)
        return value

    def _get_interaction_namespace(self, interaction: BaseInteraction) -> SimpleNamespace:
        """Return the namespace an interaction reads, with the namespaces of its roles and its parameters."""
        roles: dict[str, object] = {}
        for name in declarations_of(type(interaction), VariableKind.ROLE):
            value = getattr(interaction, name)
            if isinstance(value, str):
                roles[name] = self._resolve(value)
            elif value:
                roles[name] = tuple(self._resolve(item) for item in value)
        own = getattr(self.namespace_, interaction.label, SimpleNamespace())
        return SimpleNamespace(**vars(own), **roles, t=self.problem_.time)

    def _get_hamiltonians(self) -> list[HamiltonianInteraction]:
        """Return the Hamiltonian interactions of the system."""
        return [item for item in self.system_.interactions if isinstance(item, HamiltonianInteraction)]

    def _get_pairs(self) -> list[tuple[Expression, Expression]]:
        """Return the canonical pairs of the components of the system's Hamiltonian interactions, each once."""
        pairs: list[tuple[Expression, Expression]] = []
        for interaction in self._get_hamiltonians():
            for q, p in interaction._get_pairs(self._get_interaction_namespace(interaction)):
                for pair in zip(self._components(q), self._components(p), strict=True):
                    if pair not in pairs:
                        pairs.append(pair)
        if not pairs:
            error_msg = (
                'The system has no canonical pairs, which come from a HamiltonianInteraction. Write its dynamics with '
                'a HamiltonianInteraction.'
            )
            raise AnalysisError(error_msg)
        return pairs

    def _get_hamiltonian(self) -> Expression:
        """Return the Hamiltonian of the system, the sum of those of its Hamiltonian interactions."""
        return sum(
            (
                SR(interaction._get_hamiltonian(self._get_interaction_namespace(interaction)))
                for interaction in self._get_hamiltonians()
            ),
            SR(0),
        )

    def _get_lagrangian(self) -> tuple[Expression, dict[Expression, Expression]]:
        """Return the Lagrangian of the system in the symbols of its states, and the velocity of every coordinate."""
        lagrangian, velocities = SR(0), {}
        for interaction in self.system_.interactions:
            if not isinstance(interaction, EulerLagrangeInteraction):
                continue
            namespace = self._get_interaction_namespace(interaction)
            for name in declarations_of(type(interaction), VariableKind.ROLE):
                member = getattr(namespace, name)
                if not isinstance(member, SimpleNamespace):
                    continue
                for state in interaction._get_names():
                    positions = self._components(getattr(member, state))
                    rates = self._components(getattr(member, f'{state}_d1'))
                    velocities.update(dict(zip(positions, rates, strict=True)))
            lagrangian += SR(interaction._get_lagrangian(namespace))
        if not velocities:
            error_msg = (
                'The system has neither a HamiltonianInteraction nor an EulerLagrangeInteraction, so it has no '
                'Hamiltonian or Lagrangian to check a symmetry against.'
            )
            raise AnalysisError(error_msg)
        lagrangian = lagrangian.subs({derivative(q, 1): v for q, v in velocities.items()})
        return lagrangian, velocities

    @staticmethod
    def _components(value: object) -> list[Expression]:
        """Return the components of a scalar or vector value as expressions."""
        return [SR(item) for item in value] if isinstance(value, FreeModuleElement | list | tuple) else [SR(value)]

    def _evaluate(self, expression: ObservableSpec) -> Expression:
        """Return the expression of an observable in the symbols of the system."""
        value = expression.expression
        namespace = self.namespace_ if expression.element is None else self._resolve(expression.element)
        return SR(value(namespace) if callable(value) else value)

    def _get_bracket_terms(
        self,
        first: Expression,
        second: Expression,
        pairs: list[tuple[Expression, Expression]],
    ) -> list[Expression]:
        """Return the terms of the Poisson bracket of two expressions over canonical pairs."""
        return [term for q, p in pairs for term in (first.diff(q) * second.diff(p), -first.diff(p) * second.diff(q))]

    def _is_hamiltonian(self, dynamics: _DynamicsSpec) -> bool:
        """Check whether the vector field of the system is that of Hamilton's equations of its Hamiltonian."""
        if not self._get_hamiltonians():
            return False
        pairs = self._get_pairs()
        hamiltonian = self._get_hamiltonian()
        rates = {q: hamiltonian.diff(p) for q, p in pairs} | {p: -hamiltonian.diff(q) for q, p in pairs}
        return set(dynamics.symbols) == set(rates) and all(
            (rate - rates[item]).simplify_full().is_zero()
            for item, rate in zip(dynamics.symbols, dynamics.rates, strict=True)
        )

    def _get_rate_terms(
        self,
        expression: Expression,
        dynamics: _DynamicsSpec,
        *,
        hamiltonian: bool,
    ) -> list[Expression]:
        """Return the terms of the rate of an expression, of its bracket with the Hamiltonian or its Lie derivative."""
        terms = [expression.diff(SR(self.problem_.time))]
        if hamiltonian:
            return terms + self._get_bracket_terms(expression, self._get_hamiltonian(), self._get_pairs())
        return terms + [
            expression.diff(item) * rate for item, rate in zip(dynamics.symbols, dynamics.rates, strict=True)
        ]

    def _vanishes(self, terms: list[Expression], rng: np.random.Generator) -> bool:
        """Check whether a sum of terms vanishes relative to the size of its terms at random states."""
        dynamics = self.dynamics_
        found = False
        for _ in range(self.samples):
            draws = rng.uniform(-1, 1, len(dynamics.symbols))
            point = {item: SR(float(value)) for item, value in zip(dynamics.symbols, draws, strict=True)}
            point |= {**dynamics.parameters, SR(self.problem_.time): SR(float(rng.uniform()))}
            values = np.array([complex(SR(term).subs(point)) for term in terms])
            if not np.isfinite(values).all():
                continue
            if abs(values.sum()) > TOLERANCE * max(1.0, float(np.abs(values).sum())):
                return False
            found = True
        return found

    def _decide(self, terms: list[Expression], rng: np.random.Generator) -> tuple[Expression, bool, str]:
        """Return the simplified sum of terms, whether it is zero, and the method that decided."""
        total = sum(terms, SR(0)).simplify_full()
        if total.is_zero():
            return total, True, 'symbolic'
        if total.is_numeric():
            return total, False, 'symbolic'
        return total, self._vanishes(terms, rng), 'numeric'

    def _run_conservation(self) -> None:
        """Set the rates, the conservation, and the deciding method of every expression."""
        dynamics = self.dynamics_
        hamiltonian = self._is_hamiltonian(dynamics)
        rng = np.random.default_rng(self.random_state)
        self.rates_, self.conserved_, self.methods_ = {}, {}, {}
        for expression in self.expressions:
            terms = self._get_rate_terms(self._evaluate(expression), dynamics, hamiltonian=hamiltonian)
            rate, conserved, method = self._decide(terms, rng)
            self.rates_[expression.label] = rate
            self.conserved_[expression.label], self.methods_[expression.label] = conserved, method

    def _get_variations(self) -> dict[Expression, Expression]:
        """Return the variation of every component the symmetry moves."""
        assert self.symmetry is not None
        variations: dict[Expression, Expression] = {}
        for path, variation in self.symmetry.generator(self.namespace_).items():
            states, changes = self._components(self._resolve(path)), self._components(variation)
            if len(states) != len(changes):
                error_msg = (
                    f'The symmetry {self.symmetry.label!r} varies {path} by {len(changes)} components for '
                    f'{len(states)}. Give one variation per component.'
                )
                raise AnalysisError(error_msg)
            variations.update(dict(zip(states, changes, strict=True)))
        return variations

    def _get_lagrangian_change(
        self,
        variations: dict[Expression, Expression],
    ) -> tuple[list[Expression], Expression]:
        """Return the terms of the first-order change of the Lagrangian under variations, and the Noether charge."""
        lagrangian, velocities = self._get_lagrangian()
        terms, charge = [], SR(0)
        for item, dq in variations.items():
            terms.append(lagrangian.diff(item) * dq)
            if item not in velocities:
                continue
            momentum = lagrangian.diff(velocities[item])
            terms += [momentum * dq.diff(other) * velocity for other, velocity in velocities.items()]
            charge += momentum * dq
        return terms, charge

    def _run_symmetry(self) -> None:
        """Set whether the symmetry keeps the Hamiltonian or the Lagrangian, the remainder, and the charge."""
        variations = self._get_variations()
        if self._get_hamiltonians():
            hamiltonian = self._get_hamiltonian()
            terms = [hamiltonian.diff(item) * dq for item, dq in variations.items()]
            charge = sum((p * variations[q] for q, p in self._get_pairs() if q in variations), SR(0))
        else:
            terms, charge = self._get_lagrangian_change(variations)
        remainder, self.invariant_, _ = self._decide(terms, np.random.default_rng(self.random_state))
        self.remainder_ = SR(0) if self.invariant_ else remainder
        self.charge_ = SR(charge).simplify_full()

    def _run(self, inputs: object = None) -> None:
        """Derive the brackets, the rates and conservation of the expressions, or the symmetry and its charge."""
        if self.kind == 'brackets':
            pairs = self._get_pairs()
            values = {expression.label: self._evaluate(expression) for expression in self.expressions}
            self.brackets_ = {
                (first, second): sum(
                    self._get_bracket_terms(values[first], values[second], pairs),
                    SR(0),
                ).simplify_full()
                for first, second in itertools.combinations(values, 2)
            }
        elif self.kind == 'conservation':
            self._run_conservation()
        else:
            self._run_symmetry()
