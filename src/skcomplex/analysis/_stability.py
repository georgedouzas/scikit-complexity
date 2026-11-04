"""Derive the fixed points and their stability, or the dispersion relation of a field around a uniform state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, matrix, solve
from scipy import optimize

from ..core.elements import FieldElement
from ..core.interactions import BaseInteraction
from ..core.laws import RateLaw
from ..core.spaces import LatticeSpace
from ..core.system import System
from ..engine._utils import bound_labels, components_of, declarations_of, default_of, symbol
from ..engine.problems import ProblemSpec
from ..engine.problems._dynamics_spec import _DynamicsSpec
from ..errors import AnalysisError
from ..types import Array, DataKind, FormKind, VariableKind
from ._base import BaseAnalysis
from ._groups import _Expansion
from ._symbolic import SymbolicAnalysis
from .types import Point, Source

KINDS = ('fixed-points', 'dispersion')
SETTINGS = {
    'fixed-points': {'parameters', 'points'},
    'dispersion': {'at'},
}
"""Settings every kind reads, besides `kind`."""

OPERATOR_NAMES = ('laplacian', 'partial', 'upwind', 'inverse_helmholtz', 'convolve', 'shifted')
EIGENVALUE = symbol('_eigenvalue_')
ZERO = 1e-9
MEMBER_LIMIT = 64


class StabilityAnalysis(BaseAnalysis):
    """Linear stability of a system, of one kind.

    - `'fixed-points'`: the fixed points of a flow or a map, their linear stability, and the Jacobian, nullclines, and
      divergence.
    - `'dispersion'`: the growth rate of every Fourier mode of a system of one field, a site group on a periodic
      lattice whose laws are rate laws. The laws are linearised symbolically around a uniform state, and every
      Laplacian, nested ones included, becomes its eigenvalue on a Fourier mode of the lattice,
      `Σ (2 cos(k h) - 2) / h²` for finite differences and `-|k|²` for a spectral field. Without `at`, the uniform
      state is the fixed point of the laws nearest the mean initial values.

    The fixed points of a flow are where every rate is 0, and those of a map where every state equals its next value.
    Sage solves the equations symbolically, and when it finds no closed form, roots are searched numerically from a
    grid of starting points with the parameters at their values. The stability of every point comes from the
    eigenvalues of the Jacobian there, evaluated at the parameter values.

    Args:
        kind:
            The stability to derive, one of the kinds above.

        parameters:
            Values of parameters by dotted path that replace the system's values. They are substituted before solving,
            and without them the symbolic fixed points keep the parameters as symbols.

        points:
            The points whose stability to analyse, each a mapping from state component name to its value, symbolic
            or numeric, or `None` for every fixed point found.

        at:
            The uniform state of the dispersion relation by state name, or `None` for the nearest fixed point.

    Attributes:
        problem_:
            The compiled problem.

        names_:
            The state component names, in the order of the rows of the Jacobian.

        jacobian_:
            The Sage matrix of the partial derivatives of the right sides by the states.

        divergence_:
            The trace of the Jacobian, 0 for volume-preserving flows.

        nullclines_:
            A mapping from state component name to its nullcline as a Sage relation, where its rate is 0 or where it
            is its own next value.

        fixed_points_:
            One mapping from state component name to its value per point, the fixed points found or the given points.

        eigenvalues_:
            The eigenvalues of the Jacobian at every point.

        classifications_:
            The class of every point: `stable node`, `stable focus`, `unstable node`, `unstable focus`, `saddle`,
            `centre`, or `marginal` for flows, and `stable`, `unstable`, `saddle`, or `marginal` for maps.

        system_:
            The system of one field, for `'dispersion'`.

        wavenumbers_:
            The wave vector of every mode, one row per mode, for `'dispersion'`.

        magnitudes_:
            The length of every wave vector.

        growth_rates_:
            The largest real part of the eigenvalues of the linearisation at every mode.

        frequencies_:
            The imaginary part of the leading eigenvalue at every mode, nonzero for travelling or oscillating modes.

        uniform_:
            The uniform state the model is linearised around, by state name.

        band_:
            The lengths of the wave vectors whose modes grow.

    Examples:
        >>> from skcomplex.analysis import StabilityAnalysis
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> analysis = StabilityAnalysis().check().prepare(logistic_growth()).run()
        >>> sorted(str(point['population.n']) for point in analysis.fixed_points_)
        ['0', 'K_growth']
        >>> analysis = StabilityAnalysis(points=[{'population.n': 100.0}]).check().prepare(logistic_growth())
        >>> analysis.run().classifications_
        ['stable node']
        >>> from skcomplex.models.chemistry.pattern_formation import brusselator
        >>> relation = StabilityAnalysis('dispersion').check().prepare(brusselator((64,), b=1.9, Dv=8.0)).run()
        >>> bool(relation.growth_rates_.max() > 0), bool(relation.growth_rates_[0] < 0)
        (True, True)
    """

    def __init__(
        self,
        kind: str = 'fixed-points',
        parameters: Mapping[str, float] | None = None,
        points: Sequence[Point] | None = None,
        *,
        at: Mapping[str, float] | None = None,
    ) -> None:
        self.kind = kind
        self.parameters = parameters
        self.points = points
        self.at = at

    def _check(self) -> None:
        """Validate the kind and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind])

    def _get_data_kind(self) -> DataKind:
        """Return the system kind."""
        return DataKind.SYSTEM

    def _prepare(self, source: Source) -> None:
        """Compile the system of the source, checking that it has a symbolic vector field, or read a field model."""
        if self.kind == 'dispersion':
            if not isinstance(source, System):
                error_msg = f'A dispersion relation is computed from a system, not from {type(source).__name__}.'
                raise AnalysisError(error_msg)
            self.system_ = source
            return
        self.problem_ = self._dynamics_of(self._read_system(source), self.parameters).problem

    def _find_points(self, dynamics: _DynamicsSpec) -> list[dict[str, Expression | float]]:
        """Return the fixed points, solved symbolically or else searched numerically."""
        substitution = {item: value for item, value in dynamics.parameters.items() if self.parameters}
        SymbolicAnalysis._declare_domains([*dynamics.symbols, *dynamics.parameters], dynamics.parameters)
        equations = [
            (rate - item if dynamics.discrete else rate).subs(substitution) == 0
            for item, rate in zip(dynamics.symbols, dynamics.rates, strict=True)
        ]
        try:
            solutions = solve(equations, dynamics.symbols, solution_dict=True)
        except (TypeError, ValueError, NotImplementedError):
            solutions = []
        points = [
            {name: solution[item] for name, item in zip(dynamics.names, dynamics.symbols, strict=True)}
            for solution in solutions
            if self._is_point(solution, dynamics.symbols)
        ]
        return points or self._numeric_points(dynamics)

    def _eigenvalues_at(self, dynamics: _DynamicsSpec, point: Point) -> npt.NDArray[np.complex128]:
        """Return the eigenvalues of the Jacobian at a point, with the parameters at their values."""
        location = {item: SR(point[name]) for name, item in zip(dynamics.names, dynamics.symbols, strict=True)}
        numbers = {**location, **dynamics.parameters}
        values = self.jacobian_.subs(location)
        numeric = np.array(
            [[complex(SR(entry).subs(numbers)) for entry in row] for row in values.rows()],
            dtype=complex,
        )
        return np.linalg.eigvals(numeric)

    def _run(self, inputs: object = None) -> None:
        """Derive the Jacobian, nullclines, and fixed points with their classes, or the dispersion relation."""
        if self.kind == 'dispersion':
            waves, magnitudes, growth_rates, frequencies, uniform = self._dispersion_relation(self.system_, self.at)
            self.wavenumbers_ = waves
            self.magnitudes_ = magnitudes
            self.growth_rates_ = growth_rates
            self.frequencies_ = frequencies
            self.uniform_ = uniform
            self.band_ = np.unique(np.round(magnitudes[growth_rates > 0], 12))
            return
        dynamics = self._dynamics_of(self.problem_, self.parameters)
        self.names_ = dynamics.names
        self.jacobian_ = matrix(SR, [[rate.diff(item) for item in dynamics.symbols] for rate in dynamics.rates])
        self.divergence_ = self.jacobian_.trace().simplify_full()
        self.nullclines_ = {
            name: (rate - item if dynamics.discrete else rate) == 0
            for name, item, rate in zip(dynamics.names, dynamics.symbols, dynamics.rates, strict=True)
        }
        self.fixed_points_ = (
            self._find_points(dynamics) if self.points is None else [dict(item) for item in self.points]
        )
        self.eigenvalues_ = [self._eigenvalues_at(dynamics, point) for point in self.fixed_points_]
        self.classifications_ = [self._classify(values, discrete=dynamics.discrete) for values in self.eigenvalues_]

    @staticmethod
    def _check_groups(problem: ProblemSpec) -> None:
        """Raise an error for a group above the member limit or a weighted reduction read by an element set."""
        for element in problem.elements:
            if element.members is not None and element.members > MEMBER_LIMIT:
                error_msg = (
                    f'The group {element.path} has {element.members} members, above the limit of {MEMBER_LIMIT} '
                    'members that symbolic analysis expands into one equation each. Analyse a smaller group.'
                )
                raise AnalysisError(error_msg)
        for reduction in problem.reductions:
            if reduction.weights is not None and reduction.entry.members is None and reduction.weights.shape[0] > 1:
                error_msg = (
                    f'A weighted reduction over {reduction.group} is read by an element set, which symbolic analysis '
                    'does not expand. Build the readers as a group.'
                )
                raise AnalysisError(error_msg)

    @staticmethod
    def _expand_groups(
        problem: ProblemSpec,
        parameter_values: npt.NDArray[np.float64],
    ) -> tuple[list[str], list[Expression], list[Expression], dict[Expression, float]]:
        """Return the equations of a problem with every group member and every reduction written out.

        Args:
            problem:
                The compiled problem.

            parameter_values:
                The parameter vector whose values the member parameters take.

        Returns:
            expansion:
                The state component names, their symbols, their right sides, and the values of the member parameters.

        Raises:
            AnalysisError:
                A group has more members than the limit, or an element set reads a weighted reduction.
        """
        StabilityAnalysis._check_groups(problem)
        expansion = _Expansion(parameter_values).check().prepare(problem)
        names: list[str] = []
        symbols: list[Expression] = []
        rates: list[Expression] = []
        for name, item in problem.equation_symbols.items():
            rate = SR(problem.equations[name])
            entry_name = name.partition('[')[0]
            entry = problem.states.entry(entry_name)
            if entry.members is None:
                placeholders = expansion.run((set(), 0)).expansion_
                names.append(name)
                symbols.append(item)
                rates.append(rate.subs(placeholders) if placeholders else rate)
                continue
            column = components_of(problem.symbols[entry_name]).index(item)
            owner = entry_name.rpartition('.')[0]
            own = {
                other.name
                for layout in (problem.states, problem.parameters)
                for other in layout.entries
                if other.name.rpartition('.')[0] == owner
            }
            for member in range(entry.rows):
                mapping = expansion.run((own, member)).expansion_
                names.append(entry.component_names[member * entry.columns + column])
                symbols.append(mapping[item])
                rates.append(rate.subs(mapping))
        return names, symbols, rates, expansion.values_

    @staticmethod
    def _dynamics_of(system: System | ProblemSpec, parameters: Mapping[str, float] | None = None) -> _DynamicsSpec:
        """Return the first-order equations of a system with the current values of its parameters.

        Args:
            system:
                The system, or the problem compiled from one.

            parameters:
                The values of parameters by name that replace their initial values, or `None` for none.

        Returns:
            dynamics:
                The names, symbols, and right sides of the first-order equations, with the values of the parameters.

        Raises:
            AnalysisError:
                The system is not an ODE or a discrete map, has callable laws or sums over members into single elements,
                or has a group of more than 64 members.
        """
        problem = BaseAnalysis._compiled_problem(system, 'Linear stability')
        if problem.form not in {FormKind.ODE, FormKind.DISCRETE}:
            error_msg = f'Only ODE systems and maps have a vector field to analyse, but the system is a {problem.form}.'
            raise AnalysisError(error_msg)
        if problem.has_callables or problem.has_group_sums:
            error_msg = (
                'The system has callable laws or sums over group members into single elements, which have no symbolic '
                'vector field. Write the sums as reductions, or analyse one member.'
            )
            raise AnalysisError(error_msg)
        if problem.delays:
            error_msg = (
                'The system reads delayed states, so its state space is infinite-dimensional and it has no vector '
                'field of finitely many states. Analyse its trajectories instead.'
            )
            raise AnalysisError(error_msg)
        values = problem.initial_parameters.copy()
        for name, value in (parameters or {}).items():
            if not problem.parameters.has(name):
                error_msg = (
                    f'{name} is not a parameter. Available parameters are: {", ".join(problem.parameters.names)}.'
                )
                raise AnalysisError(error_msg)
            values[problem.parameters.entry(name).indices] = value
        current = {}
        for entry in problem.parameters.entries:
            for column, item in enumerate(components_of(problem.symbols[entry.name])):
                current[item] = float(values[entry.start + column])
        names, symbols, rates, members = StabilityAnalysis._expand_groups(problem, values)
        return _DynamicsSpec(
            problem=problem,
            names=names,
            symbols=symbols,
            rates=rates,
            parameters={**current, **members},
            discrete=problem.form == FormKind.DISCRETE,
        )

    @staticmethod
    def _is_point(solution: Mapping[Expression, Expression], symbols: list[Expression]) -> bool:
        """Check that a solution fixes every state, without the free parameters `r1`, `r2` of solution families."""
        free = {str(item) for value in solution.values() for item in SR(value).variables()}
        return all(item in solution for item in symbols) and not any(re.fullmatch(r'r\d+', name) for name in free)

    @staticmethod
    def _numeric_points(dynamics: _DynamicsSpec) -> list[dict[str, Expression | float]]:
        """Return fixed points found numerically from a grid of starting points."""
        values = [
            rate.subs(dynamics.parameters) - (item if dynamics.discrete else 0)
            for item, rate in zip(dynamics.symbols, dynamics.rates, strict=True)
        ]
        functions = [value.function(*dynamics.symbols) for value in values]
        found: list[npt.NDArray[np.float64]] = []
        rng = np.random.default_rng(0)
        for start in rng.uniform(-10, 10, (50, len(dynamics.symbols))):
            result = optimize.root(lambda point: [float(function(*point)) for function in functions], start)
            if result.success and not any(np.allclose(result.x, other, atol=1e-6) for other in found):
                found.append(result.x)
        return [dict(zip(dynamics.names, map(float, point), strict=True)) for point in found]

    @staticmethod
    def _classify_map(eigenvalues: npt.NDArray[np.complex128]) -> str:
        """Return the stability class of a fixed point of a map from its eigenvalues."""
        moduli = np.abs(eigenvalues)
        if np.all(moduli < 1 - ZERO):
            return 'stable'
        if np.all(moduli > 1 + ZERO):
            return 'unstable'
        return 'saddle' if np.any(moduli < 1 - ZERO) and np.any(moduli > 1 + ZERO) else 'marginal'

    @staticmethod
    def _classify(eigenvalues: npt.NDArray[np.complex128], *, discrete: bool) -> str:
        """Return the stability class of a fixed point from its eigenvalues."""
        if discrete:
            return StabilityAnalysis._classify_map(eigenvalues)
        scale = max(1.0, float(np.abs(eigenvalues).max()))
        real = eigenvalues.real / scale
        oscillating = bool(np.any(np.abs(eigenvalues.imag) > ZERO * scale))
        if np.all(real < -ZERO):
            return 'stable focus' if oscillating else 'stable node'
        if np.all(real > ZERO):
            return 'unstable focus' if oscillating else 'unstable node'
        if np.any(real > ZERO) and np.any(real < -ZERO):
            return 'saddle'
        return 'centre' if oscillating and np.all(np.abs(real) <= ZERO) else 'marginal'

    @staticmethod
    def _with_eigenvalue(expression: Expression, eigenvalue: Expression) -> Expression:
        """Return an expression with every Laplacian, nested ones included, as a multiplication by its eigenvalue."""
        operator = expression.operator()
        if operator is None:
            return expression
        name = getattr(operator, 'name', lambda: '')()
        if name == 'laplacian':
            return eigenvalue * StabilityAnalysis._with_eigenvalue(expression.operands()[0], eigenvalue)
        if name in OPERATOR_NAMES:
            error_msg = f'The dispersion relation linearises laplacian only, but a law applies {name}.'
            raise AnalysisError(error_msg)
        return operator(*[StabilityAnalysis._with_eigenvalue(item, eigenvalue) for item in expression.operands()])

    @staticmethod
    def _laplacian_eigenvalues(field: FieldElement, lattice: LatticeSpace, waves: Array) -> Array:
        """Return the eigenvalue of the field's Laplacian at every wave vector, for its discretization."""
        spacings = np.asarray(lattice.spacings_, dtype=float)
        if field.discretization == 'spectral':
            return -np.sum(waves**2, axis=1)
        return np.sum((2 * np.cos(waves * spacings) - 2) / spacings**2, axis=1)

    @staticmethod
    def _bound_interactions(system: System, field: FieldElement) -> list[BaseInteraction]:
        """Return the interactions of a system whose every role binds one field, or raise an error for another."""
        bound = []
        for interaction in system.interactions:
            labels = [
                label
                for name in declarations_of(type(interaction), VariableKind.ROLE)
                for label in bound_labels(getattr(interaction, name) or ())
            ]
            if not labels or any(label != field.label for label in labels):
                error_msg = f'The interaction {interaction.label} binds a part other than the field {field.label}.'
                raise AnalysisError(error_msg)
            bound.append(interaction)
        return bound

    @staticmethod
    def _single_field(system: System) -> tuple[FieldElement, LatticeSpace, list[BaseInteraction]]:
        """Return the one field of a system, its periodic lattice, and its bound interactions, or raise an error."""
        fields = [element for element in system.elements if isinstance(element, FieldElement)]
        if len(fields) != 1 or len(system.elements) != 1 or system.systems:
            error_msg = 'The dispersion relation takes a system of one field on a lattice.'
            raise AnalysisError(error_msg)
        field = fields[0]
        if not isinstance(field.space, LatticeSpace) or field.space.boundary != 'periodic':
            error_msg = f'The field {field.label} is not on a periodic lattice, so its modes are not Fourier modes.'
            raise AnalysisError(error_msg)
        interactions = StabilityAnalysis._bound_interactions(system, field)
        return field, field.space.check().prepare(), interactions

    @staticmethod
    def _add_rate(rates: dict[str, Expression], names: dict[str, Expression], law: object, label: str) -> None:
        """Add a rate law of a state of a field to the rates, with its Laplacians as eigenvalues, or raise an error."""
        targets = law.check()._get_targets() if isinstance(law, RateLaw) else ()
        target = next((name for name, item in names.items() if targets and targets[0] is item), None)
        if not isinstance(law, RateLaw) or target is None:
            error_msg = f'The dispersion relation takes rate laws of the states of {label}, but got {law!r}.'
            raise AnalysisError(error_msg)
        rates[target] = SR(rates.get(target, 0) + StabilityAnalysis._with_eigenvalue(SR(law.value), EIGENVALUE))

    @staticmethod
    def _linearised_rates(
        field: FieldElement,
        interactions: Sequence[BaseInteraction],
    ) -> tuple[dict[str, Expression], dict[str, Expression], dict[str, npt.ArrayLike]]:
        """Return the rate of each state with its Laplacians as eigenvalues, the state symbols, and the field values.

        The rates add the laws of the field's element and of the interactions bound to the field.
        """
        part = field.get_declaring_part()
        states = list(declarations_of(part, VariableKind.STATE))
        values = {name: default_of(part, name) for name in states} | dict(field.values or {})
        parameters = {}
        for name in declarations_of(part, VariableKind.PARAMETER):
            value = values.get(name, default_of(part, name))
            if np.ndim(value):
                error_msg = f'The parameter {field.label}.{name} varies over the lattice, so there is no uniform state.'
                raise AnalysisError(error_msg)
            parameters[name] = float(value)
        names = {name: symbol(f'_{name}_') for name in states}
        rates: dict[str, Expression] = {}
        time = symbol('t')
        for law in member.check()._get_laws(SimpleNamespace(**names, **parameters, t=time)):
            StabilityAnalysis._add_rate(rates, names, law, field.label)
        for interaction in interactions:
            own = {
                name: getattr(interaction, name) for name in declarations_of(type(interaction), VariableKind.PARAMETER)
            }
            roles = {
                name: SimpleNamespace(**names, **parameters)
                for name in declarations_of(type(interaction), VariableKind.ROLE)
            }
            for law in interaction.check().prepare().run(SimpleNamespace(**roles, **own, t=time)).laws_:
                StabilityAnalysis._add_rate(rates, names, law, field.label)
        return {name: rates[name] for name in states if name in rates}, names, values

    @staticmethod
    def _uniform_point(
        rates: dict[str, Expression],
        variables: list[Expression],
    ) -> Callable[[list[float]], list[float]]:
        """Return the function that finds the uniform fixed point of the rates nearest a starting point."""
        local = [rate.subs({EIGENVALUE: 0}) for rate in rates.values()]

        def residual(point: Array) -> list[float]:
            substitution = dict(zip(variables, map(float, point), strict=True))
            return [float(item.subs(substitution)) for item in local]

        return lambda start: [float(item) for item in optimize.fsolve(residual, start)]

    @staticmethod
    def _eigenvalue_terms(entry: Expression, name: str) -> list[tuple[float, int]]:
        """Return the coefficients and powers of the Laplacian eigenvalue in an entry of the linearisation."""
        try:
            return [(float(coefficient), int(power)) for coefficient, power in entry.coefficients(EIGENVALUE)]
        except (TypeError, ValueError) as error:
            error_msg = f'The linearisation of {name} is not a polynomial in its Laplacians: {entry}.'
            raise AnalysisError(error_msg) from error

    @staticmethod
    def _dispersion_relation(
        system: System,
        at: Mapping[str, float] | None = None,
    ) -> tuple[Array, Array, Array, Array, dict[str, float]]:
        """Return the wave vectors, their lengths, growth rates, and frequencies, and the uniform state of a field."""
        field, lattice, interactions = StabilityAnalysis._single_field(system)
        rates, names, values = StabilityAnalysis._linearised_rates(field, interactions)
        order = list(rates)
        variables = [names[name] for name in order]
        given: dict[str, npt.ArrayLike] = {**values, **(at or {})}
        start = [float(np.mean(np.asarray(given.get(name, 0.0), dtype=np.float64))) for name in order]
        uniform = start if at is not None else StabilityAnalysis._uniform_point(rates, variables)(start)
        substitution = dict(zip(variables, uniform, strict=True))
        axes = [2 * np.pi * np.fft.fftfreq(n, d=h) for n, h in zip(lattice.shape_, lattice.spacings_, strict=True)]
        waves = np.stack([grid.ravel() for grid in np.meshgrid(*axes, indexing='ij')], axis=1)
        eigenvalues = StabilityAnalysis._laplacian_eigenvalues(field, lattice, waves)
        matrices = np.zeros((len(waves), len(order), len(order)))
        for row, a in enumerate(order):
            for column, b in enumerate(order):
                entry = SR(rates[a].diff(names[b]).subs(substitution)).expand()
                for coefficient, power in StabilityAnalysis._eigenvalue_terms(entry, f'{field.label}.{a}'):
                    matrices[:, row, column] += coefficient * eigenvalues**power
        spectra = np.linalg.eigvals(matrices)
        leading = spectra[np.arange(len(waves)), np.argmax(spectra.real, axis=1)]
        magnitudes = np.linalg.norm(waves, axis=1)
        return waves, magnitudes, leading.real, leading.imag, dict(zip(order, uniform, strict=True))
