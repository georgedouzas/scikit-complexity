"""Derive the fixed points, Jacobian, stability, nullclines, and divergence of a system symbolically."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, matrix, solve
from scipy.optimize import root

from ..core.errors import AnalysisError
from ..core.system import System
from ..engine.compile import compiled_problem
from ..engine.problems import Form, Problem, components_of
from ._base import BaseAnalysis, Source
from ._groups import _Expansion
from ._kinds import DataKind
from ._symbolic import SymbolicAnalysis

type Point = Mapping[str, Expression | float]
ZERO = 1e-9
MEMBER_LIMIT = 64


@dataclass(frozen=True)
class _Dynamics:
    """First-order equations of a system: its state symbols, their right sides, and its parameter values."""

    problem: Problem
    names: list[str]
    symbols: list[Expression]
    rates: list[Expression]
    parameters: dict[Expression, float]
    discrete: bool

    @property
    def time(self) -> Expression:
        """Time symbol of the equations."""
        return self.problem.time


class StabilityAnalysis(BaseAnalysis):
    """Fixed points of a flow or a map, their linear stability, and the Jacobian, nullclines, and divergence.

    The fixed points of a flow are where every rate is 0, and those of a map where every state equals its next value.
    Sage solves the equations symbolically, and when it finds no closed form, roots are searched numerically from a
    grid of starting points with the parameters at their values. The stability of every point comes from the
    eigenvalues of the Jacobian there, evaluated at the parameter values.

    Args:
        parameters:
            Values of parameters by dotted path that replace the system's values. They are substituted before solving,
            and without them the symbolic fixed points keep the parameters as symbols.

        points:
            The points whose stability to analyse, each a mapping from state component name to its value, symbolic
            or numeric, or `None` for every fixed point found.

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

    Examples:
        >>> from skcomplex.analysis import StabilityAnalysis
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> analysis = StabilityAnalysis().check().prepare(logistic_growth()).run()
        >>> sorted(str(point['population.n']) for point in analysis.fixed_points_)
        ['0', 'K_growth']
        >>> analysis = StabilityAnalysis(points=[{'population.n': 100.0}]).check().prepare(logistic_growth())
        >>> analysis.run().classifications_
        ['stable node']
    """

    def __init__(
        self,
        parameters: Mapping[str, float] | None = None,
        points: Sequence[Point] | None = None,
    ) -> None:
        self.parameters = parameters
        self.points = points

    def _get_data_kind(self) -> DataKind:
        """Return the system kind."""
        return DataKind.SYSTEM

    def _prepare(self, source: Source) -> Self:
        """Compile the system of the source, checking that it has a symbolic vector field."""
        self.problem_ = self._dynamics_of(self._read_system(source), self.parameters).problem
        return self

    def _find_points(self, dynamics: _Dynamics) -> list[dict[str, Expression | float]]:
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

    def _eigenvalues_at(self, dynamics: _Dynamics, point: Point) -> npt.NDArray[np.complex128]:
        """Return the eigenvalues of the Jacobian at a point, with the parameters at their values."""
        location = {item: SR(point[name]) for name, item in zip(dynamics.names, dynamics.symbols, strict=True)}
        numbers = {**location, **dynamics.parameters}
        values = self.jacobian_.subs(location)
        numeric = np.array(
            [[complex(SR(entry).subs(numbers)) for entry in row] for row in values.rows()],
            dtype=complex,
        )
        return np.linalg.eigvals(numeric)

    def _run(self) -> Self:
        """Derive the Jacobian, divergence, and nullclines, find the fixed points, and classify them."""
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
        return self

    @staticmethod
    def _check_groups(problem: Problem) -> None:
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
        problem: Problem,
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
        expansion = _Expansion(problem, parameter_values)
        names: list[str] = []
        symbols: list[Expression] = []
        rates: list[Expression] = []
        for name, item in problem.equation_symbols.items():
            rate = SR(problem.equations[name])
            entry_name = name.partition('[')[0]
            entry = problem.states.entry(entry_name)
            if entry.members is None:
                placeholders = {
                    reduction.symbol: expansion.value(reduction, 0 if reduction.weights is not None else None)
                    for reduction in problem.reductions
                }
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
                mapping = expansion.member_map(own, member)
                mapping.update(
                    {
                        reduction.symbol: expansion.value(reduction, member if reduction.weights is not None else None)
                        for reduction in problem.reductions
                    },
                )
                names.append(entry.component_names[member * entry.columns + column])
                symbols.append(expansion.member_symbol(entry, column, member))
                rates.append(rate.subs(mapping))
        return names, symbols, rates, expansion.values

    @staticmethod
    def _dynamics_of(system: System | Problem, parameters: Mapping[str, float] | None = None) -> _Dynamics:
        """Return the first-order equations of a system with the current values of its parameters.

        Raises:
            AnalysisError:
                The system is not an ODE or a discrete map, has callable laws or sums over members into single elements,
                or has a group of more than 64 members.
        """
        problem = compiled_problem(system, 'Linear stability')
        if problem.form not in {Form.ODE, Form.DISCRETE}:
            error_msg = (
                f'Only ODE systems and maps have a vector field to analyse, but the system is a {problem.form.value}.'
            )
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
        return _Dynamics(
            problem=problem,
            names=names,
            symbols=symbols,
            rates=rates,
            parameters={**current, **members},
            discrete=problem.form == Form.DISCRETE,
        )

    @staticmethod
    def _is_point(solution: Mapping[Expression, Expression], symbols: list[Expression]) -> bool:
        """Check that a solution fixes every state, without the free parameters `r1`, `r2`, ... of solution families."""
        free = {str(item) for value in solution.values() for item in SR(value).variables()}
        return all(item in solution for item in symbols) and not any(re.fullmatch(r'r\d+', name) for name in free)

    @staticmethod
    def _numeric_points(dynamics: _Dynamics) -> list[dict[str, Expression | float]]:
        """Return fixed points found numerically from a grid of starting points."""
        values = [
            rate.subs(dynamics.parameters) - (item if dynamics.discrete else 0)
            for item, rate in zip(dynamics.symbols, dynamics.rates, strict=True)
        ]
        functions = [value.function(*dynamics.symbols) for value in values]
        found: list[npt.NDArray[np.float64]] = []
        rng = np.random.default_rng(0)
        for start in rng.uniform(-10, 10, (50, len(dynamics.symbols))):
            result = root(lambda point: [float(function(*point)) for function in functions], start)
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
