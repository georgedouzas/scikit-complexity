"""Simulate isotope labelling by elementary metabolite units and estimate fluxes from measured labelling patterns."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
from scipy.stats import chi2

from ....analysis import BaseAnalysis
from ....analysis.types import Source
from ....core.spaces import StaticTime
from ....core.system import System
from ....engine.compile import compile_system
from ....engine.problems import ProblemSpec
from ....errors import AnalysisError, ValidationError
from ....specs import BaseSpec, RunStateSpec
from ....types import Array, DataKind
from ..common.metabolism import MetabolicNetworkElement
from ._labelling_specs import AtomMapSpec, FragmentSpec, LabelMeasurementSpec, TracerSpec

type _Unit = tuple[str, tuple[int, ...]]

KINDS = ('simulation', 'estimation')
INTERVALS = ('linear', 'profile')
CONFIDENCE = 0.95
INTERVAL_THRESHOLD = 3.841458820694124
NORMAL_QUANTILE = 1.959963984540054
PENALTY = 1e3
FIT_TOLERANCE = 1e-13
MAXIMUM_EVALUATIONS = 400
PROFILE_ITERATIONS = 12
PROFILE_TOLERANCE = 1e-9
PROFILE_RELATIVE_STEP = 0.1
PROFILE_WEIGHT = 1e4
BOUND_TOLERANCE = 1e-6
PROFILE_WIDTH = 1e-3
VERTEX_DRAWS = 3


@dataclass(frozen=True)
class _TermSpec(BaseSpec):
    """Inflow to an elementary metabolite unit: its directed flux, its weight, and the units it combines."""

    flux: int
    weight: float
    substrates: tuple[_Unit, ...]


class LabellingAnalysis(BaseAnalysis):
    """Isotope labelling of a metabolic network at metabolic and isotopic steady state, of one kind.

    The analysis takes a system that holds a metabolic network. It walks the atom maps backward from the fragments to
    collect the elementary metabolite units (Antoniewicz et al. 2007), and solves per unit size one sparse linear system
    with `scipy.sparse.linalg.splu`.

    - `'simulation'`: the mass isotopomer distributions of the fragments for given fluxes.
    - `'estimation'`: the fluxes by weighted least squares over measured distributions and measured fluxes, from seeded
      feasible starts that the core solve draws. The net fluxes run over a null space basis of the balances, with one
      exchange flux per reversible reaction, so every balance holds exactly. A 95% interval of every flux and a
      chi-square test of the fit follow.

    Args:
        kind:
            `'simulation'` or `'estimation'`.

        atoms:
            The atom map of every reaction that moves traced atoms.

        tracers:
            The labelled substrates of every parallel experiment, by the name of the experiment.

        fragments:
            The fragments to simulate or that measurements name.

        measurements:
            The measured distributions, for `'estimation'`.

        fluxes:
            The flux of every reaction for `'simulation'`, a number, or a pair of a forward and a backward flux for a
            reversible reaction.

        measured_fluxes:
            The value and the standard deviation of every measured flux for `'estimation'`, by reaction. A deviation of
            zero fixes the flux.

        symmetric:
            The metabolites whose atoms a producing reaction scrambles, half in each direction.

        starts:
            The number of feasible starts of the fit.

        intervals:
            `'linear'` for intervals of the linearised fit, or `'profile'` for profiled ones (Antoniewicz et al. 2006).

        random_state:
            The seed or generator of the starts.

    Attributes:
        distributions_:
            The distribution of every fragment of every experiment.

        fluxes_:
            The estimated net flux of every reaction, and the exchange `<reaction>.exchange` of a reversible one.

        intervals_:
            The 95% interval of every estimated flux.

        identifiable_:
            Whether the data identify every estimated flux, false for an interval that reaches a bound.

        residual_:
            The sum of the squares of the weighted residuals at the optimum.

        chi2_:
            The range of the sum of squares that the test accepts at the 5% level.

        accepted_:
            Whether the residual lies in that range.

    Raises:
        AnalysisError:
            A setting is unknown or a metabolite has no source of labels, raised by `prepare`.

        ValidationError:
            An atom map is not valid, or two maps give one metabolite different numbers of atoms, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.spaces import StaticTime
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.common.metabolism import ReactionSpec, metabolic_network_type
        >>> from skcomplex.models.biology.systems_biology import (
        ...     AtomMapSpec,
        ...     FragmentSpec,
        ...     LabellingAnalysis,
        ...     TracerSpec,
        ... )
        >>> reactions = {'in': ReactionSpec({'B': 1.0}), 'out': ReactionSpec({'B': -1.0})}
        >>> network = metabolic_network_type('ChainElement', reactions)('net')
        >>> analysis = LabellingAnalysis(
        ...     atoms=[AtomMapSpec('in', [('A', 'ab')], [('B', 'ba')])],
        ...     tracers={'first': [TracerSpec('A', {'00': 0.5, '01': 0.5})]},
        ...     fragments=[FragmentSpec('B', (1, 2))],
        ...     fluxes={'in': 1.0, 'out': 1.0},
        ... )
        >>> system = System(elements=[network], time=StaticTime())
        >>> analysis.prepare(system).run().distributions_['first']['B'].tolist()
        [0.5, 0.5, 0.0]
    """

    def __init__(
        self,
        kind: str = 'simulation',
        *,
        atoms: Sequence[AtomMapSpec] = (),
        tracers: Mapping[str, Sequence[TracerSpec]] | None = None,
        fragments: Sequence[FragmentSpec] = (),
        measurements: Sequence[LabelMeasurementSpec] = (),
        fluxes: Mapping[str, float | tuple[float, float]] | None = None,
        measured_fluxes: Mapping[str, tuple[float, float]] | None = None,
        symmetric: Sequence[str] = (),
        starts: int = 5,
        intervals: str = 'linear',
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.kind = kind
        self.atoms = atoms
        self.tracers = tracers
        self.fragments = fragments
        self.measurements = measurements
        self.fluxes = fluxes
        self.measured_fluxes = measured_fluxes
        self.symmetric = symmetric
        self.starts = starts
        self.intervals = intervals
        self.random_state = random_state

    def _get_data_kind(self) -> DataKind:
        """Return the kind of data the analysis reads, the network of a system."""
        return DataKind.SERIES

    def _check(self) -> None:
        """Raise an error for an unknown setting, an invalid atom map, or inconsistent atom counts.

        Raises:
            AnalysisError:
                The kind or the intervals are unknown.

            ValidationError:
                An atom map is not valid, or two reactions give one metabolite different numbers of atoms.
        """
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('intervals', self.intervals, INTERVALS)
        counts: dict[str, tuple[int, str]] = {}
        for item in self.atoms:
            item._validated()
            for metabolite, letters in (*item.reactants, *item.products):
                known = counts.setdefault(metabolite, (len(letters), item.reaction))
                if known[0] != len(letters):
                    error_msg = (
                        f'The reaction {known[1]} gives {metabolite} {known[0]} atoms, but the reaction '
                        f'{item.reaction} gives it {len(letters)}.'
                    )
                    raise ValidationError(error_msg)
        self._atom_counts = {metabolite: count for metabolite, (count, _) in counts.items()}

    def _network_of(self, system: System) -> MetabolicNetworkElement:
        """Return the metabolic network of a system."""
        for element in system.elements:
            if isinstance(element, MetabolicNetworkElement):
                return element
        error_msg = 'The analysis reads a system that holds a metabolic network element.'
        raise AnalysisError(error_msg)

    def _directed_maps(self) -> list[tuple[int, AtomMapSpec]]:
        """Return the maps in both directions, each with the position of its directed flux."""
        reactions = list(self.network_.reactions)
        count = len(reactions)
        directed = []
        for item in self.atoms:
            if item.reaction not in reactions:
                error_msg = f'An atom map names the reaction {item.reaction}, which the network does not hold.'
                raise AnalysisError(error_msg)
            position = reactions.index(item.reaction)
            directed.append((position, item))
            if self._reversible(item.reaction):
                reverse = AtomMapSpec(item.reaction, item.products, item.reactants)
                directed.append((position + count, reverse))
        return directed

    def _bounds_of(self, reaction: str) -> tuple[float, float]:
        """Return the bounds of a reaction, from the parameters of the network."""
        identifier = self.network_._identifier(reaction)
        return self.network_.get_number(f'lower_{identifier}'), self.network_.get_number(f'upper_{identifier}')

    def _reversible(self, reaction: str) -> bool:
        """Check whether a reaction may run backward."""
        return self._bounds_of(reaction)[0] < 0

    def _sources(self) -> set[str]:
        """Return the metabolites that tracers label in any experiment."""
        return {item.metabolite for tracers in (self.tracers or {}).values() for item in tracers}

    def _terms_of(self, unit: _Unit, maps: Sequence[tuple[int, AtomMapSpec]]) -> list[_TermSpec]:
        """Return the inflows to a unit from every reaction that produces its metabolite."""
        metabolite, atoms = unit
        size = self._atom_counts.get(metabolite, max(atoms))
        scrambled = tuple(sorted(size + 1 - atom for atom in atoms))
        choices = [(atoms, 0.5), (scrambled, 0.5)] if metabolite in self.symmetric else [(atoms, 1.0)]
        terms = []
        for flux, item in maps:
            places = {
                letter: (name, position + 1)
                for name, letters in item.reactants
                for position, letter in enumerate(letters)
            }
            for name, letters in item.products:
                if name != metabolite:
                    continue
                for chosen, weight in choices:
                    grouped: dict[str, list[int]] = {}
                    for atom in chosen:
                        origin, place = places[letters[atom - 1]]
                        grouped.setdefault(origin, []).append(place)
                    substrates = tuple(sorted((origin, tuple(sorted(found))) for origin, found in grouped.items()))
                    terms.append(_TermSpec(flux, weight, substrates))
        return terms

    def _collect_units(self) -> None:
        """Walk the atom maps backward from the fragments, and set the terms and the sizes of the units."""
        maps = self._directed_maps()
        sources = self._sources()
        terms: dict[_Unit, list[_TermSpec]] = {}
        leaves: set[_Unit] = set()
        stack: list[_Unit] = [(item.metabolite, tuple(sorted(item.atoms))) for item in self.fragments]
        while stack:
            unit = stack.pop()
            if unit in terms or unit in leaves:
                continue
            if unit[0] in sources:
                leaves.add(unit)
                continue
            found = self._terms_of(unit, maps)
            if not found:
                error_msg = (
                    f'No reaction produces {unit[0]} and no tracer labels it. Give it a tracer, even an unlabelled one.'
                )
                raise AnalysisError(error_msg)
            terms[unit] = found
            stack += [substrate for term in found for substrate in term.substrates]
        self._terms = terms
        self._leaves = leaves
        self._sizes = sorted({len(unit[1]) for unit in terms})

    def _prepare(self, context: Source) -> None:
        """Read the network of a system and collect the elementary metabolite units of the fragments.

        Args:
            context:
                The system.

        Raises:
            AnalysisError:
                The source is not a system with a metabolic network, or a metabolite has no source of labels.
        """
        if not isinstance(context, System):
            error_msg = f'The analysis reads a system, not a {type(context).__name__}.'
            raise AnalysisError(error_msg)
        self.network_ = self._network_of(context)
        self._reactions = list(self.network_.reactions)
        self._collect_units()

    @staticmethod
    def _source_distribution(tracer: TracerSpec, atoms: Sequence[int]) -> Array:
        """Return the distribution of the labelled atoms of a tracer among some atoms."""
        distribution = np.zeros(len(atoms) + 1)
        for isotopomer, fraction in tracer.fractions.items():
            distribution[sum(isotopomer[atom - 1] == '1' for atom in atoms)] += fraction
        return distribution

    def _source_units(self, tracers: Sequence[TracerSpec]) -> dict[_Unit, Array]:
        """Return the distribution of every leaf unit in one experiment."""
        found = {item.metabolite: item for item in tracers}
        known = {}
        for unit in self._leaves:
            tracer = found.get(unit[0])
            unlabelled = np.zeros(len(unit[1]) + 1)
            unlabelled[0] = 1.0
            known[unit] = unlabelled if tracer is None else self._source_distribution(tracer, unit[1])
        return known

    def _solve_size(self, size: int, flows: Array, known: dict[_Unit, Array]) -> None:
        """Solve the units of one size, `A X = B`, and add their distributions to the known ones."""
        units = [unit for unit in self._terms if len(unit[1]) == size]
        positions = {unit: position for position, unit in enumerate(units)}
        rows, columns, values = [], [], []
        right = np.zeros((len(units), size + 1))
        totals = np.zeros(len(units))
        for unit, position in positions.items():
            for term in self._terms[unit]:
                rate = term.weight * flows[term.flux]
                totals[position] += rate
                if len(term.substrates) == 1 and term.substrates[0] in positions:
                    rows.append(position)
                    columns.append(positions[term.substrates[0]])
                    values.append(-rate)
                    continue
                combined = np.ones(1)
                for substrate in term.substrates:
                    combined = np.convolve(combined, known[substrate])
                right[position] += rate * combined
        closed = totals <= 0
        totals[closed] = 1.0
        right[closed, 0] = 1.0
        matrix = sparse.coo_array((values, (rows, columns)), shape=(len(units), len(units))).tocsc()
        matrix = (matrix + sparse.diags_array(totals, format='csc')).tocsc()
        solution = splu(matrix).solve(right)
        known.update({unit: solution[position] for unit, position in positions.items()})

    def _simulated(self, flows: Array, tracers: Sequence[TracerSpec]) -> dict[str, Array]:
        """Return the distribution of every fragment at directed fluxes in one experiment."""
        known = self._source_units(tracers)
        for size in self._sizes:
            self._solve_size(size, flows, known)
        return {item.label: known[(item.metabolite, tuple(sorted(item.atoms)))] for item in self.fragments}

    def _directed_given(self) -> Array:
        """Return the directed fluxes that the `fluxes` setting gives."""
        count = len(self._reactions)
        flows = np.zeros(2 * count)
        for reaction, value in (self.fluxes or {}).items():
            position = self._reactions.index(reaction)
            forward, backward = value if isinstance(value, tuple) else (value, 0.0)
            flows[position], flows[position + count] = forward, backward
        return flows

    def _run_simulation(self) -> None:
        """Set the distributions at the given fluxes."""
        flows = self._directed_given()
        self.distributions_ = {
            experiment: self._simulated(flows, tracers) for experiment, tracers in (self.tracers or {}).items()
        }

    def _balances(self) -> Array:
        """Return the matrix of the balances of the network over its net fluxes."""
        metabolites = list(
            dict.fromkeys(name for spec in self.network_.reactions.values() for name in spec.coefficients),
        )
        matrix = np.zeros((len(metabolites), len(self._reactions)))
        for column, spec in enumerate(self.network_.reactions.values()):
            for name, value in spec.coefficients.items():
                matrix[metabolites.index(name), column] = value
        return matrix

    def _set_parametrization(self) -> None:
        """Set a particular solution, the null space basis of the net fluxes, and the exchanges."""
        balances = self._balances()
        fixed = {name: value for name, (value, deviation) in (self.measured_fluxes or {}).items() if deviation == 0}
        rows = np.zeros((len(fixed), len(self._reactions)))
        for row, name in enumerate(fixed):
            rows[row, self._reactions.index(name)] = 1.0
        system = np.vstack([balances, rows])
        right = np.concatenate([np.zeros(balances.shape[0]), np.fromiter(fixed.values(), dtype=float)])
        self._particular = linalg.lstsq(system, right)[0]
        self._basis = linalg.null_space(system)
        self._exchanges = [name for name in self._reactions if self._reversible(name)]
        self._lower = np.asarray([self._bounds_of(name)[0] for name in self._reactions], dtype=float)
        self._upper = np.asarray([self._bounds_of(name)[1] for name in self._reactions], dtype=float)
        self._exchange_limit = np.asarray([self._bounds_of(name)[1] for name in self._exchanges], dtype=float)

    def _net_of(self, parameters: Array) -> Array:
        """Return the net fluxes of parameters."""
        return self._particular + self._basis @ parameters[: self._basis.shape[1]]

    def _flows_of(self, parameters: Array) -> Array:
        """Return the directed fluxes of parameters, the exchanges added in both directions."""
        net = self._net_of(parameters)
        count = len(self._reactions)
        flows = np.concatenate([np.maximum(net, 0.0), np.maximum(-net, 0.0)])
        exchanges = parameters[self._basis.shape[1] :]
        for name, exchange in zip(self._exchanges, exchanges, strict=True):
            position = self._reactions.index(name)
            flows[position] += exchange
            flows[position + count] += exchange
        return flows

    def _residuals(self, parameters: Array) -> Array:
        """Return the weighted residuals of the measurements and the measured fluxes, and the penalties of bounds."""
        flows = self._flows_of(parameters)
        simulated = {
            experiment: self._simulated(flows, tracers) for experiment, tracers in (self.tracers or {}).items()
        }
        terms = [
            (simulated[item.experiment][item.fragment] - item.distribution) / item.deviation
            for item in self.measurements
        ]
        net = self._net_of(parameters)
        measured = [
            (net[self._reactions.index(name)] - value) / deviation
            for name, (value, deviation) in (self.measured_fluxes or {}).items()
            if deviation > 0
        ]
        penalties = PENALTY * np.concatenate([np.minimum(net - self._lower, 0.0), np.maximum(net - self._upper, 0.0)])
        return np.concatenate([*terms, np.asarray(measured, dtype=float), penalties])

    def _drawn_vertices(self, rng: np.random.Generator) -> Array:
        """Return a feasible net flux, a random combination of vertices that the core solve finds."""
        fixed = {name: value for name, (value, deviation) in (self.measured_fluxes or {}).items() if deviation == 0}
        identifiers = {
            **{f'lower_{self.network_._identifier(name)}': value for name, value in fixed.items()},
            **{f'upper_{self.network_._identifier(name)}': value for name, value in fixed.items()},
        }
        vertices = []
        for _ in range(VERTEX_DRAWS):
            weights = {
                name: float(value)
                for name, value in zip(self._reactions, rng.normal(size=len(self._reactions)), strict=True)
            }
            element = type(self.network_)(
                self.network_.label,
                objective=weights,
                reactions=self.network_.reactions,
                default_objective=self.network_.default_objective,
            ).set_params(**identifiers)
            problem = compile_system(System(elements=[element], time=StaticTime()))
            if not isinstance(problem, ProblemSpec):
                error_msg = f'The network compiles to a {type(problem).__name__}, not a linear program.'
                raise AnalysisError(error_msg)
            state = RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)
            solved = problem.programs[0].run(state).values_
            vertices.append(
                [
                    float(
                        np.real(np.ravel(problem.states.read(solved, f'{element.label}.{element.flux_name(name)}'))[0]),
                    )
                    for name in self._reactions
                ],
            )
        shares = rng.dirichlet(np.ones(VERTEX_DRAWS))
        return shares @ np.asarray(vertices, dtype=float)

    def _start(self, rng: np.random.Generator) -> Array:
        """Return a feasible start: the coordinates of a drawn flux and random exchanges."""
        net = self._drawn_vertices(rng)
        coordinates = self._basis.T @ (net - self._particular)
        return np.concatenate([coordinates, rng.uniform(0.0, self._exchange_limit / 2.0)])

    def _fit(self, start: Array, extra: tuple[int, float] | None = None) -> tuple[Array, float]:
        """Return the parameters that minimise the residuals from a start, and the sum of squares of the residuals."""
        count = self._basis.shape[1]
        lower = np.concatenate([np.full(count, -np.inf), np.zeros(len(self._exchanges))])
        upper = np.concatenate([np.full(count, np.inf), self._exchange_limit])

        def residuals(parameters: Array) -> Array:
            found = self._residuals(parameters)
            if extra is None:
                return found
            target = self._estimate_of(parameters)[extra[0]]
            return np.concatenate([found, [PROFILE_WEIGHT * (target - extra[1])]])

        tolerance = FIT_TOLERANCE if extra is None else PROFILE_TOLERANCE
        result = least_squares(
            residuals,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
            max_nfev=MAXIMUM_EVALUATIONS,
        )
        return result.x, float(np.sum(self._residuals(result.x) ** 2))

    def _estimate_of(self, parameters: Array) -> Array:
        """Return the net fluxes and the exchanges of parameters."""
        return np.concatenate([self._net_of(parameters), parameters[self._basis.shape[1] :]])

    def _estimate_names(self) -> list[str]:
        """Return the names of the net fluxes and the exchanges."""
        return [*self._reactions, *(f'{name}.exchange' for name in self._exchanges)]

    def _bounds_of_estimate(self) -> tuple[Array, Array]:
        """Return the bounds of the net fluxes and the exchanges."""
        count = len(self._exchanges)
        return (
            np.concatenate([self._lower, np.zeros(count)]),
            np.concatenate([self._upper, self._exchange_limit]),
        )

    def _linear_intervals(self, parameters: Array) -> tuple[Array, Array]:
        """Return the intervals of the linearised fit."""
        steps = 1e-6 * np.maximum(1.0, np.abs(parameters))
        center = self._residuals(parameters)
        jacobian = np.zeros((center.size, parameters.size))
        for column, step in enumerate(steps):
            shifted = parameters.copy()
            shifted[column] += step
            jacobian[:, column] = (self._residuals(shifted) - center) / step
        covariance = np.linalg.pinv(jacobian.T @ jacobian)
        mapping = np.zeros((len(self._estimate_names()), parameters.size))
        count = self._basis.shape[1]
        mapping[: self._basis.shape[0], :count] = self._basis
        mapping[self._basis.shape[0] :, count:] = np.eye(parameters.size - count)
        deviations = np.sqrt(np.maximum(np.diag(mapping @ covariance @ mapping.T), 0.0))
        estimate = self._estimate_of(parameters)
        lower, upper = self._bounds_of_estimate()
        return (
            np.maximum(estimate - NORMAL_QUANTILE * deviations, lower),
            np.minimum(estimate + NORMAL_QUANTILE * deviations, upper),
        )

    def _profile_side(self, parameters: Array, position: int, best: float, limit: float) -> float:
        """Return where the sum of squares rises by the threshold on the way from the optimum to a bound."""
        estimate = float(self._estimate_of(parameters)[position])
        step = max(PROFILE_RELATIVE_STEP * abs(estimate), 1.0)
        direction = 1.0 if limit > estimate else -1.0
        inner, inner_start = estimate, parameters
        outer = estimate
        while True:
            outer = min(outer + step, limit) if direction > 0 else max(outer - step, limit)
            fitted, value = self._fit(inner_start, (position, outer))
            if value - best >= INTERVAL_THRESHOLD:
                break
            inner, inner_start = outer, fitted
            if outer == limit:
                return limit
            step *= 2.0
        for _ in range(PROFILE_ITERATIONS):
            middle = 0.5 * (inner + outer)
            fitted, value = self._fit(inner_start, (position, middle))
            if value - best >= INTERVAL_THRESHOLD:
                outer = middle
            else:
                inner, inner_start = middle, fitted
            if abs(outer - inner) <= PROFILE_WIDTH * max(1.0, abs(inner)):
                break
        return inner

    def _profile_intervals(self, parameters: Array, best: float) -> tuple[Array, Array]:
        """Return the intervals where the sum of squares stays within the threshold of its optimum."""
        lower, upper = self._bounds_of_estimate()
        low = np.empty_like(lower)
        high = np.empty_like(upper)
        for position in range(low.size):
            low[position] = self._profile_side(parameters, position, best, float(lower[position]))
            high[position] = self._profile_side(parameters, position, best, float(upper[position]))
        return low, high

    def _set_estimates(self, parameters: Array, best: float) -> None:
        """Set the estimates, their intervals, the test of the fit, and the distributions at the optimum."""
        names = self._estimate_names()
        low, high = (
            self._profile_intervals(parameters, best)
            if self.intervals == 'profile'
            else self._linear_intervals(parameters)
        )
        estimate = self._estimate_of(parameters)
        lower, upper = self._bounds_of_estimate()
        self.fluxes_ = dict(zip(names, estimate.tolist(), strict=True))
        self.intervals_ = {name: (float(a), float(b)) for name, a, b in zip(names, low, high, strict=True)}
        self.identifiable_ = {
            name: bool(a > bottom + BOUND_TOLERANCE and b < top - BOUND_TOLERANCE) or bottom == top
            for name, a, b, bottom, top in zip(names, low, high, lower, upper, strict=True)
        }
        self.residual_ = best
        size = sum(item.distribution.size for item in self.measurements)
        size += sum(1 for _, deviation in (self.measured_fluxes or {}).values() if deviation > 0)
        freedom = max(size - parameters.size, 1)
        self.chi2_ = (float(chi2.ppf(0.025, freedom)), float(chi2.ppf(0.975, freedom)))
        self.accepted_ = bool(self.chi2_[0] <= best <= self.chi2_[1])
        flows = self._flows_of(parameters)
        self.distributions_ = {
            experiment: self._simulated(flows, tracers) for experiment, tracers in (self.tracers or {}).items()
        }

    def _run_estimation(self) -> None:
        """Fit the fluxes from seeded feasible starts and set the estimates of the best fit."""
        self._set_parametrization()
        rng = np.random.default_rng(self.random_state)
        fits = [self._fit(self._start(rng)) for _ in range(int(self.starts))]
        parameters, best = min(fits, key=lambda item: item[1])
        self._set_estimates(parameters, best)

    def _run(self, inputs: object = None) -> None:
        """Run the kind of the analysis.

        Args:
            inputs:
                Not read.
        """
        if self.kind == 'simulation':
            self._run_simulation()
        else:
            self._run_estimation()
