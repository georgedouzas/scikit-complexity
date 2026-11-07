"""Carbon-13 metabolic flux analysis of the TCA example of Antoniewicz et al. (2007)."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy import linalg, sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
from scipy.stats import chi2

from ....core.analysis import bind_analysis
from ....core.analysis.types import Source
from ....core.components.spaces import StaticTime
from ....core.components.system import System
from ....core.engine.compile import PassCompiler
from ....core.engine.problems import ProblemSpec
from ....core.errors import AnalysisError, ValidationError
from ....core.specs import BaseSpec, DeclaredVariableSpec, RunStateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ..common.metabolism import MetabolicNetworkElement, ReactionSpec

SUM_TOLERANCE = 1e-9

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

BOUND = 300.0

DEVIATION = 0.002

TCA_REACTIONS = {
    'v1': ReactionSpec({'OAC': -1.0, 'Cit': 1.0}, upper=BOUND),
    'v2': ReactionSpec({'Cit': -1.0, 'AKG': 1.0}, upper=BOUND),
    'v3': ReactionSpec({'AKG': -1.0, 'Glu': 1.0}, upper=BOUND),
    'v4': ReactionSpec({'AKG': -1.0, 'Suc': 1.0}, upper=BOUND),
    'v5': ReactionSpec({'Suc': -1.0, 'Fum': 1.0}, upper=BOUND),
    'v6': ReactionSpec({'Fum': -1.0, 'OAC': 1.0}, lower=-BOUND, upper=BOUND),
    'v8': ReactionSpec({'OAC': 1.0}, upper=BOUND),
    'v9': ReactionSpec({'Glu': -1.0}, upper=BOUND),
}
"""The net reactions of the TCA example, with `v6` the fumarase whose exchange the labelling reads."""

TCA_TRUE_FLUXES: dict[str, float | tuple[float, float]] = {
    'v1': 100.0,
    'v2': 100.0,
    'v3': 50.0,
    'v4': 50.0,
    'v5': 50.0,
    'v6': (125.0, 75.0),
    'v8': 50.0,
    'v9': 50.0,
}

TCA_NET_FLUXES = {'v1': 100.0, 'v2': 100.0, 'v3': 50.0, 'v4': 50.0, 'v5': 50.0, 'v6': 50.0, 'v8': 50.0, 'v9': 50.0}

TCA_SYMMETRIC = ('Suc', 'Fum')

TCA_OBJECTIVE: Mapping[str, float] = {}


class TcaNetworkElement(MetabolicNetworkElement):
    """The TCA example of Antoniewicz et al. (2007).

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for the bars of its program.

        objective:
            The coefficient of every reaction in the objective, or `None` for `default_objective`.

        parsimonious:
            Whether to minimise the total absolute flux after the objective.

        infeasible:
            The infeasibility policy of the program.

        name:
            The name of the network.

        reactions:
            The reactions by name.

        default_objective:
            The objective of the network when `objective` is `None`.

        variables:
            The declared states and parameters that replace the ones of the reactions, by name, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import carbon_13_mfa
        >>> carbon_13_mfa().elements[0].get_value('upper_v1')
        300.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        objective: Mapping[str, float] | None = None,
        parsimonious: bool = False,
        infeasible: str = 'raise',
        name: str | None = 'TcaNetwork',
        reactions: Mapping[str, ReactionSpec] = TCA_REACTIONS,
        default_objective: Mapping[str, float] = TCA_OBJECTIVE,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(
            label,
            visual,
            objective=objective,
            parsimonious=parsimonious,
            infeasible=infeasible,
            name=name,
            reactions=reactions,
            default_objective=default_objective,
            variables=variables,
        )


@dataclass(frozen=True)
class AtomMapSpec(BaseSpec):
    """Atom transitions of a reaction, a string of atom letters for every occurrence of a metabolite.

    Args:
        reaction:
            The name of the reaction.

        reactants:
            Every occurrence of a reactant with its letters, as pairs of a metabolite and a string.

        products:
            Every occurrence of a product with its letters, as pairs of a metabolite and a string.

    Raises:
        ValidationError:
            A product names a letter that no reactant carries, or a reactant repeats a letter, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import AtomMapSpec
        >>> atoms = AtomMapSpec('v1', [('AcCoA', 'ab'), ('OAC', 'cdef')], [('Cit', 'fedbac')])
        >>> atoms.origins[('Cit', 1)], atoms.origins[('Cit', 5)]
        (('OAC', 4), ('AcCoA', 1))
    """

    reaction: str
    reactants: Sequence[tuple[str, str]]
    products: Sequence[tuple[str, str]]

    def _check(self) -> None:
        """Raise an error for a product letter that no reactant carries, or a letter two reactant atoms share.

        Raises:
            ValidationError:
                A letter is missing or repeated.
        """
        letters = [letter for _, atoms in self.reactants for letter in atoms]
        for letter in letters:
            if letters.count(letter) > 1:
                error_msg = f'The reactants of the reaction {self.reaction} repeat the atom letter {letter!r}.'
                raise ValidationError(error_msg)
        for metabolite, atoms in self.products:
            for letter in atoms:
                if letter not in letters:
                    error_msg = (
                        f'The product {metabolite} of the reaction {self.reaction} names the atom letter {letter!r}, '
                        'which no reactant carries.'
                    )
                    raise ValidationError(error_msg)

    @property
    def origins(self) -> dict[tuple[str, int], tuple[str, int]]:
        """Reactant atom of every product atom, as pairs of a metabolite and an atom counted from 1."""
        places = {
            letter: (metabolite, position + 1)
            for metabolite, atoms in self.reactants
            for position, letter in enumerate(atoms)
        }
        return {
            (metabolite, position + 1): places[letter]
            for metabolite, atoms in self.products
            for position, letter in enumerate(atoms)
        }


TCA_ATOMS = (
    AtomMapSpec('v1', [('AcCoA', 'ab'), ('OAC', 'cdef')], [('Cit', 'fedbac')]),
    AtomMapSpec('v2', [('Cit', 'abcdef')], [('AKG', 'abcde')]),
    AtomMapSpec('v3', [('AKG', 'abcde')], [('Glu', 'abcde')]),
    AtomMapSpec('v4', [('AKG', 'abcde')], [('Suc', 'bcde')]),
    AtomMapSpec('v5', [('Suc', 'abcd')], [('Fum', 'abcd')]),
    AtomMapSpec('v6', [('Fum', 'abcd')], [('OAC', 'abcd')]),
    AtomMapSpec('v8', [('Asp', 'abcd')], [('OAC', 'abcd')]),
)


@dataclass(frozen=True)
class TracerSpec(BaseSpec):
    """Labelling of a substrate, the fraction of every isotopomer as a bit string with atom 1 first.

    Args:
        metabolite:
            The metabolite.

        fractions:
            The fraction of every isotopomer, such as `{'00': 0.5, '01': 0.25, '11': 0.25}`, where `1` is a labelled
            atom. The fractions sum to 1.

    Raises:
        ValidationError:
            The fractions do not sum to 1, or their bit strings differ in length, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import TracerSpec
        >>> TracerSpec('AcCoA', {'00': 0.5, '11': 0.5}).size
        2
    """

    metabolite: str
    fractions: Mapping[str, float]

    def _check(self) -> None:
        """Raise an error for fractions that do not sum to 1, or isotopomers of different lengths.

        Raises:
            ValidationError:
                The fractions are not a distribution over isotopomers of one length.
        """
        if abs(sum(self.fractions.values()) - 1.0) > SUM_TOLERANCE:
            error_msg = f'The isotopomer fractions of {self.metabolite} sum to {sum(self.fractions.values())}, not 1.'
            raise ValidationError(error_msg)
        if len({len(item) for item in self.fractions}) != 1:
            error_msg = f'The isotopomers of {self.metabolite} differ in length: {", ".join(self.fractions)}.'
            raise ValidationError(error_msg)

    @property
    def size(self) -> int:
        """Number of atoms of the metabolite."""
        return len(next(iter(self.fractions)))


ASPARTATE = TracerSpec('Asp', {'0000': 1.0})

PUBLISHED_TRACERS = (TracerSpec('AcCoA', {'00': 0.5, '01': 0.25, '11': 0.25}), ASPARTATE)

PARALLEL_TRACERS = {
    'one': (TracerSpec('AcCoA', {'00': 0.5, '10': 0.5}), ASPARTATE),
    'two': (TracerSpec('AcCoA', {'00': 0.5, '11': 0.5}), ASPARTATE),
}


@dataclass(frozen=True)
class FragmentSpec(BaseSpec):
    """Fragment of a metabolite whose mass isotopomer distribution a simulation gives or an experiment measures.

    Args:
        metabolite:
            The metabolite.

        atoms:
            The atoms of the fragment, counted from 1.

        name:
            The name of the fragment, or the empty string for the name of the metabolite.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import FragmentSpec
        >>> FragmentSpec('Glu', (1, 2, 3, 4, 5)).label
        'Glu'
    """

    metabolite: str
    atoms: Sequence[int]
    name: str = ''

    def _check(self) -> None:
        """Accept the spec, which has nothing to check."""

    @property
    def label(self) -> str:
        """Name of the fragment, the name of its metabolite without one of its own."""
        return self.name or self.metabolite


GLUTAMATE = FragmentSpec('Glu', (1, 2, 3, 4, 5))


@dataclass(frozen=True)
class LabelMeasurementSpec(BaseSpec):
    """Measured mass isotopomer distribution of a fragment in one tracer experiment.

    Args:
        experiment:
            The name of the tracer experiment.

        fragment:
            The name of the fragment.

        values:
            The measured distribution.

        deviation:
            The standard deviation of every value.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import LabelMeasurementSpec
        >>> LabelMeasurementSpec('a', 'Glu', (0.5, 0.5), 0.002).deviation
        0.002
    """

    experiment: str
    fragment: str
    values: npt.ArrayLike = field(repr=False)
    deviation: float = 0.002

    def _check(self) -> None:
        """Raise an error for a deviation that is not positive.

        Raises:
            ValidationError:
                The deviation is not positive.
        """
        if not self.deviation > 0:
            error_msg = f'The deviation of a measurement of {self.fragment} must be positive, not {self.deviation}.'
            raise ValidationError(error_msg)

    @property
    def distribution(self) -> npt.NDArray[np.float64]:
        """Measured distribution as an array."""
        return np.asarray(self.values, dtype=float)


@dataclass(frozen=True)
class _TermSpec(BaseSpec):
    """Inflow to an elementary metabolite unit: its directed flux, its weight, and the units it combines."""

    flux: int
    weight: float
    substrates: tuple[_Unit, ...]


@dataclass(frozen=True)
class _LabellingSpec(BaseSpec):
    """Units of a labelling problem: the network, the settings that read it, and the units of its fragments."""

    network: MetabolicNetworkElement
    reactions: list[str]
    tracers: Mapping[str, Sequence[TracerSpec]]
    fragments: Sequence[FragmentSpec]
    terms: dict[_Unit, list[_TermSpec]]
    leaves: set[_Unit]
    sizes: list[int]


@dataclass(frozen=True)
class _FitSpec(BaseSpec):
    """Parametrization of the net fluxes over a null space basis of the balances, with one exchange per reversible one.

    It also holds the data the residuals read.
    """

    units: _LabellingSpec
    measurements: Sequence[LabelMeasurementSpec]
    measured_fluxes: Mapping[str, tuple[float, float]]
    particular: Array
    basis: Array
    exchanges: list[str]
    lower: Array
    upper: Array
    exchange_limit: Array


@dataclass(frozen=True)
class LabellingResult:
    """Isotope labelling of a metabolic network: simulated distributions, or estimated fluxes with their test.

    Args:
        network:
            The metabolic network.

        distributions:
            The distribution of every fragment of every experiment.

        fluxes:
            The estimated net flux of every reaction, and the exchange `<reaction>.exchange` of a reversible one, for
            `'estimation'`, or `None`.

        intervals:
            The 95% interval of every estimated flux, for `'estimation'`, or `None`.

        identifiable:
            Whether the data identify every estimated flux, false for an interval that reaches a bound, for
            `'estimation'`, or `None`.

        residual:
            The sum of the squares of the weighted residuals at the optimum, for `'estimation'`, or `None`.

        chi2:
            The range of the sum of squares that the test accepts at the 5% level, for `'estimation'`, or `None`.

        accepted:
            Whether the residual lies in that range, for `'estimation'`, or `None`.
    """

    network: MetabolicNetworkElement
    distributions: dict[str, dict[str, Array]]
    fluxes: dict[str, float] | None = None
    intervals: dict[str, tuple[float, float]] | None = None
    identifiable: dict[str, bool] | None = None
    residual: float | None = None
    chi2: tuple[float, float] | None = None
    accepted: bool | None = None


def labelling(
    source: Source,
    *,
    kind: str = 'simulation',
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
) -> LabellingResult:
    """Return the isotope labelling of a metabolic network at metabolic and isotopic steady state, of one kind.

    The function takes a system that holds a metabolic network. It walks the atom maps backward from the fragments to
    collect the elementary metabolite units (Antoniewicz et al. 2007), and solves per unit size one sparse linear system
    with `scipy.sparse.linalg.splu`.

    - `'simulation'`: the mass isotopomer distributions of the fragments for given fluxes.
    - `'estimation'`: the fluxes by weighted least squares over measured distributions and measured fluxes, from seeded
      feasible starts that the core solve draws. The net fluxes run over a null space basis of the balances, with one
      exchange flux per reversible reaction, so every balance holds exactly. A 95% interval of every flux and a
      chi-square test of the fit follow.

    Args:
        source:
            The system that holds the metabolic network.

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

    Returns:
        result:
            The distributions, and for `'estimation'` the fluxes, their intervals, and the test of the fit.

    Raises:
        AnalysisError:
            A setting is unknown, the source is not a system with a metabolic network, or a metabolite has no source of
            labels.

        ValidationError:
            An atom map is not valid, or two maps give one metabolite different numbers of atoms.

    Examples:
        >>> from skcomplex.core.components.spaces import StaticTime
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.metabolism import ReactionSpec, metabolic_network_type
        >>> from skcomplex.systems.biology.systems_biology import (
        ...     AtomMapSpec,
        ...     FragmentSpec,
        ...     TracerSpec,
        ...     labelling,
        ... )
        >>> reactions = {'in': ReactionSpec({'B': 1.0}), 'out': ReactionSpec({'B': -1.0})}
        >>> network = metabolic_network_type('ChainElement', reactions)('net')
        >>> system = System(elements=[network], time=StaticTime())
        >>> result = labelling(
        ...     system,
        ...     atoms=[AtomMapSpec('in', [('A', 'ab')], [('B', 'ba')])],
        ...     tracers={'first': [TracerSpec('A', {'00': 0.5, '01': 0.5})]},
        ...     fragments=[FragmentSpec('B', (1, 2))],
        ...     fluxes={'in': 1.0, 'out': 1.0},
        ... )
        >>> result.distributions['first']['B'].tolist()
        [0.5, 0.5, 0.0]
    """
    _check_choice('kind', kind, KINDS)
    _check_choice('intervals', intervals, INTERVALS)
    atom_counts = _atom_counts(atoms)
    if not isinstance(source, System):
        error_msg = f'labelling reads a system, not a {type(source).__name__}.'
        raise AnalysisError(error_msg)
    units = _collect_units(
        _network_of(source),
        atoms,
        tracers=tracers or {},
        fragments=fragments,
        symmetric=symmetric,
        atom_counts=atom_counts,
    )
    if kind == 'simulation':
        flows = _directed_given(units, fluxes)
        distributions = {experiment: _simulated(units, flows, found) for experiment, found in units.tracers.items()}
        return LabellingResult(units.network, distributions)
    fit = _parametrization(units, measurements, measured_fluxes or {})
    rng = np.random.default_rng(random_state)
    fits = [_fit(fit, _start(fit, rng)) for _ in range(int(starts))]
    parameters, best = min(fits, key=lambda item: item[1])
    return _estimates(fit, parameters, best, intervals)


def _check_choice(name: str, value: object, choices: tuple[object, ...]) -> None:
    """Raise an error for a setting of the labelling that is not one of its choices."""
    if value not in choices:
        error_msg = f'labelling has {name}={value!r}. Use one of {", ".join(map(str, choices))}.'
        raise AnalysisError(error_msg)


def _atom_counts(atoms: Sequence[AtomMapSpec]) -> dict[str, int]:
    """Return the number of atoms of every metabolite, after checking that every atom map agrees on it.

    Args:
        atoms:
            The atom maps.

    Returns:
        counts:
            The number of atoms by metabolite.

    Raises:
        ValidationError:
            An atom map is not valid, or two reactions give one metabolite different numbers of atoms.
    """
    counts: dict[str, tuple[int, str]] = {}
    for item in atoms:
        item._validated()
        for metabolite, letters in (*item.reactants, *item.products):
            known = counts.setdefault(metabolite, (len(letters), item.reaction))
            if known[0] != len(letters):
                error_msg = (
                    f'The reaction {known[1]} gives {metabolite} {known[0]} atoms, but the reaction '
                    f'{item.reaction} gives it {len(letters)}.'
                )
                raise ValidationError(error_msg)
    return {metabolite: count for metabolite, (count, _) in counts.items()}


def _network_of(system: System) -> MetabolicNetworkElement:
    """Return the metabolic network of a system."""
    for element in system.elements:
        if isinstance(element, MetabolicNetworkElement):
            return element
    error_msg = 'labelling reads a system that holds a metabolic network element.'
    raise AnalysisError(error_msg)


def _bounds_of(network: MetabolicNetworkElement, reaction: str) -> tuple[float, float]:
    """Return the bounds of a reaction, from the parameters of the network."""
    identifier = network._identifier(reaction)
    return network.get_number(f'lower_{identifier}'), network.get_number(f'upper_{identifier}')


def _reversible(network: MetabolicNetworkElement, reaction: str) -> bool:
    """Check whether a reaction may run backward."""
    return _bounds_of(network, reaction)[0] < 0


def _directed_maps(network: MetabolicNetworkElement, atoms: Sequence[AtomMapSpec]) -> list[tuple[int, AtomMapSpec]]:
    """Return the maps in both directions, each with the position of its directed flux."""
    reactions = list(network.reactions)
    count = len(reactions)
    directed = []
    for item in atoms:
        if item.reaction not in reactions:
            error_msg = f'An atom map names the reaction {item.reaction}, which the network does not hold.'
            raise AnalysisError(error_msg)
        position = reactions.index(item.reaction)
        directed.append((position, item))
        if _reversible(network, item.reaction):
            reverse = AtomMapSpec(item.reaction, item.products, item.reactants)
            directed.append((position + count, reverse))
    return directed


def _terms_of(
    unit: _Unit,
    maps: Sequence[tuple[int, AtomMapSpec]],
    symmetric: Sequence[str],
    atom_counts: Mapping[str, int],
) -> list[_TermSpec]:
    """Return the inflows to a unit from every reaction that produces its metabolite."""
    metabolite, atoms = unit
    size = atom_counts.get(metabolite, max(atoms))
    scrambled = tuple(sorted(size + 1 - atom for atom in atoms))
    choices = [(atoms, 0.5), (scrambled, 0.5)] if metabolite in symmetric else [(atoms, 1.0)]
    terms = []
    for flux, item in maps:
        places = {
            letter: (name, position + 1) for name, letters in item.reactants for position, letter in enumerate(letters)
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


def _collect_units(
    network: MetabolicNetworkElement,
    atoms: Sequence[AtomMapSpec],
    *,
    tracers: Mapping[str, Sequence[TracerSpec]],
    fragments: Sequence[FragmentSpec],
    symmetric: Sequence[str],
    atom_counts: Mapping[str, int],
) -> _LabellingSpec:
    """Walk the atom maps backward from the fragments, and return the terms and the sizes of the units."""
    maps = _directed_maps(network, atoms)
    sources = {item.metabolite for found in tracers.values() for item in found}
    terms: dict[_Unit, list[_TermSpec]] = {}
    leaves: set[_Unit] = set()
    stack: list[_Unit] = [(item.metabolite, tuple(sorted(item.atoms))) for item in fragments]
    while stack:
        unit = stack.pop()
        if unit in terms or unit in leaves:
            continue
        if unit[0] in sources:
            leaves.add(unit)
            continue
        found = _terms_of(unit, maps, symmetric, atom_counts)
        if not found:
            error_msg = (
                f'No reaction produces {unit[0]} and no tracer labels it. Give it a tracer, even an unlabelled one.'
            )
            raise AnalysisError(error_msg)
        terms[unit] = found
        stack += [substrate for term in found for substrate in term.substrates]
    sizes = sorted({len(unit[1]) for unit in terms})
    return _LabellingSpec(network, list(network.reactions), tracers, fragments, terms, leaves, sizes)


def _source_distribution(tracer: TracerSpec, atoms: Sequence[int]) -> Array:
    """Return the distribution of the labelled atoms of a tracer among some atoms."""
    distribution = np.zeros(len(atoms) + 1)
    for isotopomer, fraction in tracer.fractions.items():
        distribution[sum(isotopomer[atom - 1] == '1' for atom in atoms)] += fraction
    return distribution


def _source_units(units: _LabellingSpec, tracers: Sequence[TracerSpec]) -> dict[_Unit, Array]:
    """Return the distribution of every leaf unit in one experiment."""
    found = {item.metabolite: item for item in tracers}
    known = {}
    for unit in units.leaves:
        tracer = found.get(unit[0])
        unlabelled = np.zeros(len(unit[1]) + 1)
        unlabelled[0] = 1.0
        known[unit] = unlabelled if tracer is None else _source_distribution(tracer, unit[1])
    return known


def _solve_size(units: _LabellingSpec, size: int, flows: Array, known: dict[_Unit, Array]) -> None:
    """Solve the units of one size, `A X = B`, and add their distributions to the known ones."""
    chosen = [unit for unit in units.terms if len(unit[1]) == size]
    positions = {unit: position for position, unit in enumerate(chosen)}
    rows, columns, values = [], [], []
    right = np.zeros((len(chosen), size + 1))
    totals = np.zeros(len(chosen))
    for unit, position in positions.items():
        for term in units.terms[unit]:
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
    matrix = sparse.coo_array((values, (rows, columns)), shape=(len(chosen), len(chosen))).tocsc()
    matrix = (matrix + sparse.diags_array(totals, format='csc')).tocsc()
    solution = splu(matrix).solve(right)
    known.update({unit: solution[position] for unit, position in positions.items()})


def _simulated(units: _LabellingSpec, flows: Array, tracers: Sequence[TracerSpec]) -> dict[str, Array]:
    """Return the distribution of every fragment at directed fluxes in one experiment."""
    known = _source_units(units, tracers)
    for size in units.sizes:
        _solve_size(units, size, flows, known)
    return {item.label: known[(item.metabolite, tuple(sorted(item.atoms)))] for item in units.fragments}


def _directed_given(units: _LabellingSpec, fluxes: Mapping[str, float | tuple[float, float]] | None) -> Array:
    """Return the directed fluxes that the `fluxes` setting gives."""
    count = len(units.reactions)
    flows = np.zeros(2 * count)
    for reaction, value in (fluxes or {}).items():
        position = units.reactions.index(reaction)
        forward, backward = value if isinstance(value, tuple) else (value, 0.0)
        flows[position], flows[position + count] = forward, backward
    return flows


def _balances(units: _LabellingSpec) -> Array:
    """Return the matrix of the balances of the network over its net fluxes."""
    network = units.network
    metabolites = list(dict.fromkeys(name for spec in network.reactions.values() for name in spec.coefficients))
    matrix = np.zeros((len(metabolites), len(units.reactions)))
    for column, spec in enumerate(network.reactions.values()):
        for name, value in spec.coefficients.items():
            matrix[metabolites.index(name), column] = value
    return matrix


def _parametrization(
    units: _LabellingSpec,
    measurements: Sequence[LabelMeasurementSpec],
    measured_fluxes: Mapping[str, tuple[float, float]],
) -> _FitSpec:
    """Return a particular solution, the null space basis of the net fluxes, and the exchanges."""
    balances = _balances(units)
    reactions = units.reactions
    network = units.network
    fixed = {name: value for name, (value, deviation) in measured_fluxes.items() if deviation == 0}
    rows = np.zeros((len(fixed), len(reactions)))
    for row, name in enumerate(fixed):
        rows[row, reactions.index(name)] = 1.0
    system = np.vstack([balances, rows])
    right = np.concatenate([np.zeros(balances.shape[0]), np.fromiter(fixed.values(), dtype=float)])
    exchanges = [name for name in reactions if _reversible(network, name)]
    return _FitSpec(
        units,
        measurements,
        measured_fluxes,
        linalg.lstsq(system, right)[0],
        linalg.null_space(system),
        exchanges,
        np.asarray([_bounds_of(network, name)[0] for name in reactions], dtype=float),
        np.asarray([_bounds_of(network, name)[1] for name in reactions], dtype=float),
        np.asarray([_bounds_of(network, name)[1] for name in exchanges], dtype=float),
    )


def _net_of(fit: _FitSpec, parameters: Array) -> Array:
    """Return the net fluxes of parameters."""
    return fit.particular + fit.basis @ parameters[: fit.basis.shape[1]]


def _flows_of(fit: _FitSpec, parameters: Array) -> Array:
    """Return the directed fluxes of parameters, the exchanges added in both directions."""
    net = _net_of(fit, parameters)
    reactions = fit.units.reactions
    count = len(reactions)
    flows = np.concatenate([np.maximum(net, 0.0), np.maximum(-net, 0.0)])
    exchanges = parameters[fit.basis.shape[1] :]
    for name, exchange in zip(fit.exchanges, exchanges, strict=True):
        position = reactions.index(name)
        flows[position] += exchange
        flows[position + count] += exchange
    return flows


def _residuals(fit: _FitSpec, parameters: Array) -> Array:
    """Return the weighted residuals of the measurements and the measured fluxes, and the penalties of bounds."""
    flows = _flows_of(fit, parameters)
    units = fit.units
    simulated = {experiment: _simulated(units, flows, tracers) for experiment, tracers in units.tracers.items()}
    terms = [
        (simulated[item.experiment][item.fragment] - item.distribution) / item.deviation for item in fit.measurements
    ]
    net = _net_of(fit, parameters)
    measured = [
        (net[units.reactions.index(name)] - value) / deviation
        for name, (value, deviation) in fit.measured_fluxes.items()
        if deviation > 0
    ]
    penalties = PENALTY * np.concatenate([np.minimum(net - fit.lower, 0.0), np.maximum(net - fit.upper, 0.0)])
    return np.concatenate([*terms, np.asarray(measured, dtype=float), penalties])


def _drawn_vertices(fit: _FitSpec, rng: np.random.Generator) -> Array:
    """Return a feasible net flux, a random combination of vertices that the core solve finds."""
    network = fit.units.network
    reactions = fit.units.reactions
    fixed = {name: value for name, (value, deviation) in fit.measured_fluxes.items() if deviation == 0}
    identifiers = {
        **{f'lower_{network._identifier(name)}': value for name, value in fixed.items()},
        **{f'upper_{network._identifier(name)}': value for name, value in fixed.items()},
    }
    vertices = []
    for _ in range(VERTEX_DRAWS):
        weights = {name: float(value) for name, value in zip(reactions, rng.normal(size=len(reactions)), strict=True)}
        element = type(network)(
            network.label,
            objective=weights,
            reactions=network.reactions,
            default_objective=network.default_objective,
        ).set_params(**identifiers)
        problem = PassCompiler().prepare(System(elements=[element], time=StaticTime())).run().problem_
        if not isinstance(problem, ProblemSpec):
            error_msg = f'The network compiles to a {type(problem).__name__}, not a linear program.'
            raise AnalysisError(error_msg)
        state = RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)
        solved = problem.programs[0].run(state).values_
        vertices.append(
            [
                float(np.real(np.ravel(problem.states.read(solved, f'{element.label}.{element.flux_name(name)}'))[0]))
                for name in reactions
            ],
        )
    shares = rng.dirichlet(np.ones(VERTEX_DRAWS))
    return shares @ np.asarray(vertices, dtype=float)


def _start(fit: _FitSpec, rng: np.random.Generator) -> Array:
    """Return a feasible start: the coordinates of a drawn flux and random exchanges."""
    net = _drawn_vertices(fit, rng)
    coordinates = fit.basis.T @ (net - fit.particular)
    return np.concatenate([coordinates, rng.uniform(0.0, fit.exchange_limit / 2.0)])


def _fit(fit: _FitSpec, start: Array, extra: tuple[int, float] | None = None) -> tuple[Array, float]:
    """Return the parameters that minimise the residuals from a start, and the sum of squares of the residuals."""
    count = fit.basis.shape[1]
    lower = np.concatenate([np.full(count, -np.inf), np.zeros(len(fit.exchanges))])
    upper = np.concatenate([np.full(count, np.inf), fit.exchange_limit])

    def residuals(parameters: Array) -> Array:
        found = _residuals(fit, parameters)
        if extra is None:
            return found
        target = _estimate_of(fit, parameters)[extra[0]]
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
    return result.x, float(np.sum(_residuals(fit, result.x) ** 2))


def _estimate_of(fit: _FitSpec, parameters: Array) -> Array:
    """Return the net fluxes and the exchanges of parameters."""
    return np.concatenate([_net_of(fit, parameters), parameters[fit.basis.shape[1] :]])


def _estimate_names(fit: _FitSpec) -> list[str]:
    """Return the names of the net fluxes and the exchanges."""
    return [*fit.units.reactions, *(f'{name}.exchange' for name in fit.exchanges)]


def _bounds_of_estimate(fit: _FitSpec) -> tuple[Array, Array]:
    """Return the bounds of the net fluxes and the exchanges."""
    count = len(fit.exchanges)
    return np.concatenate([fit.lower, np.zeros(count)]), np.concatenate([fit.upper, fit.exchange_limit])


def _linear_intervals(fit: _FitSpec, parameters: Array) -> tuple[Array, Array]:
    """Return the intervals of the linearised fit."""
    steps = 1e-6 * np.maximum(1.0, np.abs(parameters))
    center = _residuals(fit, parameters)
    jacobian = np.zeros((center.size, parameters.size))
    for column, step in enumerate(steps):
        shifted = parameters.copy()
        shifted[column] += step
        jacobian[:, column] = (_residuals(fit, shifted) - center) / step
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    mapping = np.zeros((len(_estimate_names(fit)), parameters.size))
    count = fit.basis.shape[1]
    mapping[: fit.basis.shape[0], :count] = fit.basis
    mapping[fit.basis.shape[0] :, count:] = np.eye(parameters.size - count)
    deviations = np.sqrt(np.maximum(np.diag(mapping @ covariance @ mapping.T), 0.0))
    estimate = _estimate_of(fit, parameters)
    lower, upper = _bounds_of_estimate(fit)
    return (
        np.maximum(estimate - NORMAL_QUANTILE * deviations, lower),
        np.minimum(estimate + NORMAL_QUANTILE * deviations, upper),
    )


def _profile_side(fit: _FitSpec, parameters: Array, position: int, best: float, limit: float) -> float:
    """Return where the sum of squares rises by the threshold on the way from the optimum to a bound."""
    estimate = float(_estimate_of(fit, parameters)[position])
    step = max(PROFILE_RELATIVE_STEP * abs(estimate), 1.0)
    direction = 1.0 if limit > estimate else -1.0
    inner, inner_start = estimate, parameters
    outer = estimate
    while True:
        outer = min(outer + step, limit) if direction > 0 else max(outer - step, limit)
        fitted, value = _fit(fit, inner_start, (position, outer))
        if value - best >= INTERVAL_THRESHOLD:
            break
        inner, inner_start = outer, fitted
        if outer == limit:
            return limit
        step *= 2.0
    for _ in range(PROFILE_ITERATIONS):
        middle = 0.5 * (inner + outer)
        fitted, value = _fit(fit, inner_start, (position, middle))
        if value - best >= INTERVAL_THRESHOLD:
            outer = middle
        else:
            inner, inner_start = middle, fitted
        if abs(outer - inner) <= PROFILE_WIDTH * max(1.0, abs(inner)):
            break
    return inner


def _profile_intervals(fit: _FitSpec, parameters: Array, best: float) -> tuple[Array, Array]:
    """Return the intervals where the sum of squares stays within the threshold of its optimum."""
    lower, upper = _bounds_of_estimate(fit)
    low = np.empty_like(lower)
    high = np.empty_like(upper)
    for position in range(low.size):
        low[position] = _profile_side(fit, parameters, position, best, float(lower[position]))
        high[position] = _profile_side(fit, parameters, position, best, float(upper[position]))
    return low, high


def _estimates(fit: _FitSpec, parameters: Array, best: float, intervals: str) -> LabellingResult:
    """Return the estimates, their intervals, the test of the fit, and the distributions at the optimum."""
    names = _estimate_names(fit)
    low, high = (
        _profile_intervals(fit, parameters, best) if intervals == 'profile' else _linear_intervals(fit, parameters)
    )
    estimate = _estimate_of(fit, parameters)
    lower, upper = _bounds_of_estimate(fit)
    identifiable = {
        name: bool(a > bottom + BOUND_TOLERANCE and b < top - BOUND_TOLERANCE) or bottom == top
        for name, a, b, bottom, top in zip(names, low, high, lower, upper, strict=True)
    }
    size = sum(item.distribution.size for item in fit.measurements)
    size += sum(1 for _, deviation in fit.measured_fluxes.values() if deviation > 0)
    freedom = max(size - parameters.size, 1)
    accepted_range = (float(chi2.ppf(0.025, freedom)), float(chi2.ppf(0.975, freedom)))
    flows = _flows_of(fit, parameters)
    units = fit.units
    return LabellingResult(
        units.network,
        {experiment: _simulated(units, flows, tracers) for experiment, tracers in units.tracers.items()},
        fluxes=dict(zip(names, estimate.tolist(), strict=True)),
        intervals={name: (float(a), float(b)) for name, a, b in zip(names, low, high, strict=True)},
        identifiable=identifiable,
        residual=best,
        chi2=accepted_range,
        accepted=bool(accepted_range[0] <= best <= accepted_range[1]),
    )


def tca_labelling_analysis(
    kind: str = 'estimation',
    *,
    noise: float = DEVIATION,
    random_state: int = 0,
    starts: int = 5,
    intervals: str = 'linear',
) -> partial[LabellingResult]:
    """Return the labelling of the TCA example bound to its settings, with data simulated from its known fluxes.

    The data of the estimation are simulated from the known fluxes plus seeded noise.

    Args:
        kind:
            `'simulation'` for the published tracer at the true fluxes, or `'estimation'` for a fit to two parallel
            tracer experiments, 50% [1-13C] and 50% [1,2-13C] acetyl-CoA, with `v1` fixed at 100.

        noise:
            The standard deviation of the noise of the data.

        random_state:
            The seed of the noise and of the starts.

        starts:
            The number of feasible starts of the fit.

        intervals:
            `'linear'` or `'profile'`.

    Returns:
        analysis:
            The function `labelling` bound to the settings, which takes the system.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import carbon_13_mfa, tca_labelling_analysis
        >>> result = tca_labelling_analysis('simulation')(carbon_13_mfa())
        >>> [round(float(item), 4) for item in result.distributions['published']['Glu']]
        [0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039]
    """
    if kind == 'simulation':
        return bind_analysis(
            labelling,
            kind='simulation',
            atoms=TCA_ATOMS,
            tracers={'published': PUBLISHED_TRACERS},
            fragments=(GLUTAMATE,),
            fluxes=TCA_TRUE_FLUXES,
            symmetric=TCA_SYMMETRIC,
        )
    simulated = labelling(
        carbon_13_mfa(),
        kind='simulation',
        atoms=TCA_ATOMS,
        tracers=PARALLEL_TRACERS,
        fragments=(GLUTAMATE,),
        fluxes=TCA_TRUE_FLUXES,
        symmetric=TCA_SYMMETRIC,
    ).distributions
    rng = np.random.default_rng(random_state)
    measurements = []
    for experiment, values in simulated.items():
        clean = values[GLUTAMATE.label]
        noisy = clean + rng.normal(0.0, noise, clean.size) if noise else clean
        measurements.append(LabelMeasurementSpec(experiment, GLUTAMATE.label, noisy, DEVIATION))
    return bind_analysis(
        labelling,
        kind='estimation',
        atoms=TCA_ATOMS,
        tracers=PARALLEL_TRACERS,
        fragments=(GLUTAMATE,),
        measurements=measurements,
        measured_fluxes={'v1': (100.0, 0.0)},
        symmetric=TCA_SYMMETRIC,
        starts=starts,
        intervals=intervals,
        random_state=random_state,
    )


@register_system(
    field='biology/systems_biology',
    tags=('carbon-13 flux analysis', 'isotope labelling', 'elementary metabolite units', 'metabolism', 'static'),
    references=('Raman 2021', 'Antoniewicz et al. 2007', 'Antoniewicz et al. 2006'),
    defaults={'until': 0.0},
)
def carbon_13_mfa(network: MetabolicNetworkElement | None = None) -> System:
    """Return a metabolic network that `labelling` fits to labelling data, the TCA example by default.

    Labelled atoms of a tracer spread through the network along its atom transitions, so the measured mass isotopomer
    distributions of its products carry the fluxes (Raman 2021, Section 9.7). `tca_labelling_analysis` gives the
    labelling of the TCA example of Antoniewicz et al. (2007) with data simulated from its known fluxes.

    Args:
        network:
            The metabolic network, or `None` for the TCA example.

    Returns:
        system:
            The system.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import carbon_13_mfa
        >>> carbon_13_mfa().elements[0].label
        'tca'
    """
    return System(elements=[TcaNetworkElement('tca') if network is None else network], time=StaticTime())
