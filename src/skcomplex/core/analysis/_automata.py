"""Decide and measure cellular automata, subshifts, and finite networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import itertools
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import QQ, SR, Expression, LaurentPolynomialRing, matrix
from sage.rings.polynomial.laurent_polynomial import LaurentPolynomial
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.structure.element import Matrix
from sklearn.base import clone

from ..components import SubshiftSpec
from ..components._parameter import _Parameter
from ..components.interactions import BaseInteraction, Interaction
from ..components.laws import (
    BaseLaw,
    CallSpec,
    LinearRuleSpec,
    OuterTotalisticSpec,
    TableSpec,
    TotalisticSpec,
    UpdateLaw,
)
from ..components.laws._local_rule import _LocalRuleLaw
from ..components.spaces import BaseNeighborhood, BaseSpace, LatticeNeighborhood, LatticeSpace
from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems._composite_system import _CompositeSystem
from ..engine.problems._declared_element import _DeclaredElement
from ..engine.problems._group_element import _GroupElement
from ..enums import DataKind
from ..errors import AnalysisError
from ..systems import System
from ..types import Indices, IntArray
from ._analysis_tags import AnalysisTags
from ._base import _check_choice, _check_settings, _composite, _get_system
from ._utils import load_module
from .types import DataSource

MAX_CONFIGURATIONS = 2**16
MAX_PATTERNS = 2**20
MAX_STATE_SPACE = 2**24
CHUNK = 2**16
MATRIX_AXES = 2
INTEGER_KINDS = 'biu'
SAT_SOLVER = 'cadical153'
METHODS = ('auto', 'enumeration', 'sat')
SYNCHRONOUS = 'synchronous'
TAGS = AnalysisTags(DataKind.SYSTEM)
KINDS = (
    'injective',
    'surjective',
    'pre-injective',
    'garden-of-eden',
    'inverse',
    'compose',
    'difference',
    'entropy',
    'matrix',
    'polynomial',
    'admissible',
    'violations',
    'count',
    'sample',
    'attractors',
)
SETTINGS = {
    'injective': {'method', 'infinite'},
    'surjective': {'method', 'infinite'},
    'pre-injective': {'method', 'window', 'infinite'},
    'garden-of-eden': {'window', 'tries', 'random_state'},
    'inverse': {'max_radius'},
    'compose': {'second'},
    'difference': {'configuration', 'site', 'steps'},
    'entropy': {'windows', 'base'},
    'matrix': set(),
    'polynomial': set(),
    'admissible': {'configuration'},
    'violations': {'configuration'},
    'count': set(),
    'sample': {'sweeps', 'configuration', 'random_state'},
    'attractors': set(),
}
AUTOMATON = 'an automaton'
LINEAR = 'a linear automaton'
SUBSHIFT = 'a subshift'
NETWORK = 'a finite network'
SOURCES = {
    'injective': (AUTOMATON, LINEAR),
    'surjective': (AUTOMATON, LINEAR),
    'pre-injective': (AUTOMATON, LINEAR),
    'garden-of-eden': (AUTOMATON,),
    'inverse': (AUTOMATON,),
    'compose': (AUTOMATON,),
    'difference': (AUTOMATON, NETWORK),
    'entropy': (SUBSHIFT, AUTOMATON),
    'matrix': (LINEAR,),
    'polynomial': (LINEAR,),
    'admissible': (SUBSHIFT,),
    'violations': (SUBSHIFT,),
    'count': (SUBSHIFT,),
    'sample': (SUBSHIFT,),
    'attractors': (AUTOMATON, NETWORK),
}


class _Clauses:
    """Clauses of a SAT problem over one-hot variables of site values."""

    def __init__(self) -> None:
        self.clauses: list[list[int]] = []
        self.count = 0

    def get_one_hot(self, sites: int, states: int, *, exactly: bool = True) -> npt.NDArray[np.int64]:
        """Return one variable per site and value.

        Exactly one, or at most one, is true per site.
        """
        variables = np.arange(self.count + 1, self.count + 1 + sites * states, dtype=np.int64).reshape(sites, states)
        self.count += sites * states
        for row in variables.tolist():
            if exactly:
                self.clauses.append(row)
            self.clauses += [[-first, -second] for first, second in itertools.combinations(row, 2)]
        return variables

    def get_fresh(self) -> int:
        """Return a new variable."""
        self.count += 1
        return self.count

    def solve(self) -> list[int] | None:
        """Return a system of the clauses, or `None` when they are unsatisfiable."""
        solvers = load_module('pysat.solvers')
        with solvers.Solver(name=SAT_SOLVER, bootstrap_with=self.clauses) as solver:
            return solver.get_model() if solver.solve() else None


@dataclass(frozen=True)
class AutomatonResult:
    """A result of `automaton`, a decision or a measure of a cellular automaton.

    It holds the answer the kind asks for, such as whether no two grids share their next grid, a grid that no grid leads
    to, an automaton that undoes another, the number of allowed grids, or the cycle every configuration falls on, and
    leaves the other fields `None`. It answers whether a model of traffic, of growth, or of a crystal loses information,
    and which cell types a gene network settles into. A configuration has the code that reads its values as the digits
    of a number in the base of the number of values, the first value most significant, in the order of the cells, or of
    the states of the elements of a network and the members of a group.

    Read more in the user guide, under Analyse.

    Args:
        automaton:
            The `System` of the automaton or of the subshift, the one a ready-made system builds for it.

        injective:
            Whether the automaton is injective.

        surjective:
            Whether the automaton is surjective.

        pre_injective:
            Whether no two configurations that differ only inside the window share their image.

        pattern:
            A configuration whose values on the window have no preimage, or `None` when none is found.

        inverse:
            The `System` of the inverse automaton, whose interaction sets an update law with a table rule and its
            neighbourhood, or `None` when the automaton is not injective or no memory up to the radius determines the
            cells.

        composition:
            The `System` of the automaton of the composition, whose interaction sets an update law with a function rule
            and its neighbourhood.

        differences:
            One row per step, the initial one first, true where the runs differ.

        distances:
            The Hamming distance between the runs at every step, the initial one first.

        entropy:
            One entropy per window.

        matrix:
            The Sage matrix of the linear automaton.

        polynomial:
            The Sage Laurent polynomial in the variables `x0` to `x(d-1)`.

        admissible:
            Whether the configuration belongs to the subshift.

        violations:
            One flag per site, true where a forbidden pattern appears.

        count:
            The number of configurations without a forbidden pattern.

        sample:
            The admissible configuration.

        transients:
            The number of steps every configuration takes to reach its cycle, indexed by its code.

        attractors:
            The index of the cycle every configuration falls on, indexed by its code.

        cycles:
            The configurations of every cycle, one row per step around it from the one with the smallest code, ordered
            by that code.

        lengths:
            The length of every cycle.

        basins:
            The number of configurations that fall on every cycle, the cycle included.

    See Also:
        automaton: Returns this result.

    Examples:
        >>> from skcomplex.core.analysis import automaton
        >>> from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem
        >>> automaton(ElementaryAutomatonSystem(rule=170, size=8)).injective
        True
    """

    automaton: System
    injective: bool | None = None
    surjective: bool | None = None
    pre_injective: bool | None = None
    pattern: IntArray | None = None
    inverse: System | None = None
    composition: System | None = None
    differences: npt.NDArray[np.bool_] | None = None
    entropy: npt.NDArray[np.float64] | None = None
    matrix: Matrix | None = None
    polynomial: LaurentPolynomial | None = None
    admissible: bool | None = None
    violations: npt.NDArray[np.bool_] | None = None
    count: int | None = None
    sample: IntArray | None = None
    transients: IntArray | None = None
    attractors: IntArray | None = None
    cycles: tuple[IntArray, ...] | None = None
    lengths: IntArray | None = None
    basins: IntArray | None = None
    distances: IntArray | None = None


@dataclass(frozen=True)
class _LocalRule:
    """Local rule read from an interaction.

    It holds the label of the interaction, the path of the group it binds, the state it sets, the states it reads,
    and its law.
    """

    label: str
    target: str
    state: str | tuple[str, ...]
    reads: tuple[str, ...] | None
    law: _LocalRuleLaw


@dataclass(frozen=True)
class _Automaton:
    """Automaton read from a system.

    It holds its rule, law, state, space, number of sites and of values, and local map.
    """

    system: _CompositeSystem
    rule: _LocalRule
    law: _LocalRuleLaw
    state: str
    space: BaseSpace
    size: int
    states: int | None
    local_map: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


@dataclass(frozen=True)
class _NetworkLaw:
    """Update law of a finite network read from a system.

    It holds the label of its interaction, the columns of the values it sets and reads, and the function that maps
    rows of the values it reads to rows of the values it sets.
    """

    label: str
    targets: IntArray
    reads: IntArray
    function: Callable[[IntArray], npt.ArrayLike]


@dataclass(frozen=True)
class _Network:
    """Finite network read from a system.

    It holds the system, the number of values a state takes, the number of values of a configuration, and the update
    laws.
    """

    system: _CompositeSystem
    states: int
    size: int
    laws: tuple[_NetworkLaw, ...]


@dataclass(frozen=True)
class _ComposedRule:
    """Rule of the composition of two automata, on the composite neighbourhood.

    Args:
        first:
            The local map of the automaton applied first.

        second:
            The local map of the automaton applied second.

        columns:
            For every cell the second rule reads, the cells of the composite neighbourhood the first rule reads there.

        states:
            The number of states of both automata.
    """

    first: IntArray
    second: IntArray
    columns: tuple[tuple[int, ...], ...]
    states: int


def automaton(
    source: DataSource,
    *,
    kind: str = 'injective',
    method: str = 'auto',
    window: Sequence[int] | None = None,
    infinite: bool = False,
    tries: int = 200,
    random_state: int | np.random.Generator | None = None,
    max_radius: int = 2,
    second: DataSource | None = None,
    configuration: npt.ArrayLike | None = None,
    site: int | Sequence[int] = 0,
    steps: int = 1,
    windows: Sequence[BaseSpace] | None = None,
    base: float = 2.0,
    sweeps: int = 100,
) -> AutomatonResult:
    """Return a decision or a measure of a cellular automaton or subshift of a system.

    An automaton is a group of cells on a grid with one rule that updates every cell at once from its neighbours, such
    as `ElementaryAutomatonSystem(rule=170, size=8)`, which shifts every cell left. A subshift is a group whose value
    lists forbidden patterns. A finite network is a system whose states take finitely many values and change by update
    laws, such as a Boolean network of genes. The function decides whether the rule loses information, finds a grid no
    grid leads to, builds the rule that undoes it, counts or samples the allowed grids, and maps every configuration to
    the cycle it falls on. Small grids are checked one by one and larger ones with a SAT solver. It tells whether a
    model of traffic, of crystal growth, or of a computer loses information, and reversible rules model physics that
    keeps energy.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The system of the automaton or of the subshift.

        kind:
            What to decide or measure. `'injective'`, `'surjective'`, and `'pre-injective'` ask whether no two grids
            share their next grid, whether every grid has a grid before it, and whether two grids that differ in a
            window never share their next grid. `'garden-of-eden'` finds a pattern no grid leads to. `'inverse'` finds
            the rule that undoes a reversible rule, trying radii up to `max_radius`. `'compose'` applies the rule and
            then `second`. `'difference'` shows where two runs of an automaton or a finite network differ when the
            values of `configuration` at `site` change, and how many values differ. `'entropy'` is `log |X_F| / |F|`
            of the allowed patterns of each window `F`. `'matrix'` and `'polynomial'` give a linear rule as a matrix or
            a Laurent polynomial. `'admissible'` and `'violations'` check `configuration` against the forbidden
            patterns. `'count'` counts the allowed grids of a subshift and `'sample'` draws one. `'attractors'` applies
            the synchronous map of an automaton or a deterministic finite network to all its configurations at once,
            up to `MAX_STATE_SPACE` of them, and gives the transient and the cycle of every configuration, and the
            length and the basin of every cycle. A finite network sets its values by update laws of expressions or by
            calls that take the values of a group with leading axes of configurations.

        method:
            `'enumeration'`, `'sat'`, or `'auto'` to enumerate up to `MAX_CONFIGURATIONS` configurations.

        window:
            The sites where configurations may differ, or of the Garden of Eden pattern, or `None` for every site.

        infinite:
            Whether to decide a linear automaton on `Z^d` from its Laurent polynomial instead of on its space.

        tries:
            The number of random patterns tried by SAT.

        random_state:
            The seed or generator of the random patterns or of the moves of a sample.

        max_radius:
            The largest radius of the memory of the inverse.

        second:
            The system of the automaton applied after the analysed one.

        configuration:
            The values of the sites: the initial ones of the difference pattern or of a sample, or the ones checked.

        site:
            The site of the changed cell, or the sites or values of a network that change.

        steps:
            The number of steps of the difference pattern.

        windows:
            The windows of the entropy, such as lattices with fixed edges of growing size.

        base:
            The base of the logarithm.

        sweeps:
            The number of sweeps of a sample, each proposing a random value at every site in a random order.

    Returns:
        result:
            The system and the decision or measure of the kind.

    Raises:
        AnalysisError:
            The system holds nothing the kind reads, the method is unknown, enumeration is asked for too many
            configurations, composed automata differ in space or number of states, the rule returns a value outside the
            states, a linear automaton has a matrix on a ball or a polynomial without offsets, the initial
            configuration of a sample is not admissible, a network is not finite or deterministic, or the attractors
            are asked for more than `MAX_STATE_SPACE` configurations.

    See Also:
        symbolic: Turns a series into symbols and measures their patterns.

    Examples:
        >>> from skcomplex.core.analysis import automaton
        >>> from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem
        >>> automaton(ElementaryAutomatonSystem(rule=30, size=8), kind='injective').injective
        False
    """
    system = _get_system(source)
    source = _get_declared(system)
    second_system = None if second is None else _get_system(second)
    second = None if second_system is None else _get_second(second_system)
    _check_choice('automaton', 'kind', kind, KINDS)
    _check_settings(
        automaton,
        'kind',
        SETTINGS[kind],
        kind=kind,
        method=method,
        window=window,
        infinite=infinite,
        tries=tries,
        random_state=random_state,
        max_radius=max_radius,
        second=second,
        configuration=configuration,
        site=site,
        steps=steps,
        windows=windows,
        base=base,
        sweeps=sweeps,
    )
    if method not in METHODS:
        error_msg = f'automaton has method={method!r}. Use one of {", ".join(METHODS)}.'
        raise AnalysisError(error_msg)
    needed = {
        'compose': ('second', second),
        'difference': ('configuration', configuration),
        'entropy': ('windows', windows),
        'admissible': ('configuration', configuration),
        'violations': ('configuration', configuration),
    }.get(kind)
    if needed is not None and needed[1] is None:
        error_msg = f'automaton of kind {kind!r} needs {needed[0]}. Give it.'
        raise AnalysisError(error_msg)
    sources = SOURCES[kind]
    error_msg = (
        f'automaton of kind {kind!r} reads a system with {" or ".join(sources)}, not {type(source).__name__}. Build '
        'one of them.'
    )
    if not isinstance(system, System) or not isinstance(source, _CompositeSystem):
        raise AnalysisError(error_msg)
    found = _find_subshift(source)
    rule = _find_rule(source)
    if rule is None and NETWORK in sources and found is None:
        network = _read_network(source)
        result = _run_network(network, kind, configuration=configuration, site=site, steps=steps)
        return replace(result, automaton=system)
    if found is not None and SUBSHIFT in sources:
        result = _run_subshift(
            source,
            kind,
            found[1],
            _get_space(source, found[0]),
            configuration=configuration,
            windows=windows,
            base=base,
            sweeps=sweeps,
            random_state=random_state,
        )
        return replace(result, automaton=system)
    if rule is None or _get_rule_source(rule) not in sources:
        raise AnalysisError(error_msg)
    read = _read_automaton(source, rule)
    if kind == 'attractors':
        step = functools.partial(_step, read)
        result = _get_attractors(read.system, step, _get_states_count(read), read.size)
    elif kind in {'injective', 'surjective', 'pre-injective'}:
        result = _decide(read, kind, method=method, window=window, infinite=infinite)
    elif kind in {'compose', 'difference'}:
        result = _run_pair(read, kind, system, second=second, configuration=configuration, site=site, steps=steps)
    else:
        result = _measure(
            read,
            kind,
            system,
            window=window,
            tries=tries,
            random_state=random_state,
            max_radius=max_radius,
            windows=windows,
            base=base,
        )
    return replace(result, automaton=system)


def _decide(
    automaton: _Automaton,
    kind: str,
    *,
    method: str,
    window: Sequence[int] | None,
    infinite: bool,
) -> AutomatonResult:
    """Return whether the automaton is injective, surjective, or pre-injective."""
    if isinstance(automaton.law._get_rule(), LinearRuleSpec):
        decision = (
            _linear_injective(automaton, infinite=infinite)
            if kind == 'injective'
            else _linear_surjective(automaton, infinite=infinite)
        )
    elif kind == 'pre-injective':
        decision = _is_pre_injective(automaton, window, method=method)
    else:
        decision = _is_injective(automaton, method=method)
    if kind == 'injective':
        return AutomatonResult(automaton.system, injective=decision)
    if kind == 'surjective':
        return AutomatonResult(automaton.system, surjective=decision)
    return AutomatonResult(automaton.system, pre_injective=decision)


def _run_pair(
    automaton: _Automaton,
    kind: str,
    system: System,
    *,
    second: _CompositeSystem | None,
    configuration: npt.ArrayLike | None,
    site: int | Sequence[int],
    steps: int,
) -> AutomatonResult:
    """Return the composition with the second automaton, or a difference pattern.

    The difference pattern is that of a changed configuration. The composition holds the elements of the system.
    """
    if kind == 'compose':
        if second is None:
            error_msg = 'automaton of kind compose needs second. Give it.'
            raise AnalysisError(error_msg)
        return AutomatonResult(automaton.system, composition=_compose(automaton, second, system))
    if configuration is None:
        error_msg = 'automaton of kind difference needs configuration. Give it.'
        raise AnalysisError(error_msg)
    step = functools.partial(_step, automaton)
    differences = _difference_pattern(step, _get_states_count(automaton), configuration, site, steps)
    return AutomatonResult(automaton.system, differences=differences, distances=differences.sum(axis=1))


def _measure(
    automaton: _Automaton,
    kind: str,
    system: System,
    *,
    window: Sequence[int] | None,
    tries: int,
    random_state: int | np.random.Generator | None,
    max_radius: int,
    windows: Sequence[BaseSpace] | None,
    base: float,
) -> AutomatonResult:
    """Return a Garden of Eden pattern, inverse, matrix, polynomial, or entropy.

    They are those of the automaton, and the inverse holds the elements of the system.
    """
    declared = automaton.system
    if kind == 'garden-of-eden':
        return AutomatonResult(
            declared,
            pattern=_garden_of_eden(automaton, window, tries=tries, random_state=random_state),
        )
    if kind == 'inverse':
        return AutomatonResult(declared, inverse=_inverse(automaton, system, max_radius))
    if kind == 'matrix':
        return AutomatonResult(declared, matrix=_get_matrix(automaton))
    if kind == 'polynomial':
        return AutomatonResult(declared, polynomial=_get_polynomial(automaton))
    entropies = [_get_window_entropy(automaton, item) for item in windows or ()]
    return AutomatonResult(declared, entropy=np.asarray(entropies) / np.log(base))


def _run_subshift(
    system: _CompositeSystem,
    kind: str,
    subshift: SubshiftSpec,
    space: BaseSpace,
    *,
    configuration: npt.ArrayLike | None,
    windows: Sequence[BaseSpace] | None,
    base: float,
    sweeps: int,
    random_state: int | np.random.Generator | None,
) -> AutomatonResult:
    """Return admissibility, violations, count, sample, or entropy of a subshift.

    The admissibility and the violations are those of a configuration.
    """
    if kind == 'entropy':
        entropies = [np.log(_count_configurations(subshift, window)) / _size_of(window) for window in windows or ()]
        return AutomatonResult(system, entropy=np.asarray(entropies) / np.log(base))
    if kind == 'count':
        return AutomatonResult(system, count=_count_configurations(subshift, space))
    if kind == 'sample':
        return AutomatonResult(system, sample=_sample(subshift, space, configuration, sweeps, random_state))
    if configuration is None:
        error_msg = f'automaton of kind {kind!r} needs configuration. Give it.'
        raise AnalysisError(error_msg)
    violations = _get_violations(subshift, space, configuration)
    if kind == 'violations':
        return AutomatonResult(system, violations=violations)
    return AutomatonResult(system, admissible=not violations.any())


def _get_space(system: _CompositeSystem, element: _GroupElement) -> BaseSpace:
    """Return the prepared space of a group of a system.

    It is its own space or the space of the system.
    """
    space = element.space if element.space is not None else system.space
    if space is None:
        error_msg = f'The group {element.label!r} of the system lives on no space. Give it a lattice or a group.'
        raise AnalysisError(error_msg)
    return _fitted(space)


def _get_declared(source: DataSource) -> DataSource:
    """Return the composite of a system with its plain elements declared.

    Any other source is returned as it is.
    """
    source = _composite(source)
    return PassCompiler()._get_declared_system(source) if isinstance(source, _CompositeSystem) else source


def _get_second(source: DataSource) -> _CompositeSystem:
    """Return the declared composite of the second automaton of a composition.

    Args:
        source:
            The second automaton of the composition.

    Returns:
        composite:
            The declared composite of the automaton.

    Raises:
        AnalysisError:
            The second automaton is not a system.
    """
    declared = _get_declared(source)
    if not isinstance(declared, _CompositeSystem):
        error_msg = f'automaton composes with a system, not {type(source).__name__}. Give a System.'
        raise AnalysisError(error_msg)
    return declared


def _get_local_rule(interaction: BaseInteraction, system: _CompositeSystem) -> _LocalRule | None:
    """Return the local rule an interaction states, or `None` without one.

    An interaction that sets its laws in `_prepare`, such as one that sets
    `UpdateLaw(s.target.alive, rule, neighborhood=...)`, is prepared on the states of the elements its roles bind, and
    its first local rule is returned.
    """
    roles = {}
    for role in interaction.get_requirements().roles:
        path = interaction._get_binding(role)
        element = next((item for item in system.elements if item.label == path), None)
        if element is None or not isinstance(path, str):
            return None
        states = {name: SR.var(name) for name in element.get_states()}
        roles[role] = (path, SimpleNamespace(**states, **_get_parameters(element)))
    namespace = SimpleNamespace(**{role: states for role, (_, states) in roles.items()}, t=SR.var('t'))
    laws = [
        item._get_local_rule() if isinstance(item, UpdateLaw) and item.neighborhood is not None else item
        for item in clone(interaction).prepare(namespace).laws_
    ]
    law = next((item for item in laws if isinstance(item, _LocalRuleLaw)), None)
    if law is None:
        return None
    path = next(path for path, states in roles.values() if str(law._get_targets()[0]) in vars(states))
    targets = tuple(map(str, law._get_targets()))
    return _LocalRule(
        interaction.label,
        target=path,
        state=targets[0] if len(targets) == 1 else targets,
        reads=None if law.reads is None else tuple(map(str, law.reads)),
        law=law,
    )


def _find_rule(system: _CompositeSystem) -> _LocalRule | None:
    """Return the local rule of a system.

    The result is `None` without one, and an error is raised for several.
    """
    found = [_get_local_rule(item, system) for item in system.interactions]
    rules = [item for item in found if item is not None]
    if len(rules) > 1:
        names = ', '.join(rule.label for rule in rules)
        error_msg = f'An automaton has one local rule, but the system has {names}. Give it one.'
        raise AnalysisError(error_msg)
    return rules[0] if rules else None


def _find_subshift(system: _CompositeSystem) -> tuple[_GroupElement, SubshiftSpec] | None:
    """Return the first group of a system with a value whose domain is a subshift.

    It comes with its spec, or is `None`. The value is a state, or a parameter when no law changes it, as the cells
    of a window whose words are counted.
    """
    for element in system.elements:
        if not isinstance(element, _GroupElement):
            continue
        for spec in element._get_member_specs().values():
            domain = getattr(spec, 'domain', None)
            if isinstance(domain, SubshiftSpec):
                domain._check()
                return element, domain
    return None


def _get_rule_source(rule: _LocalRule | None) -> str | None:
    """Return what a local rule makes of its system.

    It is a linear automaton or an automaton, or `None` without one.
    """
    if rule is None:
        return None
    return LINEAR if isinstance(rule.law.rule, LinearRuleSpec) else AUTOMATON


def _get_states(domain: object, rule: object) -> int | None:
    """Return the number of values of a cell.

    It comes from a finite domain of a state or else from a rule, or is `None`.
    """
    if isinstance(domain, SubshiftSpec):
        return domain.states
    if isinstance(domain, FiniteEnumeratedSet):
        values = list(domain)
        if values != list(range(len(values))):
            error_msg = f'An automaton has the cell values 0 to n - 1, but its state has the domain {domain}.'
            raise AnalysisError(error_msg)
        return len(values)
    if isinstance(rule, LinearRuleSpec):
        return rule.characteristic or None
    if isinstance(rule, TableSpec):
        return rule.n_states
    return rule.states if isinstance(rule, TotalisticSpec | OuterTotalisticSpec) else None


def _read_automaton(system: _CompositeSystem, rule: _LocalRule) -> _Automaton:
    """Return the law, space, size, number of states, and local map of an automaton.

    The automaton is that of a system.
    """
    target = rule.target
    element = next((item for item in system.elements if item.label == target), None)
    if not isinstance(element, _GroupElement) or not isinstance(rule.state, str) or rule.reads is not None:
        error_msg = (
            f'The local rule {rule.label!r} of an automaton sets and reads one state of a group of the system, '
            f'but it binds {target!r} and the states {rule.state!r} and {rule.reads!r}.'
        )
        raise AnalysisError(error_msg)
    if rule.law.schedule != SYNCHRONOUS or rule.law.stochastic:
        error_msg = (
            f'An automaton maps configurations to configurations, so the local rule {rule.label!r} is synchronous '
            f'and deterministic, but it has the schedule {rule.law.schedule!r} and '
            f'stochastic={rule.law.stochastic!r}.'
        )
        raise AnalysisError(error_msg)
    law = rule.law
    law._check()
    space = _get_space(system, element)
    states = _get_states(element.get_states()[rule.state].domain, law._get_rule())
    read = _Automaton(system, rule, law, rule.state, space, int(space.size_ or 0), states)
    return replace(read, local_map=_get_local_map(read))


def _get_local_map(automaton: _Automaton) -> IntArray:
    """Return the next value of every pattern of the cells the rule reads.

    The values are indexed by the code of the pattern.
    """
    law = automaton.law
    if isinstance(law._get_rule(), LinearRuleSpec):
        return np.empty(0, dtype=np.int64)
    if automaton.states is None:
        error_msg = (
            f'A rule {law.rule!r} on a state without a finite domain has no number of values. Give the state its '
            'allowed values, such as `alive: Annotated[int, {0, 1}]`.'
        )
        raise AnalysisError(error_msg)
    width = _get_table(automaton, automaton.space).shape[1]
    patterns = _patterns(automaton.states, width)
    try:
        values = np.asarray(law._get_next_values(patterns), dtype=float)
    except TypeError as error:
        error_msg = f'An automaton has a rule of the cells alone, but {law.rule!r} needs more: {error}'
        raise AnalysisError(error_msg) from error
    values = np.broadcast_to(values, (patterns.shape[0],))
    if values.size and (values.min() < 0 or values.max() >= automaton.states or not np.all(values == np.rint(values))):
        error_msg = f'The rule returns values outside the integers 0 to {automaton.states - 1}.'
        raise AnalysisError(error_msg)
    return values.astype(np.int64)


def _get_table(automaton: _Automaton, space: BaseSpace) -> Indices:
    """Return the site of every cell the rule reads at every site of a space.

    The table includes the sites outside it.
    """
    return _fitted(space)._get_index_table(automaton.law._get_neighborhood(_fitted(space)))


def _step(automaton: _Automaton, configuration: npt.ArrayLike, space: BaseSpace | None = None) -> IntArray:
    """Return the image of a configuration by one run of the law.

    It also takes one configuration per row.
    """
    inputs = {'space': automaton.space if space is None else space, automaton.state: np.asarray(configuration)}
    return np.asarray(automaton.law._get_value(inputs), dtype=np.int64)


def _get_states_count(automaton: _Automaton) -> int:
    """Return the number of values of a cell of an automaton of a rule."""
    return int(automaton.states or 0)


def _fitted(space: BaseSpace) -> BaseSpace:
    """Return a space of sites, preparing it on its first use."""
    if not hasattr(space, 'size_'):
        space.prepare()
    return space


def _size_of(space: BaseSpace) -> int:
    """Return the number of sites of a space."""
    return int(_fitted(space).size_ or 0)


def _patterns(states: int, width: int) -> IntArray:
    """Return every pattern of values of a number of cells.

    The patterns come in the order of their codes.
    """
    if states**width > MAX_PATTERNS:
        error_msg = f'A rule on {width} cells with {states} values has more than {MAX_PATTERNS} patterns.'
        raise AnalysisError(error_msg)
    return np.asarray(list(itertools.product(range(states), repeat=width)), dtype=np.int64).reshape(-1, width)


def _codes(values: IntArray, states: int) -> IntArray:
    """Return the code of every row of values.

    The values are read as digits in the base of the number of states.
    """
    weights = states ** np.arange(values.shape[-1] - 1, -1, -1, dtype=np.int64)
    return values @ weights


def _configurations(states: int, size: int) -> IntArray:
    """Return every configuration of a number of sites.

    An error is raised when there are too many.
    """
    if states**size > MAX_CONFIGURATIONS:
        error_msg = (
            f'{states}^{size} configurations exceed the enumeration limit of {MAX_CONFIGURATIONS}. Use the SAT '
            'method, or a smaller universe.'
        )
        raise AnalysisError(error_msg)
    return _patterns(states, size)


def _check_method(method: str, enumerable: bool) -> bool:
    """Return whether to enumerate configurations, after checking the method."""
    if method not in METHODS:
        error_msg = f'Unknown method {method!r}. Use one of {", ".join(METHODS)}.'
        raise AnalysisError(error_msg)
    return method == 'enumeration' or (method == 'auto' and enumerable)


def _rule_clauses(
    automaton: _Automaton,
    clauses: _Clauses,
    cells: npt.NDArray[np.int64],
    sites: Sequence[int],
    output: Callable[[int, int], list[int] | None],
) -> None:
    """Add the clause that a pattern implies the output of the rule.

    There is one clause for every site and pattern of its cells.
    """
    table = _get_table(automaton, automaton.space)
    patterns = _patterns(_get_states_count(automaton), table.shape[1])
    fill = int(getattr(automaton.space, 'fill', 0))
    for site in sites:
        row = table[site]
        for pattern, value in zip(patterns, automaton.local_map, strict=True):
            outside = row >= automaton.size
            if np.any(pattern[outside] != fill):
                continue
            consequence = output(site, int(value))
            if consequence is None:
                continue
            premise = [
                -int(cells[neighbour, item]) for neighbour, item in zip(row[~outside], pattern[~outside], strict=True)
            ]
            clauses.clauses.append(premise + consequence)


def _decode(system: list[int], variables: npt.NDArray[np.int64]) -> IntArray:
    """Return the values of one-hot variables in a system."""
    true = set(system)
    return np.asarray(
        [next(value for value, item in enumerate(row) if item in true) for row in variables.tolist()],
        dtype=np.int64,
    )


def _collision(automaton: _Automaton, window: Sequence[int]) -> tuple[IntArray, IntArray] | None:
    """Return two distinct configurations with the same image, by SAT.

    They differ inside a window and agree outside it.
    """
    clauses = _Clauses()
    states = _get_states_count(automaton)
    first = clauses.get_one_hot(automaton.size, states)
    second = clauses.get_one_hot(automaton.size, states)
    image = clauses.get_one_hot(automaton.size, states, exactly=False)
    for cells in (first, second):
        _rule_clauses(automaton, clauses, cells, range(automaton.size), lambda site, value: [int(image[site, value])])
    inside = set(window)
    differences = []
    for site in range(automaton.size):
        for value in range(states):
            same = (int(first[site, value]), int(second[site, value]))
            if site not in inside:
                clauses.clauses += [[-same[0], same[1]], [same[0], -same[1]]]
                continue
            differ = clauses.get_fresh()
            clauses.clauses += [[-differ, same[0]], [-differ, -same[1]]]
            differences.append(differ)
    clauses.clauses.append(differences)
    system = clauses.solve()
    return None if system is None else (_decode(system, first), _decode(system, second))


def _preimage(automaton: _Automaton, target: IntArray, window: Sequence[int]) -> IntArray | None:
    """Return a configuration whose image equals a target on a window, by SAT.

    The result is `None` when there is none.
    """
    clauses = _Clauses()
    cells = clauses.get_one_hot(automaton.size, _get_states_count(automaton))

    def forbid(site: int, value: int) -> list[int] | None:
        """Return an empty consequence when the rule misses the target.

        The empty consequence forbids the pattern.
        """
        return None if value == target[site] else []

    _rule_clauses(automaton, clauses, cells, window, forbid)
    system = clauses.solve()
    return None if system is None else _decode(system, cells)


def _images(automaton: _Automaton) -> tuple[IntArray, IntArray]:
    """Return every configuration and the code of its image."""
    states = _get_states_count(automaton)
    configurations = _configurations(states, automaton.size)
    return configurations, _codes(_step(automaton, configurations), states)


def _ball_neighborhood(automaton: _Automaton, radius: int) -> BaseNeighborhood:
    """Return the neighbourhood of every cell within a radius.

    It has fixed slots, such as a box of offsets or a ball of group elements.
    """
    reach = automaton.law._get_neighborhood(_fitted(automaton.space))
    if isinstance(automaton.space, LatticeSpace) and not isinstance(reach, LatticeNeighborhood):
        error_msg = 'An inverse is searched on a lattice with a neighbourhood of offsets or on a group space.'
        raise AnalysisError(error_msg)
    ball = automaton.space._get_ball(radius)
    if ball._get_slots() is None:
        error_msg = f'The space of the automaton gives a ball of {type(ball).__name__}, which an automaton cannot read.'
        raise AnalysisError(error_msg)
    return ball


def _is_enumerable(automaton: _Automaton) -> bool:
    """Check whether the configurations of the automaton are few enough to enumerate."""
    return _get_states_count(automaton) ** automaton.size <= MAX_CONFIGURATIONS


def _is_injective(automaton: _Automaton, *, method: str = 'auto') -> bool:
    """Check whether two different configurations never have the same image."""
    if _check_method(method, _is_enumerable(automaton)):
        _, codes = _images(automaton)
        return bool(np.unique(codes).size == codes.size)
    return _collision(automaton, range(automaton.size)) is None


def _is_pre_injective(automaton: _Automaton, window: Sequence[int] | None = None, *, method: str = 'auto') -> bool:
    """Check whether configurations that differ inside a window never share an image."""
    sites = list(range(automaton.size)) if window is None else [int(site) for site in window]
    if not _check_method(method, _is_enumerable(automaton)):
        return _collision(automaton, sites) is None
    states = _get_states_count(automaton)
    configurations, codes = _images(automaton)
    outside = np.setdiff1d(np.arange(automaton.size), sites)
    keys = codes * states**outside.size + _codes(configurations[:, outside], states)
    return bool(np.unique(keys).size == keys.size)


def _garden_of_eden(
    automaton: _Automaton,
    window: Sequence[int] | None = None,
    *,
    tries: int = 200,
    random_state: int | np.random.Generator | None = None,
) -> IntArray | None:
    """Return a pattern on a window that no configuration maps to, or `None`."""
    states = _get_states_count(automaton)
    sites = list(range(automaton.size)) if window is None else [int(site) for site in window]
    if window is None and _is_enumerable(automaton):
        configurations, codes = _images(automaton)
        missing = np.setdiff1d(np.arange(states**automaton.size), codes)
        return configurations[missing[0]] if missing.size else None
    rng = np.random.default_rng(random_state)
    for _ in range(tries):
        target = rng.integers(states, size=automaton.size)
        if _preimage(automaton, target, sites) is None:
            return target
    return None


def _get_rule_system(
    automaton: _Automaton,
    system: System,
    neighborhood: BaseNeighborhood,
    rule: object,
    name: str,
) -> System:
    """Return a `System` of the analysed elements with a new local rule.

    The interaction sets an update law with the rule and the neighbourhood on the same state.
    """
    source = automaton.rule
    laws = functools.partial(_get_local_rule_laws, state=automaton.state, rule=rule, neighborhood=neighborhood)
    interaction = Interaction(
        f'{source.label}_{name}',
        roles={'target': source.target},
        laws=laws,
    )
    return System(system.label, elements=list(system.elements), interactions=[interaction], space=system.space)


def _get_local_rule_laws(
    s: SimpleNamespace,
    *,
    state: str,
    rule: object,
    neighborhood: BaseNeighborhood,
) -> list[BaseLaw]:
    """Return the update law of a state of the target by a rule of its neighbourhood."""
    return [UpdateLaw(getattr(s.target, state), rule, neighborhood=neighborhood)]


def _inverse(automaton: _Automaton, system: System, max_radius: int = 2) -> System | None:
    """Return the `System` of the inverse of a reversible automaton.

    It holds the elements of the system. Its composition with the automaton is the identity, and it is `None` when
    there is none.
    """
    configurations, codes = _images(automaton)
    if np.unique(codes).size != codes.size:
        return None
    states = _get_states_count(automaton)
    images = _step(automaton, configurations)
    for radius in range(max_radius + 1):
        ball = _ball_neighborhood(automaton, radius)
        table = automaton.space._get_index_table(ball)
        padded = np.concatenate([images, np.zeros((images.shape[0], 1), dtype=np.int64)], axis=1)
        patterns = _codes(padded[:, table], states).ravel()
        values = configurations.ravel()
        pairs = np.unique(np.column_stack([patterns, values]), axis=0)
        if np.unique(pairs[:, 0]).size != pairs.shape[0]:
            continue
        rows = _patterns(states, table.shape[1])
        mapping = {tuple(int(item) for item in rows[code]): int(value) for code, value in pairs}
        return _get_rule_system(automaton, system, ball, TableSpec(mapping, states=states), 'inverse')
    return None


def _get_reach(automaton: _Automaton) -> BaseNeighborhood:
    """Return the neighbourhood the rule of the automaton reads.

    It has fixed slots, such as offsets or group elements.
    """
    reach = automaton.law._get_neighborhood(_fitted(automaton.space))
    if reach._get_slots() is None:
        error_msg = f'An automaton reads offsets or a memory set, not {reach!r}.'
        raise AnalysisError(error_msg)
    return reach


def _compose(automaton: _Automaton, second: _CompositeSystem, system: System) -> System:
    """Return the `System` of the automaton that applies this one then a second one.

    It holds the elements of the system, and both automata act on the same space.
    """
    rule = _find_rule(second)
    if rule is None or isinstance(rule.law.rule, LinearRuleSpec):
        error_msg = 'Composed automata are systems with a local rule that is not linear.'
        raise AnalysisError(error_msg)
    other = _read_automaton(second, rule)
    if automaton.space is not other.space or automaton.states != other.states:
        error_msg = 'Composed automata share their space and their number of states.'
        raise AnalysisError(error_msg)
    inner, outer = _get_reach(automaton), _get_reach(other)
    composite: list[Hashable] = []
    columns = []
    for base in outer._get_slots() or ():
        row = []
        for slot in inner._get_slots() or ():
            item = inner._join_slots(base, slot)
            if item not in composite:
                composite.append(item)
            row.append(composite.index(item))
        columns.append(tuple(row))
    reach = inner._get_from_slots(composite)
    composed = _ComposedRule(automaton.local_map, other.local_map, tuple(columns), _get_states_count(automaton))
    return _get_rule_system(automaton, system, reach, functools.partial(_composed_rule, composed), 'composition')


def _composed_rule(composed: _ComposedRule, *values: npt.ArrayLike, **kwargs: object) -> IntArray:
    """Return the value of the second rule of a composition on the first rule's values.

    The values of the first rule are those around every cell.
    """
    cells = np.stack([np.asarray(value, dtype=np.int64) for value in values], axis=-1)
    inner = [composed.first[_codes(cells[..., list(columns)], composed.states)] for columns in composed.columns]
    return composed.second[_codes(np.stack(inner, axis=-1), composed.states)]


def _difference_pattern(
    step: Callable[[IntArray], IntArray],
    states: int,
    configuration: npt.ArrayLike,
    site: int | Sequence[int],
    steps: int,
) -> npt.NDArray[np.bool_]:
    """Return where two runs of a map differ when one starts with some values changed.

    Every changed value takes its next value.
    """
    initial = np.asarray(configuration, dtype=np.int64)
    changed = initial.copy()
    changed[site] = (changed[site] + 1) % states
    rows = [np.stack([initial, changed])]
    for _ in range(steps):
        rows.append(step(rows[-1]))
    runs = np.stack(rows)
    return runs[:, 0] != runs[:, 1]


def _run_network(
    network: _Network,
    kind: str,
    *,
    configuration: npt.ArrayLike | None,
    site: int | Sequence[int],
    steps: int,
) -> AutomatonResult:
    """Return the attractors of a finite network, or the difference of two of its runs."""
    step = functools.partial(_get_network_images, network)
    if kind == 'attractors':
        return _get_attractors(network.system, step, network.states, network.size)
    if configuration is None:
        error_msg = 'automaton of kind difference needs configuration. Give it.'
        raise AnalysisError(error_msg)
    differences = _difference_pattern(step, network.states, configuration, site, steps)
    return AutomatonResult(network.system, differences=differences, distances=differences.sum(axis=1))


def _get_attractors(
    system: _CompositeSystem,
    step: Callable[[IntArray], IntArray],
    states: int,
    size: int,
) -> AutomatonResult:
    """Return the transient and the cycle of every configuration of a map.

    The result also holds the length and the basin of every cycle. The map is applied to the configurations in
    chunks, and the cycles are found on the graph from every code to the code of its image.
    """
    count = states**size
    if count > MAX_STATE_SPACE:
        error_msg = (
            f'automaton of kind attractors maps every configuration, but the system has {states}^{size} = {count} '
            f'configurations, more than MAX_STATE_SPACE = {MAX_STATE_SPACE}. Use a smaller system.'
        )
        raise AnalysisError(error_msg)
    images = np.concatenate(
        [
            _codes(step(_get_digits(np.arange(start, min(start + CHUNK, count)), states, size)), states)
            for start in range(0, count, CHUNK)
        ],
    )
    current = np.unique(images)
    following = np.unique(images[current])
    while following.size < current.size:
        current, following = following, np.unique(images[following])
    labels = _get_cycle_labels(current, images)
    _, cycle_of, lengths = np.unique(labels, return_inverse=True, return_counts=True)
    transients = np.full(count, -1, dtype=np.int64)
    attractors = np.full(count, -1, dtype=np.int64)
    transients[current] = 0
    attractors[current] = cycle_of
    pending = np.flatnonzero(transients < 0)
    while pending.size:
        ready = transients[images[pending]] >= 0
        chosen = pending[ready]
        transients[chosen] = transients[images[chosen]] + 1
        attractors[chosen] = attractors[images[chosen]]
        pending = pending[~ready]
    cycles = tuple(_get_digits(_get_cycle(int(start), images), states, size) for start in current[np.unique(labels)])
    return AutomatonResult(
        system,
        transients=transients,
        attractors=attractors,
        cycles=cycles,
        lengths=lengths.astype(np.int64),
        basins=np.bincount(attractors, minlength=lengths.size).astype(np.int64),
    )


def _get_cycle_labels(cycle: IntArray, images: IntArray) -> IntArray:
    """Return the position of the smallest code of the cycle of every code on a cycle.

    The codes on cycles are sorted, and the minimum over ever longer stretches of the cycle is taken by doubling.
    """
    following = np.searchsorted(cycle, images[cycle])
    labels = np.arange(cycle.size)
    for _ in range(max(int(cycle.size).bit_length(), 1)):
        labels = np.minimum(labels, labels[following])
        following = following[following]
    return labels


def _get_cycle(start: int, images: IntArray) -> IntArray:
    """Return the codes of the cycle through a code, in the order of the map."""
    codes = [start]
    image = int(images[start])
    while image != start:
        codes.append(image)
        image = int(images[image])
    return np.asarray(codes, dtype=np.int64)


def _get_digits(codes: IntArray, states: int, size: int) -> IntArray:
    """Return the values of the configuration of every code.

    The first value is the most significant digit.
    """
    return (codes[:, None] // states ** np.arange(size - 1, -1, -1, dtype=np.int64)) % states


def _get_network_images(network: _Network, values: IntArray) -> IntArray:
    """Return the image of every row of values of a finite network.

    It is one synchronous step of its laws.
    """
    images = values.copy()
    for law in network.laws:
        read = values[..., law.reads]
        expected = (*values.shape[:-1], law.targets.size)
        try:
            image = np.broadcast_to(np.asarray(law.function(read)), expected)
        except ValueError as error:
            error_msg = (
                f'The law of {law.label!r} gives values that do not fit one value per target and configuration of '
                f'the shape {expected}: {error}'
            )
            raise AnalysisError(error_msg) from error
        integral = image.dtype.kind in INTEGER_KINDS or np.all(image == np.rint(image))
        if image.size and (image.min() < 0 or image.max() >= network.states or not integral):
            error_msg = f'The law of {law.label!r} gives values outside the integers 0 to {network.states - 1}.'
            raise AnalysisError(error_msg)
        images[..., law.targets] = image.astype(np.int64)
    return images


def _read_network(system: _CompositeSystem) -> _Network:
    """Return the finite network of a system.

    It holds the number of values of a state, the number of values of a configuration, and the update laws. Every
    state of every element takes the values 0 to m - 1, the same m for all. Every law sets one value by an
    expression of single values, or the values of a group by a deterministic call on them.
    """
    columns: dict[str, IntArray] = {}
    namespaces: dict[str, SimpleNamespace] = {}
    counts: set[int] = set()
    for element in system.elements:
        size = element._get_size_of(system.space) if isinstance(element, _GroupElement) else 1
        symbols = {}
        for name, spec in element.get_states().items():
            symbol = SR.var(f'value_{len(columns)}')
            start = sum(item.size for item in columns.values())
            columns[str(symbol)] = np.arange(start, start + size)
            symbols[name] = symbol
            count = _get_states(spec.domain, None)
            if count is None:
                error_msg = (
                    f'A finite network has states of finitely many values, but {element.label}.{name} has the domain '
                    f'{spec.domain}. Give it the values 0 to m - 1, such as `x: Annotated[int, {{0, 1}}]`.'
                )
                raise AnalysisError(error_msg)
            counts.add(count)
        namespaces[element.label] = SimpleNamespace(**symbols, **_get_parameters(element))
    if len(counts) != 1:
        error_msg = f'The states of a finite network share their number of values, but they take {sorted(counts)}.'
        raise AnalysisError(error_msg)
    states = counts.pop()
    laws = [
        _read_network_law(interaction.label, law, columns, states)
        for interaction in system.interactions
        for law in clone(interaction).prepare(_get_namespace(interaction, namespaces)).laws_
    ]
    return _Network(system, states, sum(item.size for item in columns.values()), tuple(laws))


def _get_parameters(element: object) -> dict[str, object]:
    """Return the values of the parameters of an element.

    A parameter of a group with a row per member gives one column per entry of the rows, as laws read it.
    """
    if isinstance(element, _DeclaredElement):
        variables = element.variables or {}
        return {name: item.value for name, item in variables.items() if isinstance(item.spec, _Parameter)}
    if not isinstance(element, _GroupElement):
        return {}
    values = element.values or {}
    parameters = {}
    for name, spec in element._get_member_specs().items():
        if isinstance(spec, _Parameter) and name in values:
            array = np.asarray(values[name])
            parameters[name] = tuple(array.T) if array.ndim == MATRIX_AXES else values[name]
    return parameters


def _get_namespace(interaction: BaseInteraction, namespaces: dict[str, SimpleNamespace]) -> SimpleNamespace:
    """Return the namespace an interaction states its laws on.

    It holds the elements its roles bind and its parameters.
    """
    roles = interaction.get_requirements().roles
    parameters = {name: value for name, value in interaction.get_params(deep=False).items() if name not in roles}
    bound: dict[str, object] = {}
    for role in roles:
        binding = interaction._get_binding(role)
        paths = [binding] if isinstance(binding, str) else [str(path) for path in cast('Sequence[str]', binding or ())]
        missing = [path for path in paths if path not in namespaces]
        if missing:
            error_msg = (
                f'A finite network binds its elements by their labels, but {interaction.label!r} binds {missing} to '
                f'{role!r}.'
            )
            raise AnalysisError(error_msg)
        items = [namespaces[path] for path in paths]
        bound[role] = items[0] if isinstance(binding, str) else items
    return SimpleNamespace(**parameters, **bound, t=SR.var('t'))


def _read_network_law(label: str, law: BaseLaw, columns: dict[str, IntArray], states: int) -> _NetworkLaw:
    """Return the values an update law of a finite network sets and reads.

    It comes with the function from the values it reads to those it sets.
    """
    target = columns.get(str(getattr(law, 'target', None)))
    if not isinstance(law, UpdateLaw) or law.neighborhood is not None or law.increment or target is None:
        error_msg = (
            f'A finite network sets its values by update laws without neighbourhoods, but {label!r} states {law!r}.'
        )
        raise AnalysisError(error_msg)
    if isinstance(law.value, CallSpec):
        return _read_call(label, law.value, target, columns)
    value = SR(law.value)
    symbols = list(value.variables())
    reads = [columns.get(str(symbol), np.empty(0, dtype=np.int64)) for symbol in symbols]
    if target.size != 1 or any(item.size != 1 for item in reads):
        error_msg = (
            f'An expression of a finite network sets one value from single values, but {label!r} sets {target.size} '
            f'values from {symbols}.'
        )
        raise AnalysisError(error_msg)
    rows = _patterns(states, len(symbols)).tolist() if symbols else [[]]
    try:
        table = np.asarray([float(value.subs(dict(zip(symbols, row, strict=True)))) for row in rows])
    except TypeError as error:
        error_msg = f'The update of {label!r} gives no number for some values: {error}'
        raise AnalysisError(error_msg) from error
    reads_columns = np.concatenate([np.empty(0, dtype=np.int64), *reads])
    return _NetworkLaw(label, target, reads_columns, functools.partial(_get_table_values, table, states))


def _read_call(label: str, call: CallSpec, target: IntArray, columns: dict[str, IntArray]) -> _NetworkLaw:
    """Return the values a call of a finite network sets and reads.

    It comes with the function from the values it reads to those it sets.
    """
    if call.random or call.relative or not call.vectorized or call.arguments is None:
        error_msg = (
            f'A finite network maps configurations to configurations, so the call of {label!r} is vectorized and '
            f'draws nothing, but it has random={call.random!r} and vectorized={call.vectorized!r}.'
        )
        raise AnalysisError(error_msg)
    reads: list[IntArray] = []
    slots: list[slice | None] = []
    for argument in call.arguments:
        if isinstance(argument, Expression) and str(argument) in columns:
            start = sum(item.size for item in reads)
            reads.append(columns[str(argument)])
            slots.append(slice(start, start + reads[-1].size))
        elif isinstance(argument, Expression) and argument.variables():
            error_msg = f'The call of {label!r} reads {argument}, which is no state of the network.'
            raise AnalysisError(error_msg)
        else:
            slots.append(None)
    reads_columns = np.concatenate([np.empty(0, dtype=np.int64), *reads])
    return _NetworkLaw(label, target, reads_columns, functools.partial(_get_call_values, call, tuple(slots)))


def _get_table_values(table: npt.NDArray[np.float64], states: int, values: IntArray) -> npt.NDArray[np.float64]:
    """Return the entry of a table at the code of every row of values, as a column."""
    return table[_codes(values, states)][..., None]


def _get_call_values(call: CallSpec, slots: tuple[slice | None, ...], values: IntArray) -> npt.ArrayLike:
    """Return the values of a call on rows of the values it reads.

    The arguments without a slot are passed as they are.
    """
    arguments = [
        argument if slot is None else values[..., slot]
        for argument, slot in zip(call.arguments or (), slots, strict=True)
    ]
    function = cast('Callable[..., npt.ArrayLike]', call.function)
    return function(0.0, *arguments)


def _get_window_entropy(automaton: _Automaton, window: BaseSpace) -> float:
    """Return the entropy `log |X_F| / |F|` of the image of the automaton on a window."""
    states = _get_states_count(automaton)
    interior = np.flatnonzero((_get_table(automaton, window) < _size_of(window)).all(axis=1))
    images = _step(automaton, _configurations(states, _size_of(window)), window)[:, interior]
    count = np.unique(_codes(images, states)).size
    return float(np.log(count) / max(interior.size, 1))


def _get_linear_rule(automaton: _Automaton) -> LinearRuleSpec:
    """Return the linear rule of the automaton, or raise an error for any other rule."""
    rule = automaton.law._get_rule()
    if not isinstance(rule, LinearRuleSpec):
        error_msg = f'The kind reads a linear automaton, not {rule!r}.'
        raise AnalysisError(error_msg)
    return rule


def _get_matrix(automaton: _Automaton) -> Matrix:
    """Return the matrix of a linear automaton over its field.

    Its row `g` gives the image at `g`.
    """
    rule = _get_linear_rule(automaton)
    table = _get_table(automaton, automaton.space)
    if (table >= automaton.size).any():
        error_msg = 'A linear automaton has a matrix on a periodic lattice or a finite group, not on a ball.'
        raise AnalysisError(error_msg)
    field = rule.sage_field
    entries = matrix(field, automaton.size, automaton.size)
    for site, row in enumerate(table.tolist()):
        for neighbour, coefficient in zip(row, rule.coefficients, strict=True):
            entries[site, neighbour] += field(QQ(coefficient))
    return entries


def _get_polynomial(automaton: _Automaton) -> LaurentPolynomial:
    """Return the Laurent polynomial `sum a_s x^s` of a linear automaton on a lattice.

    It is its symbol on `Z^d`.
    """
    rule = _get_linear_rule(automaton)
    reach = automaton.law._get_neighborhood(_fitted(automaton.space))
    if not isinstance(reach, LatticeNeighborhood):
        error_msg = 'A Laurent polynomial describes a linear automaton on a lattice, with offsets.'
        raise AnalysisError(error_msg)
    field = rule.sage_field
    offsets = reach.prepare().offsets_
    ring = LaurentPolynomialRing(field, [f'x{axis}' for axis in range(len(offsets[0]))])
    terms = (
        field(QQ(coefficient)) * np.prod([variable**step for variable, step in zip(ring.gens(), offset, strict=True)])
        for offset, coefficient in zip(offsets, rule.coefficients, strict=True)
    )
    return sum(terms, ring.zero())


def _linear_injective(automaton: _Automaton, *, infinite: bool) -> bool:
    """Check whether a linear automaton is invertible.

    It is when its matrix on its universe is, or when its symbol on `Z^d` is a monomial.
    """
    if infinite:
        return len(_get_polynomial(automaton).exponents()) == 1
    return bool(_get_matrix(automaton).is_invertible())


def _linear_surjective(automaton: _Automaton, *, infinite: bool) -> bool:
    """Check whether a linear automaton is surjective.

    It is when it is invertible on its universe, or when its symbol on `Z^d` is not zero.
    """
    if infinite:
        return not _get_polynomial(automaton).is_zero()
    return _linear_injective(automaton, infinite=False)


def _get_complete(subshift: SubshiftSpec, space: BaseSpace) -> tuple[Indices, npt.NDArray[np.bool_]]:
    """Return the neighbour table of a space and the sites whose patterns lie in it."""
    neighborhood = subshift.neighborhood
    if not isinstance(neighborhood, BaseNeighborhood) or neighborhood._get_slots() is None:
        error_msg = f'A subshift covers offsets or a memory set, not {neighborhood!r}.'
        raise AnalysisError(error_msg)
    table = _fitted(space)._get_index_table(neighborhood)
    return table, (table < _size_of(space)).all(axis=1)


def _get_violations(subshift: SubshiftSpec, space: BaseSpace, configuration: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Return the sites at which a forbidden pattern appears in a configuration.

    The sites are those of the space.
    """
    table, complete = _get_complete(subshift, space)
    values = np.append(np.asarray(configuration, dtype=np.int64), 0)
    forbidden = np.asarray([tuple(pattern) for pattern in subshift.forbidden], dtype=np.int64)
    codes = _codes(values[table], subshift.states)
    return complete & np.isin(codes, _codes(forbidden.reshape(-1, table.shape[1]), subshift.states))


def _count_configurations(subshift: SubshiftSpec, space: BaseSpace) -> int:
    """Return the number of admissible configurations of a space.

    It is computed by dynamic programming over its sites.
    """
    size = _size_of(space)
    table, complete = _get_complete(subshift, space)
    forbidden = {tuple(int(item) for item in pattern) for pattern in subshift.forbidden}
    checks: list[list[int]] = [[] for _ in range(size)]
    for site in np.flatnonzero(complete):
        checks[int(table[site].max())].append(int(site))
    needed = [
        tuple(
            sorted(
                {int(cell) for later in checks[site:] for anchor in later for cell in table[anchor] if cell < site},
            ),
        )
        for site in range(size + 1)
    ]
    values = [0] * size
    memo: dict[tuple[int, tuple[int, ...]], int] = {}

    def extend(site: int) -> int:
        """Return the number of admissible completions of the values before a site."""
        if site == size:
            return 1
        key = (site, tuple(values[cell] for cell in needed[site]))
        if key not in memo:
            total = 0
            for value in range(subshift.states):
                values[site] = value
                if all(tuple(values[cell] for cell in table[anchor]) not in forbidden for anchor in checks[site]):
                    total += extend(site + 1)
            memo[key] = total
        return memo[key]

    return extend(0)


def _sample(
    subshift: SubshiftSpec,
    space: BaseSpace,
    configuration: npt.ArrayLike | None,
    sweeps: int,
    random_state: int | np.random.Generator | None,
) -> IntArray:
    """Return an admissible configuration drawn by a Markov chain.

    The chain makes single-cell moves that keep it admissible.
    """
    rng = np.random.default_rng(random_state)
    size = _size_of(space)
    values = (
        np.zeros(size, dtype=np.int64) if configuration is None else np.asarray(configuration, dtype=np.int64).copy()
    )
    if _get_violations(subshift, space, values).any():
        error_msg = 'The initial configuration holds a forbidden pattern. Give an admissible one.'
        raise AnalysisError(error_msg)
    table, complete = _get_complete(subshift, space)
    touching: list[list[int]] = [[] for _ in range(size)]
    for anchor in np.flatnonzero(complete):
        for cell in set(table[anchor].tolist()):
            touching[cell].append(int(anchor))
    forbidden = {tuple(int(item) for item in pattern) for pattern in subshift.forbidden}
    for _ in range(sweeps):
        for site in rng.permutation(size):
            previous = values[site]
            values[site] = rng.integers(subshift.states)
            if any(tuple(values[table[anchor]].tolist()) in forbidden for anchor in touching[site]):
                values[site] = previous
    return values
