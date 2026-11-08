"""Decide injectivity and surjectivity of cellular automata and subshifts.

The module also measures their entropy.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import QQ, SR, LaurentPolynomialRing, matrix
from sage.rings.polynomial.laurent_polynomial import LaurentPolynomial
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.structure.element import Matrix

from ..components.elements import GroupElement
from ..components.interactions import LocalRuleInteraction
from ..components.laws import LocalRuleLaw
from ..components.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace, MemoryNeighborhood
from ..components.system import System
from ..components.types import Reach
from ..errors import AnalysisError
from ..specs import AnalysisTags, BaseSpec, LinearRuleSpec, OuterTotalisticSpec, SubshiftSpec, TableSpec, TotalisticSpec
from ..specs._composed_rule_spec import _ComposedRuleSpec
from ..types import DataKind, Indices, IntArray
from ._base import _check_choice, _check_settings, _composite, _import
from .types import Item, Source

MAX_CONFIGURATIONS = 2**16
MAX_PATTERNS = 2**20
SAT_SOLVER = 'cadical153'
METHODS = ('auto', 'enumeration', 'sat')
SYNCHRONOUS = 'synchronous'
IDENTITY = '1'

TAGS = AnalysisTags(DataKind.SYSTEM)


class _Clauses:
    """Clauses of a SAT problem over one-hot variables of site values."""

    def __init__(self) -> None:
        self.clauses: list[list[int]] = []
        self.count = 0

    def one_hot(self, sites: int, states: int, *, exactly: bool = True) -> npt.NDArray[np.int64]:
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

    def fresh(self) -> int:
        """Return a new variable."""
        self.count += 1
        return self.count

    def solve(self) -> list[int] | None:
        """Return a system of the clauses, or `None` when they are unsatisfiable."""
        solvers = _import('pysat.solvers')
        with solvers.Solver(name=SAT_SOLVER, bootstrap_with=self.clauses) as solver:
            return solver.get_model() if solver.solve() else None


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
}
"""Settings every kind reads, besides `kind`."""

AUTOMATON = 'an automaton'
LINEAR = 'a linear automaton'
SUBSHIFT = 'a subshift'
SOURCES = {
    'injective': (AUTOMATON, LINEAR),
    'surjective': (AUTOMATON, LINEAR),
    'pre-injective': (AUTOMATON, LINEAR),
    'garden-of-eden': (AUTOMATON,),
    'inverse': (AUTOMATON,),
    'compose': (AUTOMATON,),
    'difference': (AUTOMATON,),
    'entropy': (SUBSHIFT, AUTOMATON),
    'matrix': (LINEAR,),
    'polynomial': (LINEAR,),
    'admissible': (SUBSHIFT,),
    'violations': (SUBSHIFT,),
    'count': (SUBSHIFT,),
    'sample': (SUBSHIFT,),
}
"""What every kind reads in a system, in the order it looks for them."""


@dataclass(frozen=True)
class AutomatonResult:
    """Decision or measure of a cellular automaton or a subshift of a system.

    The cellular automaton is on a finite universe.

    Args:
        automaton:
            The system.

        injective:
            Whether the automaton is injective.

        surjective:
            Whether the automaton is surjective.

        pre_injective:
            Whether no two configurations that differ only inside the window share their image.

        pattern:
            A configuration whose values on the window have no preimage, or `None` when none is found.

        inverse:
            The system of the inverse automaton with a table rule, or `None` when the automaton is not injective or no
            memory up to the radius determines the cells.

        composition:
            The system of the automaton of the composition, with a function rule.

        differences:
            One row per step, the initial one first, true where the runs differ.

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


@dataclass(frozen=True)
class _AutomatonSpec(BaseSpec):
    """Automaton read from a system.

    It holds its rule, law, state, space, number of sites and of values, and local map.
    """

    system: System
    rule: LocalRuleInteraction
    law: LocalRuleLaw
    state: str
    space: BaseSpace
    size: int
    states: int | None
    local_map: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def automaton(
    source: Source,
    *,
    kind: str = 'injective',
    method: str = 'auto',
    window: Sequence[int] | None = None,
    infinite: bool = False,
    tries: int = 200,
    random_state: int | np.random.Generator | None = None,
    max_radius: int = 2,
    second: System | None = None,
    configuration: npt.ArrayLike | None = None,
    site: int = 0,
    steps: int = 1,
    windows: Sequence[BaseSpace] | None = None,
    base: float = 2.0,
    sweeps: int = 100,
) -> AutomatonResult:
    """Return a decision or a measure of a cellular automaton or subshift of a system.

    The function reads a `System`. An automaton is a group of cells with one synchronous and deterministic
    `LocalRuleInteraction`, and it is linear when the rule is a `LinearRuleSpec`. The number of values of a cell comes
    from the finite domain of the state, or else from the rule. A subshift is a group with a state whose domain is a
    `SubshiftSpec`.

    - `'injective'`, `'surjective'`, and `'pre-injective'`: whether two different configurations never share their
      image, whether every configuration has a preimage, which on a finite universe holds exactly when it is
      injective, and whether two configurations that differ only inside a window never share their image. Small
      universes are decided by enumerating every configuration, and larger ones by a SAT encoding solved by PySAT. A
      linear automaton is decided from its matrix, all three holding exactly when the matrix is invertible, or on
      `Z^d` from its Laurent polynomial, injective when it is a monomial and surjective and pre-injective when it is
      not zero.
    - `'garden-of-eden'`: a pattern on a window that no configuration maps to. Small universes enumerate the images of
      every configuration, and larger ones draw random patterns and keep the first one whose preimage constraints a
      SAT solver finds unsatisfiable.
    - `'inverse'`: an automaton whose composition with a reversible automaton is the identity, with the smallest
      memory. The search tries the neighbourhoods of every radius up to `max_radius`, a box of offsets on a lattice or
      the ball of group elements on a group space, and keeps the first on which the image determines every cell.
    - `'compose'`: the automaton that applies the automaton and then `second` on the same space, whose neighbourhood
      holds the composite `t s` of every element `t` read by the second and `s` read by the first.
    - `'difference'`: where two runs differ when one starts from `configuration` and the other from it with the cell
      at `site` changed to its next value modulo the number of states.
    - `'entropy'`: the entropy `log |X_F| / |F|` of the configurations of a subshift, or of the image of an automaton
      whose rule and neighbourhood are applied on every window. `X_F` is the set of admissible configurations of a
      window `F`, or the set of images on the sites of `F` whose cells lie inside it. Along a Følner sequence of
      windows the values approach the topological entropy.
    - `'matrix'`: the square matrix of a linear automaton over its field, whose row `g` gives the image at `g`.
    - `'polynomial'`: the Laurent polynomial `sum a_s x^s` of a linear automaton on a lattice, its symbol on `Z^d`.
    - `'admissible'` and `'violations'`: whether no forbidden pattern appears in `configuration`, and the sites at
      which one appears.
    - `'count'`: the number of admissible configurations of the space of a subshift, by dynamic programming over its
      sites. The counts of completions are shared between assignments that agree on the earlier sites that later
      patterns read, so a window of a one-dimensional subshift costs linear time.
    - `'sample'`: an admissible configuration drawn by a Markov chain of single-cell moves that keep it admissible,
      from `configuration`, or from every cell at 0.

    Args:
        source:
            The system of the automaton or of the subshift.

        kind:
            The decision or measure, one of the kinds above.

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
            The site of the changed cell.

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
            states, a linear automaton has a matrix on a ball or a polynomial without offsets, or the initial
            configuration of a sample is not admissible.

    Examples:
        >>> from typing import Annotated
        >>> from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
        >>> from skcomplex.core.analysis import automaton
        >>> from skcomplex.core.components.elements import Element, LatticeElement
        >>> from skcomplex.core.components.interactions import LocalRuleInteraction
        >>> from skcomplex.core.components.spaces import BoxNeighborhood, LatticeNeighborhood, LatticeSpace
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.specs import StateSpec, SubshiftSpec, TableSpec, TotalisticSpec
        >>> class CellElement(Element):
        ...     def __init__(self, label, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        ...         super().__init__(label)
        ...         self.alive = alive
        >>> def build(size, rule, neighborhood):
        ...     cells = LatticeElement('cells', element=CellElement, space=LatticeSpace(size))
        ...     interaction = LocalRuleInteraction(
        ...         'rule', target='cells', state='alive', rule=rule, neighborhood=neighborhood
        ...     )
        ...     return System(elements=[cells], interactions=[interaction])
        >>> shift = build(8, TableSpec({(1,): 1}), LatticeNeighborhood(((1,),)))
        >>> automaton(shift).injective
        True
        >>> majority = build(5, TotalisticSpec(lambda total: total >= 2), BoxNeighborhood())
        >>> automaton(majority, kind='garden-of-eden').pattern.tolist()
        [0, 0, 1, 0, 1]
        >>> class GoldenCellElement(Element):
        ...     def __init__(
        ...         self,
        ...         label,
        ...         alive: Annotated[
        ...             float, StateSpec(domain=SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),)))
        ...         ] = 0,
        ...     ):
        ...         super().__init__(label)
        ...         self.alive = alive
        >>> window = LatticeSpace(6, boundary='fixed')
        >>> golden = System(elements=[LatticeElement('cells', element=GoldenCellElement, space=window)])
        >>> automaton(golden, kind='count').count
        21
        >>> windows = [LatticeSpace(20, boundary='fixed')]
        >>> automaton(golden, kind='entropy', windows=windows).entropy.round(3).tolist()
        [0.706]
    """
    source = _composite(source)
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
    if not isinstance(source, System):
        raise AnalysisError(error_msg)
    found = _find_subshift(source)
    rule = _find_rule(source)
    if found is not None and SUBSHIFT in sources:
        return _run_subshift(
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
    if rule is None or _get_rule_source(rule) not in sources:
        raise AnalysisError(error_msg)
    read = _read_automaton(source, rule)
    if kind in {'injective', 'surjective', 'pre-injective'}:
        return _decide(read, kind, method=method, window=window, infinite=infinite)
    if kind in {'compose', 'difference'}:
        return _run_pair(read, kind, second=second, configuration=configuration, site=site, steps=steps)
    return _measure(
        read,
        kind,
        window=window,
        tries=tries,
        random_state=random_state,
        max_radius=max_radius,
        windows=windows,
        base=base,
    )


def _decide(
    automaton: _AutomatonSpec,
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
    automaton: _AutomatonSpec,
    kind: str,
    *,
    second: System | None,
    configuration: npt.ArrayLike | None,
    site: int,
    steps: int,
) -> AutomatonResult:
    """Return the composition with the second automaton, or a difference pattern.

    The difference pattern is that of a changed configuration.
    """
    if kind == 'compose':
        if second is None:
            error_msg = 'automaton of kind compose needs second. Give it.'
            raise AnalysisError(error_msg)
        return AutomatonResult(automaton.system, composition=_compose(automaton, second))
    if configuration is None:
        error_msg = 'automaton of kind difference needs configuration. Give it.'
        raise AnalysisError(error_msg)
    return AutomatonResult(automaton.system, differences=_difference_pattern(automaton, configuration, site, steps))


def _measure(
    automaton: _AutomatonSpec,
    kind: str,
    *,
    window: Sequence[int] | None,
    tries: int,
    random_state: int | np.random.Generator | None,
    max_radius: int,
    windows: Sequence[BaseSpace] | None,
    base: float,
) -> AutomatonResult:
    """Return a Garden of Eden pattern, inverse, matrix, polynomial, or entropy.

    They are those of the automaton.
    """
    system = automaton.system
    if kind == 'garden-of-eden':
        return AutomatonResult(
            system,
            pattern=_garden_of_eden(automaton, window, tries=tries, random_state=random_state),
        )
    if kind == 'inverse':
        return AutomatonResult(system, inverse=_inverse(automaton, max_radius))
    if kind == 'matrix':
        return AutomatonResult(system, matrix=_get_matrix(automaton))
    if kind == 'polynomial':
        return AutomatonResult(system, polynomial=_get_polynomial(automaton))
    entropies = [_get_window_entropy(automaton, item) for item in windows or ()]
    return AutomatonResult(system, entropy=np.asarray(entropies) / np.log(base))


def _run_subshift(
    system: System,
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


def _get_space(system: System, element: GroupElement) -> BaseSpace:
    """Return the prepared space of a group of a system.

    It is its own space or the space of the system.
    """
    space = element.space if element.space is not None else system.space
    if space is None:
        error_msg = f'The group {element.label!r} of the system lives on no space. Give it a lattice or a group.'
        raise AnalysisError(error_msg)
    return _fitted(space)


def _find_rule(system: System) -> LocalRuleInteraction | None:
    """Return the local rule of a system.

    The result is `None` without one, and an error is raised for several.
    """
    rules = [item for item in system.interactions if isinstance(item, LocalRuleInteraction)]
    if len(rules) > 1:
        names = ', '.join(rule.label for rule in rules)
        error_msg = f'An automaton has one local rule, but the system has {names}. Give it one.'
        raise AnalysisError(error_msg)
    return rules[0] if rules else None


def _find_subshift(system: System) -> tuple[GroupElement, SubshiftSpec] | None:
    """Return the first group of a system with a state whose domain is a subshift.

    It comes with its spec, or is `None`.
    """
    for element in system.elements:
        if not isinstance(element, GroupElement):
            continue
        for spec in element.get_states().values():
            if isinstance(spec.domain, SubshiftSpec):
                spec.domain._check()
                return element, spec.domain
    return None


def _get_rule_source(rule: LocalRuleInteraction | None) -> str | None:
    """Return what a local rule makes of its system.

    It is a linear automaton or an automaton, or `None` without one.
    """
    if rule is None:
        return None
    return LINEAR if isinstance(rule.rule, LinearRuleSpec) else AUTOMATON


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


def _read_automaton(system: System, rule: LocalRuleInteraction) -> _AutomatonSpec:
    """Return the law, space, size, number of states, and local map of an automaton.

    The automaton is that of a system.
    """
    target = getattr(rule, rule._get_role('target'))
    element = next((item for item in system.elements if item.label == target), None)
    if not isinstance(element, GroupElement) or not isinstance(rule.state, str) or rule.reads is not None:
        error_msg = (
            f'The local rule {rule.label!r} of an automaton sets and reads one state of a group of the system, '
            f'but it binds {target!r} and the states {rule.state!r} and {rule.reads!r}.'
        )
        raise AnalysisError(error_msg)
    if rule.schedule != SYNCHRONOUS or rule.stochastic:
        error_msg = (
            f'An automaton maps configurations to configurations, so the local rule {rule.label!r} is synchronous '
            f'and deterministic, but it has the schedule {rule.schedule!r} and stochastic={rule.stochastic!r}.'
        )
        raise AnalysisError(error_msg)
    namespace = SimpleNamespace(**{rule._get_role('target'): SimpleNamespace(**{rule.state: SR.var(rule.state)})})
    law = rule.prepare().run(namespace).laws_[0]
    if not isinstance(law, LocalRuleLaw):
        error_msg = f'The local rule {rule.label!r} returns {law!r}, not a local rule law.'
        raise AnalysisError(error_msg)
    law = law.prepare()
    space = _get_space(system, element)
    states = _get_states(element.get_states()[rule.state].domain, law._get_rule())
    read = _AutomatonSpec(system, rule, law, rule.state, space, int(space.size_ or 0), states)
    return replace(read, local_map=_get_local_map(read))


def _get_local_map(automaton: _AutomatonSpec) -> IntArray:
    """Return the next value of every pattern of the cells the rule reads.

    The values are indexed by the code of the pattern.
    """
    law = automaton.law
    if isinstance(law._get_rule(), LinearRuleSpec):
        return np.empty(0, dtype=np.int64)
    if automaton.states is None:
        error_msg = (
            f'A rule {law.rule!r} on a state without a finite domain has no number of values. Declare the state with '
            '`StateSpec(domain=FiniteEnumeratedSet([0, 1]))`.'
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


def _get_table(automaton: _AutomatonSpec, space: BaseSpace) -> Indices:
    """Return the site of every cell the rule reads at every site of a space.

    The table includes the sites outside it.
    """
    return _fitted(space)._index_table(automaton.law._get_neighborhood())


def _step(automaton: _AutomatonSpec, configuration: npt.ArrayLike, space: BaseSpace | None = None) -> IntArray:
    """Return the image of a configuration by one run of the law.

    It also takes one configuration per row.
    """
    inputs = {'space': automaton.space if space is None else space, automaton.state: np.asarray(configuration)}
    return np.asarray(automaton.law.run(inputs).value_, dtype=np.int64)


def _get_steps(automaton: _AutomatonSpec, configuration: npt.ArrayLike, steps: int) -> IntArray:
    """Return the configurations of a number of steps, the initial one first."""
    rows = [np.asarray(configuration, dtype=np.int64)]
    for _ in range(steps):
        rows.append(_step(automaton, rows[-1]))
    return np.stack(rows)


def _get_states_count(automaton: _AutomatonSpec) -> int:
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


def _shifted(element: Item, base: Item) -> Item:
    """Return the composite of an element of a memory set after an element of another.

    The composite is a word or an offset.
    """
    if isinstance(element, str) and isinstance(base, str):
        parts = [part for part in (base, element) if part.strip() not in {'', IDENTITY}]
        return '*'.join(parts) or IDENTITY
    if isinstance(element, tuple) and isinstance(base, tuple):
        return tuple(first + second for first, second in zip(base, element, strict=True))
    error_msg = f'Composed automata read words of a memory set or offsets of a lattice, not {element!r} after {base!r}.'
    raise AnalysisError(error_msg)


def _rule_clauses(
    automaton: _AutomatonSpec,
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


def _collision(automaton: _AutomatonSpec, window: Sequence[int]) -> tuple[IntArray, IntArray] | None:
    """Return two distinct configurations with the same image, by SAT.

    They differ inside a window and agree outside it.
    """
    clauses = _Clauses()
    states = _get_states_count(automaton)
    first = clauses.one_hot(automaton.size, states)
    second = clauses.one_hot(automaton.size, states)
    image = clauses.one_hot(automaton.size, states, exactly=False)
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
            differ = clauses.fresh()
            clauses.clauses += [[-differ, same[0]], [-differ, -same[1]]]
            differences.append(differ)
    clauses.clauses.append(differences)
    system = clauses.solve()
    return None if system is None else (_decode(system, first), _decode(system, second))


def _preimage(automaton: _AutomatonSpec, target: IntArray, window: Sequence[int]) -> IntArray | None:
    """Return a configuration whose image equals a target on a window, by SAT.

    The result is `None` when there is none.
    """
    clauses = _Clauses()
    cells = clauses.one_hot(automaton.size, _get_states_count(automaton))

    def forbid(site: int, value: int) -> list[int] | None:
        """Return an empty consequence when the rule misses the target.

        The empty consequence forbids the pattern.
        """
        return None if value == target[site] else []

    _rule_clauses(automaton, clauses, cells, window, forbid)
    system = clauses.solve()
    return None if system is None else _decode(system, cells)


def _images(automaton: _AutomatonSpec) -> tuple[IntArray, IntArray]:
    """Return every configuration and the code of its image."""
    states = _get_states_count(automaton)
    configurations = _configurations(states, automaton.size)
    return configurations, _codes(_step(automaton, configurations), states)


def _ball_neighborhood(automaton: _AutomatonSpec, radius: int) -> Reach:
    """Return the neighbourhood of every cell within a radius.

    It is a box of offsets or a ball of group elements.
    """
    if isinstance(automaton.space, LatticeSpace) and not isinstance(automaton.law.over, LatticeNeighborhood):
        error_msg = 'An inverse is searched on a lattice with a neighbourhood of offsets or on a group space.'
        raise AnalysisError(error_msg)
    ball = automaton.space._ball(radius)
    if not isinstance(ball, LatticeNeighborhood | MemoryNeighborhood):
        error_msg = f'The space of the automaton gives a ball of {type(ball).__name__}, which an automaton cannot read.'
        raise AnalysisError(error_msg)
    return ball


def _is_enumerable(automaton: _AutomatonSpec) -> bool:
    """Check whether the configurations of the automaton are few enough to enumerate."""
    return _get_states_count(automaton) ** automaton.size <= MAX_CONFIGURATIONS


def _is_injective(automaton: _AutomatonSpec, *, method: str = 'auto') -> bool:
    """Check whether two different configurations never have the same image."""
    if _check_method(method, _is_enumerable(automaton)):
        _, codes = _images(automaton)
        return bool(np.unique(codes).size == codes.size)
    return _collision(automaton, range(automaton.size)) is None


def _is_pre_injective(automaton: _AutomatonSpec, window: Sequence[int] | None = None, *, method: str = 'auto') -> bool:
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
    automaton: _AutomatonSpec,
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


def _get_system(automaton: _AutomatonSpec, neighborhood: Reach, rule: object, name: str) -> System:
    """Return the system with the elements of the analysed one and a new local rule.

    The rule acts on the same state.
    """
    system = automaton.system
    source = automaton.rule
    interaction = LocalRuleInteraction(
        f'{source.label}_{name}',
        target=getattr(source, source._get_role('target')),
        state=automaton.state,
        rule=rule,
        neighborhood=neighborhood,
    )
    return System(system.label, elements=system.elements, interactions=[interaction], space=system.space)


def _inverse(automaton: _AutomatonSpec, max_radius: int = 2) -> System | None:
    """Return the system of the inverse of a reversible automaton.

    Its composition with the automaton is the identity, and it is `None` when there is none.
    """
    configurations, codes = _images(automaton)
    if np.unique(codes).size != codes.size:
        return None
    states = _get_states_count(automaton)
    images = _step(automaton, configurations)
    for radius in range(max_radius + 1):
        ball = _ball_neighborhood(automaton, radius)
        table = automaton.space._index_table(ball)
        padded = np.concatenate([images, np.zeros((images.shape[0], 1), dtype=np.int64)], axis=1)
        patterns = _codes(padded[:, table], states).ravel()
        values = configurations.ravel()
        pairs = np.unique(np.column_stack([patterns, values]), axis=0)
        if np.unique(pairs[:, 0]).size != pairs.shape[0]:
            continue
        rows = _patterns(states, table.shape[1])
        mapping = {tuple(int(item) for item in rows[code]): int(value) for code, value in pairs}
        return _get_system(automaton, ball, TableSpec(mapping, states=states), 'inverse')
    return None


def _get_reach(automaton: _AutomatonSpec) -> Reach:
    """Return the neighbourhood the rule of the automaton reads.

    It is made of offsets or of group elements.
    """
    reach = automaton.law._get_neighborhood()
    if not isinstance(reach, LatticeNeighborhood | MemoryNeighborhood):
        error_msg = f'An automaton reads offsets or a memory set, not {reach!r}.'
        raise AnalysisError(error_msg)
    return reach


def _compose(automaton: _AutomatonSpec, second: System) -> System:
    """Return the system of the automaton that applies this one then a second one.

    Both act on the same space.
    """
    rule = _find_rule(second)
    if rule is None or isinstance(rule.rule, LinearRuleSpec):
        error_msg = 'Composed automata are systems with a local rule that is not linear.'
        raise AnalysisError(error_msg)
    other = _read_automaton(second, rule)
    if automaton.space is not other.space or automaton.states != other.states:
        error_msg = 'Composed automata share their space and their number of states.'
        raise AnalysisError(error_msg)
    inner, outer = _get_reach(automaton), _get_reach(other)
    inner_items = inner.elements if isinstance(inner, MemoryNeighborhood) else inner.prepare().offsets_
    outer_items = outer.elements if isinstance(outer, MemoryNeighborhood) else outer.prepare().offsets_
    composite: list[Item] = []
    columns = []
    for base in outer_items:
        row = []
        for element in inner_items:
            item = _shifted(element, base)
            if item not in composite:
                composite.append(item)
            row.append(composite.index(item))
        columns.append(tuple(row))
    reach: Reach = (
        MemoryNeighborhood(tuple(item for item in composite if isinstance(item, str)))
        if isinstance(inner, MemoryNeighborhood)
        else LatticeNeighborhood(tuple(item for item in composite if isinstance(item, tuple)))
    )
    composed = _ComposedRuleSpec(automaton.local_map, other.local_map, tuple(columns), _get_states_count(automaton))
    return _get_system(automaton, reach, functools.partial(_composed_rule, composed), 'composition')


def _composed_rule(composed: _ComposedRuleSpec, *values: npt.ArrayLike) -> IntArray:
    """Return the value of the second rule of a composition on the first rule's values.

    The values of the first rule are those around every cell.
    """
    cells = np.stack([np.asarray(value, dtype=np.int64) for value in values], axis=-1)
    inner = [composed.first[_codes(cells[..., list(columns)], composed.states)] for columns in composed.columns]
    return composed.second[_codes(np.stack(inner, axis=-1), composed.states)]


def _difference_pattern(
    automaton: _AutomatonSpec,
    configuration: npt.ArrayLike,
    site: int,
    steps: int,
) -> npt.NDArray[np.bool_]:
    """Return where two runs differ when one starts with one cell changed.

    The cell is changed to its next value.
    """
    initial = np.asarray(configuration, dtype=np.int64)
    changed = initial.copy()
    changed[site] = (changed[site] + 1) % _get_states_count(automaton)
    return _get_steps(automaton, initial, steps) != _get_steps(automaton, changed, steps)


def _get_window_entropy(automaton: _AutomatonSpec, window: BaseSpace) -> float:
    """Return the entropy `log |X_F| / |F|` of the image of the automaton on a window."""
    states = _get_states_count(automaton)
    interior = np.flatnonzero((_get_table(automaton, window) < _size_of(window)).all(axis=1))
    images = _step(automaton, _configurations(states, _size_of(window)), window)[:, interior]
    count = np.unique(_codes(images, states)).size
    return float(np.log(count) / max(interior.size, 1))


def _get_linear_rule(automaton: _AutomatonSpec) -> LinearRuleSpec:
    """Return the linear rule of the automaton, or raise an error for any other rule."""
    rule = automaton.law._get_rule()
    if not isinstance(rule, LinearRuleSpec):
        error_msg = f'The kind reads a linear automaton, not {rule!r}.'
        raise AnalysisError(error_msg)
    return rule


def _get_matrix(automaton: _AutomatonSpec) -> Matrix:
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


def _get_polynomial(automaton: _AutomatonSpec) -> LaurentPolynomial:
    """Return the Laurent polynomial `sum a_s x^s` of a linear automaton on a lattice.

    It is its symbol on `Z^d`.
    """
    rule = _get_linear_rule(automaton)
    if not isinstance(automaton.law.over, LatticeNeighborhood):
        error_msg = 'A Laurent polynomial describes a linear automaton on a lattice, with offsets.'
        raise AnalysisError(error_msg)
    field = rule.sage_field
    offsets = automaton.law.over.prepare().offsets_
    ring = LaurentPolynomialRing(field, [f'x{axis}' for axis in range(len(offsets[0]))])
    terms = (
        field(QQ(coefficient)) * np.prod([variable**step for variable, step in zip(ring.gens(), offset, strict=True)])
        for offset, coefficient in zip(offsets, rule.coefficients, strict=True)
    )
    return sum(terms, ring.zero())


def _linear_injective(automaton: _AutomatonSpec, *, infinite: bool) -> bool:
    """Check whether a linear automaton is invertible.

    It is when its matrix on its universe is, or when its symbol on `Z^d` is a monomial.
    """
    if infinite:
        return len(_get_polynomial(automaton).exponents()) == 1
    return bool(_get_matrix(automaton).is_invertible())


def _linear_surjective(automaton: _AutomatonSpec, *, infinite: bool) -> bool:
    """Check whether a linear automaton is surjective.

    It is when it is invertible on its universe, or when its symbol on `Z^d` is not zero.
    """
    if infinite:
        return not _get_polynomial(automaton).is_zero()
    return _linear_injective(automaton, infinite=False)


def _get_complete(subshift: SubshiftSpec, space: BaseSpace) -> tuple[Indices, npt.NDArray[np.bool_]]:
    """Return the neighbour table of a space and the sites whose patterns lie in it."""
    neighborhood = subshift.neighborhood
    if not isinstance(neighborhood, LatticeNeighborhood | MemoryNeighborhood):
        error_msg = f'A subshift covers offsets or a memory set, not {neighborhood!r}.'
        raise AnalysisError(error_msg)
    table = _fitted(space)._index_table(neighborhood)
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
