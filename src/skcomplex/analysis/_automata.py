"""Decide injectivity and surjectivity of cellular automata and subshifts, and measure their entropy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import itertools
from collections.abc import Callable, Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import QQ, SR, LaurentPolynomialRing, matrix
from sage.rings.polynomial.laurent_polynomial import LaurentPolynomial
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.structure.element import Matrix

from ..core.elements import GroupElement
from ..core.interactions import LocalRuleInteraction
from ..core.laws import LocalRuleLaw
from ..core.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace, MemoryNeighborhood
from ..core.system import System
from ..core.types import Reach
from ..errors import AnalysisError
from ..specs import LinearRuleSpec, OuterTotalisticSpec, SubshiftSpec, TableSpec, TotalisticSpec
from ..specs._composed_rule_spec import _ComposedRuleSpec
from ..types import DataKind, Indices, IntArray
from ._base import BaseAnalysis
from .types import Item, Source

MAX_CONFIGURATIONS = 2**16
MAX_PATTERNS = 2**20
SAT_SOLVER = 'cadical153'
METHODS = ('auto', 'enumeration', 'sat')
SYNCHRONOUS = 'synchronous'
IDENTITY = '1'


class _Clauses:
    """Clauses of a SAT problem over one-hot variables of site values."""

    def __init__(self) -> None:
        self.clauses: list[list[int]] = []
        self.count = 0

    def one_hot(self, sites: int, states: int, *, exactly: bool = True) -> npt.NDArray[np.int64]:
        """Return one variable per site and value, with exactly one, or at most one, true per site."""
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
        """Return a model of the clauses, or `None` when they are unsatisfiable."""
        solvers = BaseAnalysis._import('pysat.solvers')
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


class AutomatonAnalysis(BaseAnalysis):
    """Decisions and measures of a system that is a cellular automaton on a finite universe or holds a subshift.

    The analysis prepares on a `System`. An automaton is a group of cells with one synchronous and deterministic
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

    Attributes:
        automaton_:
            The system.

        injective_:
            Whether the automaton is injective.

        surjective_:
            Whether the automaton is surjective.

        pre_injective_:
            Whether no two configurations that differ only inside the window share their image.

        pattern_:
            A configuration whose values on the window have no preimage, or `None` when none is found.

        inverse_:
            The system of the inverse automaton with a table rule, or `None` when the automaton is not injective or no
            memory up to the radius determines the cells.

        composition_:
            The system of the automaton of the composition, with a function rule.

        differences_:
            One row per step, the initial one first, true where the runs differ.

        entropy_:
            One entropy per window.

        matrix_:
            The Sage matrix of the linear automaton.

        polynomial_:
            The Sage Laurent polynomial in the variables `x0` to `x(d-1)`.

        admissible_:
            Whether the configuration belongs to the subshift.

        violations_:
            One flag per site, true where a forbidden pattern appears.

        count_:
            The number of configurations without a forbidden pattern.

        sample_:
            The admissible configuration.

    Raises:
        AnalysisError:
            The system holds nothing the kind reads, the method is unknown, enumeration is asked for too many
            configurations, composed automata differ in space or number of states, the rule returns a value outside the
            states, a linear automaton has a matrix on a ball or a polynomial without offsets, or the initial
            configuration of a sample is not admissible.

    Examples:
        >>> from typing import Annotated
        >>> from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
        >>> from skcomplex.analysis import AutomatonAnalysis
        >>> from skcomplex.core.elements import Element, LatticeElement
        >>> from skcomplex.core.interactions import LocalRuleInteraction
        >>> from skcomplex.core.spaces import BoxNeighborhood, LatticeNeighborhood, LatticeSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.specs import StateSpec, SubshiftSpec, TableSpec, TotalisticSpec
        >>> class CellElement(Element):
        ...     def __init__(self, label, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        ...         super().__init__(label)
        ...         self.alive = alive
        >>> def automaton(size, rule, neighborhood):
        ...     cells = LatticeElement('cells', element=CellElement, space=LatticeSpace(size))
        ...     interaction = LocalRuleInteraction(
        ...         'rule', target='cells', state='alive', rule=rule, neighborhood=neighborhood
        ...     )
        ...     return System(elements=[cells], interactions=[interaction])
        >>> shift = automaton(8, TableSpec({(1,): 1}), LatticeNeighborhood(((1,),)))
        >>> AutomatonAnalysis().check().prepare(shift).run().injective_
        True
        >>> majority = automaton(5, TotalisticSpec(lambda total: total >= 2), BoxNeighborhood())
        >>> AutomatonAnalysis('garden-of-eden').check().prepare(majority).run().pattern_.tolist()
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
        >>> AutomatonAnalysis('count').check().prepare(golden).run().count_
        21
        >>> analysis = AutomatonAnalysis('entropy', windows=[LatticeSpace(20, boundary='fixed')])
        >>> analysis.check().prepare(golden).run().entropy_.round(3).tolist()
        [0.706]
    """

    _subshift: SubshiftSpec | None

    def __init__(
        self,
        kind: str = 'injective',
        *,
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
    ) -> None:
        self.kind = kind
        self.method = method
        self.window = window
        self.infinite = infinite
        self.tries = tries
        self.random_state = random_state
        self.max_radius = max_radius
        self.second = second
        self.configuration = configuration
        self.site = site
        self.steps = steps
        self.windows = windows
        self.base = base
        self.sweeps = sweeps

    def _check(self) -> None:
        """Validate the kind, the method, that the kind reads every changed setting, and that it has what it needs."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind])
        if self.method not in METHODS:
            error_msg = f'{type(self).__name__} has method={self.method!r}. Use one of {", ".join(METHODS)}.'
            raise AnalysisError(error_msg)
        needed = {
            'compose': 'second',
            'difference': 'configuration',
            'entropy': 'windows',
            'admissible': 'configuration',
            'violations': 'configuration',
        }.get(self.kind)
        if needed is not None and getattr(self, needed) is None:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs {needed}. Give it.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the system kind."""
        return DataKind.SYSTEM

    @staticmethod
    def _get_space(system: System, element: GroupElement) -> BaseSpace:
        """Return the prepared space of a group of a system, its own or the space of the system."""
        space = element.space if element.space is not None else system.space
        if space is None:
            error_msg = f'The group {element.label!r} of the system lives on no space. Give it a lattice or a group.'
            raise AnalysisError(error_msg)
        return AutomatonAnalysis._fitted(space)

    @staticmethod
    def _find_rule(system: System) -> LocalRuleInteraction | None:
        """Return the local rule of a system, `None` without one, or raise an error for several."""
        rules = [item for item in system.interactions if isinstance(item, LocalRuleInteraction)]
        if len(rules) > 1:
            names = ', '.join(rule.label for rule in rules)
            error_msg = f'An automaton has one local rule, but the system has {names}. Give it one.'
            raise AnalysisError(error_msg)
        return rules[0] if rules else None

    @staticmethod
    def _find_subshift(system: System) -> tuple[GroupElement, SubshiftSpec] | None:
        """Return the first group of a system with a state whose domain is a subshift, with its spec, or `None`."""
        for element in system.elements:
            if not isinstance(element, GroupElement):
                continue
            for spec in element.get_states().values():
                if isinstance(spec.domain, SubshiftSpec):
                    return element, spec.domain.check()
        return None

    @staticmethod
    def _get_rule_source(rule: LocalRuleInteraction | None) -> str | None:
        """Return what a local rule makes of its system, a linear automaton or an automaton, or `None` without one."""
        if rule is None:
            return None
        return LINEAR if isinstance(rule.rule, LinearRuleSpec) else AUTOMATON

    def _prepare(self, source: Source) -> None:
        """Keep the system, after checking that it holds what the kind reads, and read that."""
        sources = SOURCES[self.kind]
        error_msg = (
            f'{type(self).__name__} of kind {self.kind!r} reads a system with {" or ".join(sources)}, not '
            f'{type(source).__name__}. Build one of them.'
        )
        if not isinstance(source, System):
            raise AnalysisError(error_msg)
        subshift = self._find_subshift(source)
        rule = self._find_rule(source)
        self._subshift = None
        if subshift is not None and SUBSHIFT in sources:
            self._subshift = subshift[1]
            self._space = self._get_space(source, subshift[0])
            self._size = int(self._space.size_ or 0)
        elif rule is not None and self._get_rule_source(rule) in sources:
            self._read_automaton(source, rule)
        else:
            raise AnalysisError(error_msg)
        self.automaton_ = source

    @staticmethod
    def _get_states(domain: object, rule: object) -> int | None:
        """Return the number of values of a cell, from a finite domain of a state or else from a rule, or `None`."""
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

    def _read_automaton(self, system: System, rule: LocalRuleInteraction) -> None:
        """Keep the law, the space, the size, the number of states, and the local map of the automaton of a system."""
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
        law = rule.check().prepare().run(namespace).laws_[0]
        if not isinstance(law, LocalRuleLaw):
            error_msg = f'The local rule {rule.label!r} returns {law!r}, not a local rule law.'
            raise AnalysisError(error_msg)
        self._rule = rule
        self._law = law.check().prepare()
        self._state = rule.state
        self._space = self._get_space(system, element)
        self._size = int(self._space.size_ or 0)
        self._states = self._get_states(element.get_states()[rule.state].domain, self._law._get_rule())
        self._local_map = self._get_local_map()

    def _get_local_map(self) -> IntArray:
        """Return the next value of every pattern of the cells the rule reads, indexed by the code of the pattern."""
        if isinstance(self._law._get_rule(), LinearRuleSpec):
            return np.empty(0, dtype=np.int64)
        if self._states is None:
            error_msg = (
                f'A rule {self._law.rule!r} on a state without a finite domain has no number of values. Declare the '
                'state with `StateSpec(domain=FiniteEnumeratedSet([0, 1]))`.'
            )
            raise AnalysisError(error_msg)
        width = self._get_table(self._space).shape[1]
        patterns = self._patterns(self._states, width)
        try:
            values = np.asarray(self._law._get_next_values(patterns), dtype=float)
        except TypeError as error:
            error_msg = f'An automaton has a rule of the cells alone, but {self._law.rule!r} needs more: {error}'
            raise AnalysisError(error_msg) from error
        values = np.broadcast_to(values, (patterns.shape[0],))
        if values.size and (values.min() < 0 or values.max() >= self._states or not np.all(values == np.rint(values))):
            error_msg = f'The rule returns values outside the integers 0 to {self._states - 1}.'
            raise AnalysisError(error_msg)
        return values.astype(np.int64)

    def _get_table(self, space: BaseSpace) -> Indices:
        """Return the site of every cell the rule reads at every site of a space, with the sites outside it."""
        return self._fitted(space)._index_table(self._law._get_neighborhood())

    def _step(self, configuration: npt.ArrayLike, space: BaseSpace | None = None) -> IntArray:
        """Return the image of a configuration, or of one configuration per row, by one run of the law."""
        inputs = {'space': self._space if space is None else space, self._state: np.asarray(configuration)}
        return np.asarray(self._law.run(inputs).value_, dtype=np.int64)

    def _get_steps(self, configuration: npt.ArrayLike, steps: int) -> IntArray:
        """Return the configurations of a number of steps, the initial one first."""
        rows = [np.asarray(configuration, dtype=np.int64)]
        for _ in range(steps):
            rows.append(self._step(rows[-1]))
        return np.stack(rows)

    def _get_states_count(self) -> int:
        """Return the number of values of a cell of an automaton of a rule."""
        return int(self._states or 0)

    def _decide(self) -> None:
        """Set whether the automaton is injective, surjective, or pre-injective."""
        if isinstance(self._law._get_rule(), LinearRuleSpec):
            decision = (
                self._linear_injective(infinite=self.infinite)
                if self.kind == 'injective'
                else self._linear_surjective(infinite=self.infinite)
            )
        elif self.kind == 'pre-injective':
            decision = self._is_pre_injective(self.window, method=self.method)
        else:
            decision = self._is_injective(method=self.method)
        setattr(self, f'{self.kind.replace("-", "_")}_', decision)

    def _run(self, inputs: object = None) -> None:
        """Decide or measure the automaton or the subshift."""
        if self.kind in {'injective', 'surjective', 'pre-injective'}:
            self._decide()
        elif self.kind == 'garden-of-eden':
            self.pattern_ = self._garden_of_eden(self.window, tries=self.tries, random_state=self.random_state)
        elif self.kind == 'inverse':
            self.inverse_ = self._inverse(self.max_radius)
        elif self.kind in {'compose', 'difference'}:
            self._run_pair()
        elif self.kind in {'matrix', 'polynomial'}:
            self._run_linear()
        elif self.kind in {'admissible', 'violations', 'count', 'sample'}:
            self._run_subshift()
        else:
            self.entropy_ = self._configuration_entropy(self.windows or (), base=self.base)

    def _run_pair(self) -> None:
        """Set the composition with the second automaton, or the difference pattern of a changed configuration."""
        if self.kind == 'compose':
            if self.second is None:
                error_msg = f'{type(self).__name__} of kind compose needs second. Give it.'
                raise AnalysisError(error_msg)
            self.composition_ = self._compose(self.second)
            return
        if self.configuration is None:
            error_msg = f'{type(self).__name__} of kind difference needs configuration. Give it.'
            raise AnalysisError(error_msg)
        self.differences_ = self._difference_pattern(self.configuration, self.site, self.steps)

    def _run_linear(self) -> None:
        """Set the matrix or the Laurent polynomial of a linear automaton."""
        if self.kind == 'matrix':
            self.matrix_ = self._get_matrix()
        else:
            self.polynomial_ = self._get_polynomial()

    def _run_subshift(self) -> None:
        """Set whether a configuration is admissible, its violations, the count, or a sample of a subshift."""
        subshift = self._get_subshift()
        if self.kind == 'count':
            self.count_ = self._count_configurations(subshift, self._space)
            return
        if self.kind == 'sample':
            self.sample_ = self._sample(subshift)
            return
        if self.configuration is None:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs configuration. Give it.'
            raise AnalysisError(error_msg)
        if self.kind == 'violations':
            self.violations_ = self._get_violations(subshift, self.configuration)
        else:
            self.admissible_ = not self._get_violations(subshift, self.configuration).any()

    def _get_subshift(self) -> SubshiftSpec:
        """Return the subshift of the system, or raise an error for a system read as an automaton."""
        if self._subshift is None:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} reads a subshift.'
            raise AnalysisError(error_msg)
        return self._subshift

    @staticmethod
    def _fitted(space: BaseSpace) -> BaseSpace:
        """Return a space of sites, preparing it on its first use."""
        if not hasattr(space, 'size_'):
            space.check().prepare()
        return space

    @staticmethod
    def _size_of(space: BaseSpace) -> int:
        """Return the number of sites of a space."""
        return int(AutomatonAnalysis._fitted(space).size_ or 0)

    @staticmethod
    def _patterns(states: int, width: int) -> IntArray:
        """Return every pattern of values of a number of cells, in the order of their codes."""
        if states**width > MAX_PATTERNS:
            error_msg = f'A rule on {width} cells with {states} values has more than {MAX_PATTERNS} patterns.'
            raise AnalysisError(error_msg)
        return np.asarray(list(itertools.product(range(states), repeat=width)), dtype=np.int64).reshape(-1, width)

    @staticmethod
    def _codes(values: IntArray, states: int) -> IntArray:
        """Return the code of every row of values, read as digits in the base of the number of states."""
        weights = states ** np.arange(values.shape[-1] - 1, -1, -1, dtype=np.int64)
        return values @ weights

    @staticmethod
    def _configurations(states: int, size: int) -> IntArray:
        """Return every configuration of a number of sites, or raise an error when there are too many."""
        if states**size > MAX_CONFIGURATIONS:
            error_msg = (
                f'{states}^{size} configurations exceed the enumeration limit of {MAX_CONFIGURATIONS}. Use the SAT '
                'method, or a smaller universe.'
            )
            raise AnalysisError(error_msg)
        return AutomatonAnalysis._patterns(states, size)

    @staticmethod
    def _check_method(method: str, enumerable: bool) -> bool:
        """Return whether to enumerate configurations, after checking the method."""
        if method not in METHODS:
            error_msg = f'Unknown method {method!r}. Use one of {", ".join(METHODS)}.'
            raise AnalysisError(error_msg)
        return method == 'enumeration' or (method == 'auto' and enumerable)

    @staticmethod
    def _shifted(element: Item, base: Item) -> Item:
        """Return the composite of an element of a memory set after an element of another, as a word or an offset."""
        if isinstance(element, str) and isinstance(base, str):
            parts = [part for part in (base, element) if part.strip() not in {'', IDENTITY}]
            return '*'.join(parts) or IDENTITY
        if isinstance(element, tuple) and isinstance(base, tuple):
            return tuple(first + second for first, second in zip(base, element, strict=True))
        error_msg = (
            f'Composed automata read words of a memory set or offsets of a lattice, not {element!r} after {base!r}.'
        )
        raise AnalysisError(error_msg)

    def _rule_clauses(
        self,
        clauses: _Clauses,
        cells: npt.NDArray[np.int64],
        sites: Sequence[int],
        output: Callable[[int, int], list[int] | None],
    ) -> None:
        """Add, for every site and pattern of its cells, the clause that the pattern implies the output of the rule."""
        table = self._get_table(self._space)
        patterns = self._patterns(self._get_states_count(), table.shape[1])
        fill = int(getattr(self._space, 'fill', 0))
        for site in sites:
            row = table[site]
            for pattern, value in zip(patterns, self._local_map, strict=True):
                outside = row >= self._size
                if np.any(pattern[outside] != fill):
                    continue
                consequence = output(site, int(value))
                if consequence is None:
                    continue
                premise = [
                    -int(cells[neighbour, item])
                    for neighbour, item in zip(row[~outside], pattern[~outside], strict=True)
                ]
                clauses.clauses.append(premise + consequence)

    @staticmethod
    def _decode(model: list[int], variables: npt.NDArray[np.int64]) -> IntArray:
        """Return the values of one-hot variables in a model."""
        true = set(model)
        return np.asarray(
            [next(value for value, item in enumerate(row) if item in true) for row in variables.tolist()],
            dtype=np.int64,
        )

    def _collision(self, window: Sequence[int]) -> tuple[IntArray, IntArray] | None:
        """Return two configurations that differ inside a window, agree outside it, and have the same image, by SAT."""
        clauses = _Clauses()
        states = self._get_states_count()
        first = clauses.one_hot(self._size, states)
        second = clauses.one_hot(self._size, states)
        image = clauses.one_hot(self._size, states, exactly=False)
        for cells in (first, second):
            self._rule_clauses(clauses, cells, range(self._size), lambda site, value: [int(image[site, value])])
        inside = set(window)
        differences = []
        for site in range(self._size):
            for value in range(states):
                same = (int(first[site, value]), int(second[site, value]))
                if site not in inside:
                    clauses.clauses += [[-same[0], same[1]], [same[0], -same[1]]]
                    continue
                differ = clauses.fresh()
                clauses.clauses += [[-differ, same[0]], [-differ, -same[1]]]
                differences.append(differ)
        clauses.clauses.append(differences)
        model = clauses.solve()
        return None if model is None else (self._decode(model, first), self._decode(model, second))

    def _preimage(self, target: IntArray, window: Sequence[int]) -> IntArray | None:
        """Return a configuration whose image equals a target on a window, or `None`, by SAT."""
        clauses = _Clauses()
        cells = clauses.one_hot(self._size, self._get_states_count())

        def forbid(site: int, value: int) -> list[int] | None:
            """Return an empty consequence, which forbids the pattern, when the rule misses the target."""
            return None if value == target[site] else []

        self._rule_clauses(clauses, cells, window, forbid)
        model = clauses.solve()
        return None if model is None else self._decode(model, cells)

    def _images(self) -> tuple[IntArray, IntArray]:
        """Return every configuration and the code of its image."""
        states = self._get_states_count()
        configurations = self._configurations(states, self._size)
        return configurations, self._codes(self._step(configurations), states)

    def _ball_neighborhood(self, radius: int) -> Reach:
        """Return the neighbourhood of every cell within a radius, a box of offsets or a ball of group elements."""
        if isinstance(self._space, LatticeSpace) and not isinstance(self._law.over, LatticeNeighborhood):
            error_msg = 'An inverse is searched on a lattice with a neighbourhood of offsets or on a group space.'
            raise AnalysisError(error_msg)
        ball = self._space._ball(radius)
        if not isinstance(ball, LatticeNeighborhood | MemoryNeighborhood):
            error_msg = (
                f'The space of the automaton gives a ball of {type(ball).__name__}, which an automaton cannot read.'
            )
            raise AnalysisError(error_msg)
        return ball

    def _is_enumerable(self) -> bool:
        """Check whether the configurations of the automaton are few enough to enumerate."""
        return self._get_states_count() ** self._size <= MAX_CONFIGURATIONS

    def _is_injective(self, *, method: str = 'auto') -> bool:
        """Check whether two different configurations never have the same image."""
        if self._check_method(method, self._is_enumerable()):
            _, codes = self._images()
            return bool(np.unique(codes).size == codes.size)
        return self._collision(range(self._size)) is None

    def _is_pre_injective(self, window: Sequence[int] | None = None, *, method: str = 'auto') -> bool:
        """Check whether two configurations that differ only inside a window never have the same image."""
        sites = list(range(self._size)) if window is None else [int(site) for site in window]
        if not self._check_method(method, self._is_enumerable()):
            return self._collision(sites) is None
        states = self._get_states_count()
        configurations, codes = self._images()
        outside = np.setdiff1d(np.arange(self._size), sites)
        keys = codes * states**outside.size + self._codes(configurations[:, outside], states)
        return bool(np.unique(keys).size == keys.size)

    def _garden_of_eden(
        self,
        window: Sequence[int] | None = None,
        *,
        tries: int = 200,
        random_state: int | np.random.Generator | None = None,
    ) -> IntArray | None:
        """Return a pattern on a window that no configuration maps to, or `None`."""
        states = self._get_states_count()
        sites = list(range(self._size)) if window is None else [int(site) for site in window]
        if window is None and self._is_enumerable():
            configurations, codes = self._images()
            missing = np.setdiff1d(np.arange(states**self._size), codes)
            return configurations[missing[0]] if missing.size else None
        rng = np.random.default_rng(random_state)
        for _ in range(tries):
            target = rng.integers(states, size=self._size)
            if self._preimage(target, sites) is None:
                return target
        return None

    def _get_system(self, neighborhood: Reach, rule: object, name: str) -> System:
        """Return the system with the elements of the analysed one and a new local rule on the same state."""
        system = self.automaton_
        source = self._rule
        interaction = LocalRuleInteraction(
            f'{source.label}_{name}',
            target=getattr(source, source._get_role('target')),
            state=self._state,
            rule=rule,
            neighborhood=neighborhood,
        )
        return System(system.label, elements=system.elements, interactions=[interaction], space=system.space)

    def _inverse(self, max_radius: int = 2) -> System | None:
        """Return the system of the automaton whose composition with a reversible one is the identity, or `None`."""
        configurations, codes = self._images()
        if np.unique(codes).size != codes.size:
            return None
        states = self._get_states_count()
        images = self._step(configurations)
        for radius in range(max_radius + 1):
            ball = self._ball_neighborhood(radius)
            table = self._space._index_table(ball)
            padded = np.concatenate([images, np.zeros((images.shape[0], 1), dtype=np.int64)], axis=1)
            patterns = self._codes(padded[:, table], states).ravel()
            values = configurations.ravel()
            pairs = np.unique(np.column_stack([patterns, values]), axis=0)
            if np.unique(pairs[:, 0]).size != pairs.shape[0]:
                continue
            rows = self._patterns(states, table.shape[1])
            mapping = {tuple(int(item) for item in rows[code]): int(value) for code, value in pairs}
            return self._get_system(ball, TableSpec(mapping, states=states), 'inverse')
        return None

    def _get_reach(self) -> Reach:
        """Return the neighbourhood the rule of the automaton reads, of offsets or of group elements."""
        reach = self._law._get_neighborhood()
        if not isinstance(reach, LatticeNeighborhood | MemoryNeighborhood):
            error_msg = f'An automaton reads offsets or a memory set, not {reach!r}.'
            raise AnalysisError(error_msg)
        return reach

    def _compose(self, second: System) -> System:
        """Return the system of the automaton that applies this automaton and then a second one on the same space."""
        rule = self._find_rule(second)
        if rule is None or isinstance(rule.rule, LinearRuleSpec):
            error_msg = 'Composed automata are systems with a local rule that is not linear.'
            raise AnalysisError(error_msg)
        other = type(self)()
        other._read_automaton(second, rule)
        if self._space is not other._space or self._states != other._states:
            error_msg = 'Composed automata share their space and their number of states.'
            raise AnalysisError(error_msg)
        inner, outer = self._get_reach(), other._get_reach()
        inner_items = inner.elements if isinstance(inner, MemoryNeighborhood) else inner.check().prepare().offsets_
        outer_items = outer.elements if isinstance(outer, MemoryNeighborhood) else outer.check().prepare().offsets_
        composite: list[Item] = []
        columns = []
        for base in outer_items:
            row = []
            for element in inner_items:
                item = self._shifted(element, base)
                if item not in composite:
                    composite.append(item)
                row.append(composite.index(item))
            columns.append(tuple(row))
        reach: Reach = (
            MemoryNeighborhood(tuple(item for item in composite if isinstance(item, str)))
            if isinstance(inner, MemoryNeighborhood)
            else LatticeNeighborhood(tuple(item for item in composite if isinstance(item, tuple)))
        )
        composed = _ComposedRuleSpec(self._local_map, other._local_map, tuple(columns), self._get_states_count())
        return self._get_system(reach, functools.partial(self._composed_rule, composed), 'composition')

    @staticmethod
    def _composed_rule(composed: _ComposedRuleSpec, *values: npt.ArrayLike) -> IntArray:
        """Return the value of the second rule of a composition on the values of the first rule around every cell."""
        cells = np.stack([np.asarray(value, dtype=np.int64) for value in values], axis=-1)
        inner = [
            composed.first[AutomatonAnalysis._codes(cells[..., list(columns)], composed.states)]
            for columns in composed.columns
        ]
        return composed.second[AutomatonAnalysis._codes(np.stack(inner, axis=-1), composed.states)]

    def _difference_pattern(self, configuration: npt.ArrayLike, site: int, steps: int) -> npt.NDArray[np.bool_]:
        """Return where two runs differ when one starts with one cell changed to its next value."""
        initial = np.asarray(configuration, dtype=np.int64)
        changed = initial.copy()
        changed[site] = (changed[site] + 1) % self._get_states_count()
        return self._get_steps(initial, steps) != self._get_steps(changed, steps)

    def _get_window_entropy(self, window: BaseSpace) -> float:
        """Return the entropy `log |X_F| / |F|` of the image of the automaton on a window."""
        states = self._get_states_count()
        interior = np.flatnonzero((self._get_table(window) < self._size_of(window)).all(axis=1))
        images = self._step(self._configurations(states, self._size_of(window)), window)[:, interior]
        count = np.unique(self._codes(images, states)).size
        return float(np.log(count) / max(interior.size, 1))

    def _configuration_entropy(self, windows: Sequence[BaseSpace], *, base: float = 2.0) -> npt.NDArray[np.float64]:
        """Return the entropy `log |X_F| / |F|` of a subshift or of the image of an automaton on every window."""
        entropies = [
            (
                self._get_window_entropy(window)
                if self._subshift is None
                else np.log(self._count_configurations(self._subshift, window)) / self._size_of(window)
            )
            for window in windows
        ]
        return np.asarray(entropies) / np.log(base)

    def _get_linear_rule(self) -> LinearRuleSpec:
        """Return the linear rule of the automaton, or raise an error for any other rule."""
        rule = self._law._get_rule()
        if not isinstance(rule, LinearRuleSpec):
            error_msg = f'{type(self).__name__} of kind {self.kind!r} reads a linear automaton, not {rule!r}.'
            raise AnalysisError(error_msg)
        return rule

    def _get_matrix(self) -> Matrix:
        """Return the matrix of a linear automaton over its field, whose row `g` gives the image at `g`."""
        rule = self._get_linear_rule()
        table = self._get_table(self._space)
        if (table >= self._size).any():
            error_msg = 'A linear automaton has a matrix on a periodic lattice or a finite group, not on a ball.'
            raise AnalysisError(error_msg)
        field = rule.sage_field
        entries = matrix(field, self._size, self._size)
        for site, row in enumerate(table.tolist()):
            for neighbour, coefficient in zip(row, rule.coefficients, strict=True):
                entries[site, neighbour] += field(QQ(coefficient))
        return entries

    def _get_polynomial(self) -> LaurentPolynomial:
        """Return the Laurent polynomial `sum a_s x^s` of a linear automaton on a lattice, its symbol on `Z^d`."""
        rule = self._get_linear_rule()
        if not isinstance(self._law.over, LatticeNeighborhood):
            error_msg = 'A Laurent polynomial describes a linear automaton on a lattice, with offsets.'
            raise AnalysisError(error_msg)
        field = rule.sage_field
        offsets = self._law.over.check().prepare().offsets_
        ring = LaurentPolynomialRing(field, [f'x{axis}' for axis in range(len(offsets[0]))])
        terms = (
            field(QQ(coefficient))
            * np.prod([variable**step for variable, step in zip(ring.gens(), offset, strict=True)])
            for offset, coefficient in zip(offsets, rule.coefficients, strict=True)
        )
        return sum(terms, ring.zero())

    def _linear_injective(self, *, infinite: bool) -> bool:
        """Check whether a linear automaton is invertible: its matrix on its universe, or a monomial on `Z^d`."""
        if infinite:
            return len(self._get_polynomial().exponents()) == 1
        return bool(self._get_matrix().is_invertible())

    def _linear_surjective(self, *, infinite: bool) -> bool:
        """Check whether a linear automaton is surjective: invertible on its universe, or not zero on `Z^d`."""
        if infinite:
            return not self._get_polynomial().is_zero()
        return self._linear_injective(infinite=False)

    @staticmethod
    def _get_complete(subshift: SubshiftSpec, space: BaseSpace) -> tuple[Indices, npt.NDArray[np.bool_]]:
        """Return the neighbour table of a space and the sites whose patterns lie inside it."""
        neighborhood = subshift.neighborhood
        if not isinstance(neighborhood, LatticeNeighborhood | MemoryNeighborhood):
            error_msg = f'A subshift covers offsets or a memory set, not {neighborhood!r}.'
            raise AnalysisError(error_msg)
        table = AutomatonAnalysis._fitted(space)._index_table(neighborhood)
        return table, (table < AutomatonAnalysis._size_of(space)).all(axis=1)

    def _get_violations(self, subshift: SubshiftSpec, configuration: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Return the sites of the space at which a forbidden pattern appears in a configuration."""
        table, complete = self._get_complete(subshift, self._space)
        values = np.append(np.asarray(configuration, dtype=np.int64), 0)
        forbidden = np.asarray([tuple(pattern) for pattern in subshift.forbidden], dtype=np.int64)
        codes = self._codes(values[table], subshift.states)
        return complete & np.isin(codes, self._codes(forbidden.reshape(-1, table.shape[1]), subshift.states))

    @staticmethod
    def _count_configurations(subshift: SubshiftSpec, space: BaseSpace) -> int:
        """Return the number of admissible configurations of a space, by dynamic programming over its sites."""
        size = AutomatonAnalysis._size_of(space)
        table, complete = AutomatonAnalysis._get_complete(subshift, space)
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

    def _sample(self, subshift: SubshiftSpec) -> IntArray:
        """Return an admissible configuration drawn by a Markov chain of single-cell moves that keep it admissible."""
        rng = np.random.default_rng(self.random_state)
        values = (
            np.zeros(self._size, dtype=np.int64)
            if self.configuration is None
            else np.asarray(self.configuration, dtype=np.int64).copy()
        )
        if self._get_violations(subshift, values).any():
            error_msg = 'The initial configuration holds a forbidden pattern. Give an admissible one.'
            raise AnalysisError(error_msg)
        table, complete = self._get_complete(subshift, self._space)
        touching: list[list[int]] = [[] for _ in range(self._size)]
        for anchor in np.flatnonzero(complete):
            for cell in set(table[anchor].tolist()):
                touching[cell].append(int(anchor))
        forbidden = {tuple(int(item) for item in pattern) for pattern in subshift.forbidden}
        for _ in range(self.sweeps):
            for site in rng.permutation(self._size):
                previous = values[site]
                values[site] = rng.integers(subshift.states)
                if any(tuple(values[table[anchor]].tolist()) in forbidden for anchor in touching[site]):
                    values[site] = previous
        return values
