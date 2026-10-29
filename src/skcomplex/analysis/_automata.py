"""Decide injectivity and surjectivity of cellular automata and subshifts, and measure their entropy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import itertools
from collections.abc import Callable, Sequence
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt

from ..core.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace, MemoryNeighborhood
from ..core.types import AutomatonProtocol, LinearAutomatonProtocol, SubshiftProtocol
from ..errors import AnalysisError
from ..specs import TableSpec
from ..specs._composed_rule_spec import _ComposedRuleSpec
from ..types import DataKind, IntArray
from ._base import BaseAnalysis, Source

MAX_CONFIGURATIONS = 2**16
MAX_PATTERNS = 2**20
SAT_SOLVER = 'cadical153'
METHODS = ('auto', 'enumeration', 'sat')
IDENTITY = '1'

type Reach = LatticeNeighborhood | MemoryNeighborhood
type Sites = BaseSpace


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


KINDS = ('injective', 'surjective', 'pre-injective', 'garden-of-eden', 'inverse', 'compose', 'difference', 'entropy')
SETTINGS = {
    'injective': {'method', 'infinite'},
    'surjective': {'method', 'infinite'},
    'pre-injective': {'method', 'window', 'infinite'},
    'garden-of-eden': {'window', 'tries', 'random_state'},
    'inverse': {'max_radius'},
    'compose': {'second'},
    'difference': {'configuration', 'site', 'steps'},
    'entropy': {'windows', 'base'},
}
"""Settings every kind reads, besides `kind`."""
SOURCES = {
    'injective': ('automaton', 'linear automaton'),
    'surjective': ('automaton', 'linear automaton'),
    'pre-injective': ('automaton', 'linear automaton'),
    'garden-of-eden': ('automaton',),
    'inverse': ('automaton',),
    'compose': ('automaton',),
    'difference': ('automaton',),
    'entropy': ('automaton', 'subshift'),
}
"""Models of a cellular automaton every kind reads: rule automata, linear automata, and subshifts."""


class AutomatonAnalysis(BaseAnalysis):
    """Decisions and measures of a cellular automaton on a finite universe, of a linear automaton, or of a subshift.

    The kinds read the automaton, the linear automaton, or the subshift a user builds, such as `Automaton`,
    `LinearAutomaton`, and `Subshift` of the automata models. The analysis tells them apart by their methods: a linear
    automaton has `laurent_polynomial`, a subshift has `violations`, and an automaton with a rule has `local_map`.

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
            The seed or generator of the random patterns.

        max_radius:
            The largest radius of the memory of the inverse.

        second:
            The automaton applied after the analysed one.

        configuration:
            The initial values of the sites of the difference pattern.

        site:
            The site of the changed cell.

        steps:
            The number of steps of the difference pattern.

        windows:
            The windows of the entropy, such as lattices with fixed edges of growing size.

        base:
            The base of the logarithm.

    Attributes:
        automaton_:
            The automaton, the linear automaton, or the subshift.

        injective_:
            Whether the automaton is injective.

        surjective_:
            Whether the automaton is surjective.

        pre_injective_:
            Whether no two configurations that differ only inside the window share their image.

        pattern_:
            A configuration whose values on the window have no preimage, or `None` when none is found.

        inverse_:
            The inverse automaton with a table rule, or `None` when the automaton is not injective or no memory up to
            the radius determines the cells.

        composition_:
            The automaton of the composition, with a function rule.

        differences_:
            One row per step, the initial one first, true where the runs differ.

        entropy_:
            One entropy per window.

    Raises:
        AnalysisError:
            The model is not one the kind reads, the method is unknown, enumeration is asked for too many
            configurations, or composed automata differ in space or number of states.

    Examples:
        >>> from skcomplex.analysis import AutomatonAnalysis
        >>> from skcomplex.specs import TableSpec, TotalisticSpec
        >>> from skcomplex.core.spaces import BoxNeighborhood, LatticeNeighborhood, LatticeSpace
        >>> from skcomplex.models.computer_science.automata import Automaton, Subshift
        >>> shift = Automaton(LatticeSpace(8), LatticeNeighborhood(((1,),)), TableSpec({(1,): 1}))
        >>> AutomatonAnalysis().check().prepare(shift).run().injective_
        True
        >>> majority = Automaton(LatticeSpace(5), BoxNeighborhood(), TotalisticSpec(lambda total: total >= 2))
        >>> AutomatonAnalysis('garden-of-eden').check().prepare(majority).run().pattern_.tolist()
        [0, 0, 1, 0, 1]
        >>> golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
        >>> analysis = AutomatonAnalysis('entropy', windows=[LatticeSpace(20, boundary='fixed')])
        >>> analysis.check().prepare(golden).run().entropy_.round(3).tolist()
        [0.706]
    """

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
        second: AutomatonProtocol | None = None,
        configuration: npt.ArrayLike | None = None,
        site: int = 0,
        steps: int = 1,
        windows: Sequence[Sites] | None = None,
        base: float = 2.0,
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

    def _check(self) -> Self:
        """Validate the kind, the method, that the kind reads every changed setting, and that it has what it needs."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind])
        if self.method not in METHODS:
            error_msg = f'{type(self).__name__} has method={self.method!r}. Use one of {", ".join(METHODS)}.'
            raise AnalysisError(error_msg)
        needed = {'compose': 'second', 'difference': 'configuration', 'entropy': 'windows'}.get(self.kind)
        if needed is not None and getattr(self, needed) is None:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs {needed}. Give it.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the system kind."""
        return DataKind.SYSTEM

    @staticmethod
    def _model_of(source: object) -> str | None:
        """Return the model a source is by its methods: a linear automaton, a subshift, an automaton, or `None`."""
        if hasattr(source, 'laurent_polynomial'):
            return 'linear automaton'
        if hasattr(source, 'violations'):
            return 'subshift'
        if hasattr(source, 'local_map'):
            return 'automaton'
        return None

    def _prepare(self, source: Source) -> Self:
        """Keep the automaton, the linear automaton, or the subshift, after checking that the kind reads it."""
        models = SOURCES[self.kind]
        if self._model_of(source) not in models:
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} reads an {" or a ".join(models)}, not '
                f'{type(source).__name__}. Build one of them.'
            )
            raise AnalysisError(error_msg)
        self.automaton_ = source
        return self

    def _decide(self) -> None:
        """Set whether the automaton is injective, surjective, or pre-injective."""
        automaton = self.automaton_
        if self._model_of(automaton) == 'linear automaton':
            injective = self.kind == 'injective'
            decision = (
                self._linear_injective(automaton, infinite=self.infinite)
                if injective
                else self._linear_surjective(automaton, infinite=self.infinite)
            )
        elif self.kind == 'pre-injective':
            decision = self._is_pre_injective(automaton, self.window, method=self.method)  # type: ignore[arg-type]
        else:
            decision = self._is_injective(automaton, method=self.method)  # type: ignore[arg-type]
        setattr(self, f'{self.kind.replace("-", "_")}_', decision)

    def _run(self, inputs: object = None) -> Self:
        """Decide or measure the automaton."""
        automaton = self.automaton_
        if self.kind in {'injective', 'surjective', 'pre-injective'}:
            self._decide()
        elif self.kind == 'garden-of-eden':
            self.pattern_ = self._garden_of_eden(
                automaton,
                self.window,
                tries=self.tries,
                random_state=self.random_state,
            )
        elif self.kind == 'inverse':
            self.inverse_ = self._inverse(automaton, self.max_radius)
        elif self.kind == 'compose':
            self.composition_ = self._compose(automaton, self.second)  # type: ignore[arg-type]
        elif self.kind == 'difference':
            self.differences_ = self._difference_pattern(automaton, self.configuration, self.site, self.steps)
        else:
            self.entropy_ = self._configuration_entropy(automaton, self.windows, base=self.base)  # type: ignore[arg-type]
        return self

    @staticmethod
    def _fitted(space: Sites) -> Sites:
        """Return a space of sites, preparing it on its first use."""
        if not hasattr(space, 'size_'):
            space.check().prepare()
        return space

    @staticmethod
    def _size(space: Sites) -> int:
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
    def _shifted(element: Any, base: Any) -> Any:  # noqa: ANN401
        """Return the composite of an element of a memory set after an element of another, as a word or an offset."""
        if isinstance(element, str):
            parts = [part for part in (base, element) if part.strip() not in {'', IDENTITY}]
            return '*'.join(parts) or IDENTITY
        return tuple(first + second for first, second in zip(base, element, strict=True))

    @staticmethod
    def _rule_clauses(
        clauses: _Clauses,
        automaton: AutomatonProtocol,
        cells: npt.NDArray[np.int64],
        sites: Sequence[int],
        output: Callable[[int, int], list[int] | None],
    ) -> None:
        """Add, for every site and pattern of its cells, the clause that the pattern implies the output of the rule."""
        table, values = automaton.table(), automaton.local_map()
        patterns = AutomatonAnalysis._patterns(automaton.states, table.shape[1])
        fill = int(getattr(automaton.space, 'fill', 0))
        for site in sites:
            row = table[site]
            for pattern, value in zip(patterns, values, strict=True):
                outside = row >= automaton.size
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

    @staticmethod
    def _collision(automaton: AutomatonProtocol, window: Sequence[int]) -> tuple[IntArray, IntArray] | None:
        """Return two configurations that differ inside a window, agree outside it, and have the same image, by SAT."""
        clauses = _Clauses()
        first = clauses.one_hot(automaton.size, automaton.states)
        second = clauses.one_hot(automaton.size, automaton.states)
        image = clauses.one_hot(automaton.size, automaton.states, exactly=False)
        for cells in (first, second):
            AutomatonAnalysis._rule_clauses(
                clauses,
                automaton,
                cells,
                range(automaton.size),
                lambda site, value: [int(image[site, value])],
            )
        inside = set(window)
        differences = []
        for site in range(automaton.size):
            for value in range(automaton.states):
                same = (int(first[site, value]), int(second[site, value]))
                if site not in inside:
                    clauses.clauses += [[-same[0], same[1]], [same[0], -same[1]]]
                    continue
                differ = clauses.fresh()
                clauses.clauses += [[-differ, same[0]], [-differ, -same[1]]]
                differences.append(differ)
        clauses.clauses.append(differences)
        model = clauses.solve()
        return (
            None
            if model is None
            else (AutomatonAnalysis._decode(model, first), AutomatonAnalysis._decode(model, second))
        )

    @staticmethod
    def _preimage(automaton: AutomatonProtocol, target: IntArray, window: Sequence[int]) -> IntArray | None:
        """Return a configuration whose image equals a target on a window, or `None`, by SAT."""
        clauses = _Clauses()
        cells = clauses.one_hot(automaton.size, automaton.states)

        def forbid(site: int, value: int) -> list[int] | None:
            """Return an empty consequence, which forbids the pattern, when the rule misses the target."""
            return None if value == target[site] else []

        AutomatonAnalysis._rule_clauses(clauses, automaton, cells, window, forbid)
        model = clauses.solve()
        return None if model is None else AutomatonAnalysis._decode(model, cells)

    @staticmethod
    def _images(automaton: AutomatonProtocol) -> tuple[IntArray, IntArray]:
        """Return every configuration and the code of its image."""
        configurations = AutomatonAnalysis._configurations(automaton.states, automaton.size)
        return configurations, AutomatonAnalysis._codes(automaton.step(configurations), automaton.states)

    @staticmethod
    def _ball_neighborhood(automaton: AutomatonProtocol, radius: int) -> Reach:
        """Return the neighbourhood of every cell within a radius, a box of offsets or a ball of group elements."""
        if isinstance(automaton.space, LatticeSpace) and not isinstance(automaton.neighborhood, LatticeNeighborhood):
            error_msg = 'An inverse is searched on a lattice with a neighbourhood of offsets or on a group space.'
            raise AnalysisError(error_msg)
        return cast('Reach', AutomatonAnalysis._fitted(automaton.space)._ball(radius))

    @staticmethod
    def _is_injective(automaton: AutomatonProtocol, *, method: str = 'auto') -> bool:
        """Check whether two different configurations never have the same image."""
        if AutomatonAnalysis._check_method(method, automaton.states**automaton.size <= MAX_CONFIGURATIONS):
            _, codes = AutomatonAnalysis._images(automaton)
            return bool(np.unique(codes).size == codes.size)
        return AutomatonAnalysis._collision(automaton, range(automaton.size)) is None

    @staticmethod
    def _is_pre_injective(
        automaton: AutomatonProtocol,
        window: Sequence[int] | None = None,
        *,
        method: str = 'auto',
    ) -> bool:
        """Check whether two configurations that differ only inside a window never have the same image."""
        sites = list(range(automaton.size)) if window is None else [int(site) for site in window]
        if not AutomatonAnalysis._check_method(method, automaton.states**automaton.size <= MAX_CONFIGURATIONS):
            return AutomatonAnalysis._collision(automaton, sites) is None
        configurations, codes = AutomatonAnalysis._images(automaton)
        outside = np.setdiff1d(np.arange(automaton.size), sites)
        keys = codes * automaton.states**outside.size + AutomatonAnalysis._codes(
            configurations[:, outside],
            automaton.states,
        )
        return bool(np.unique(keys).size == keys.size)

    @staticmethod
    def _garden_of_eden(
        automaton: AutomatonProtocol,
        window: Sequence[int] | None = None,
        *,
        tries: int = 200,
        random_state: int | np.random.Generator | None = None,
    ) -> IntArray | None:
        """Return a pattern on a window that no configuration maps to, or `None`."""
        sites = list(range(automaton.size)) if window is None else [int(site) for site in window]
        if window is None and automaton.states**automaton.size <= MAX_CONFIGURATIONS:
            configurations, codes = AutomatonAnalysis._images(automaton)
            missing = np.setdiff1d(np.arange(automaton.states**automaton.size), codes)
            return configurations[missing[0]] if missing.size else None
        rng = np.random.default_rng(random_state)
        for _ in range(tries):
            target = rng.integers(automaton.states, size=automaton.size)
            if AutomatonAnalysis._preimage(automaton, target, sites) is None:
                return target
        return None

    @staticmethod
    def _inverse(automaton: AutomatonProtocol, max_radius: int = 2) -> AutomatonProtocol | None:
        """Return the automaton whose composition with a reversible one is the identity, or `None`."""
        configurations, codes = AutomatonAnalysis._images(automaton)
        if np.unique(codes).size != codes.size:
            return None
        images = automaton.step(configurations)
        for radius in range(max_radius + 1):
            candidate = type(automaton)(
                automaton.space,
                AutomatonAnalysis._ball_neighborhood(automaton, radius),
                TableSpec({}),
                automaton.states,
            )
            padded = np.concatenate([images, np.zeros((images.shape[0], 1), dtype=np.int64)], axis=1)
            patterns = AutomatonAnalysis._codes(padded[:, candidate.table()], automaton.states).ravel()
            values = configurations.ravel()
            pairs = np.unique(np.column_stack([patterns, values]), axis=0)
            if np.unique(pairs[:, 0]).size != pairs.shape[0]:
                continue
            rows = AutomatonAnalysis._patterns(automaton.states, candidate.table().shape[1])
            mapping = {tuple(int(item) for item in rows[code]): int(value) for code, value in pairs}
            rule = TableSpec(mapping, states=automaton.states)
            return type(automaton)(automaton.space, candidate.reads, rule, automaton.states)
        return None

    @staticmethod
    def _compose(first: AutomatonProtocol, second: AutomatonProtocol) -> AutomatonProtocol:
        """Return the automaton that applies one automaton and then another on the same space."""
        if first.space is not second.space or first.states != second.states:
            error_msg = 'Composed automata share their space and their number of states.'
            raise AnalysisError(error_msg)
        inner, outer = first.reads, second.reads
        inner_items = inner.elements if isinstance(inner, MemoryNeighborhood) else inner.check().prepare().offsets_
        outer_items = outer.elements if isinstance(outer, MemoryNeighborhood) else outer.check().prepare().offsets_
        composite: list[Any] = []
        columns = []
        for base in outer_items:
            row = []
            for element in inner_items:
                item = AutomatonAnalysis._shifted(element, base)
                if item not in composite:
                    composite.append(item)
                row.append(composite.index(item))
            columns.append(tuple(row))
        reach: Reach = (
            MemoryNeighborhood(tuple(composite))
            if isinstance(inner, MemoryNeighborhood)
            else LatticeNeighborhood(tuple(composite))
        )
        composed = _ComposedRuleSpec(first.local_map(), second.local_map(), tuple(columns), first.states)
        rule = functools.partial(AutomatonAnalysis._composed_rule, composed)
        return type(first)(first.space, reach, rule, first.states)

    @staticmethod
    def _composed_rule(composed: _ComposedRuleSpec, *values: npt.ArrayLike) -> IntArray:
        """Return the value of the second rule of a composition on the values of the first rule around every cell."""
        cells = np.stack([np.asarray(value, dtype=np.int64) for value in values], axis=-1)
        inner = [
            composed.first[AutomatonAnalysis._codes(cells[..., list(columns)], composed.states)]
            for columns in composed.columns
        ]
        return composed.second[AutomatonAnalysis._codes(np.stack(inner, axis=-1), composed.states)]

    @staticmethod
    def _difference_pattern(
        automaton: AutomatonProtocol,
        configuration: npt.ArrayLike,
        site: int,
        steps: int,
    ) -> npt.NDArray[np.bool_]:
        """Return where two runs differ when one starts with one cell changed to its next value."""
        initial = np.asarray(configuration, dtype=np.int64)
        changed = initial.copy()
        changed[site] = (changed[site] + 1) % automaton.states
        return automaton.run(initial, steps) != automaton.run(changed, steps)

    @staticmethod
    def _configuration_entropy(
        subject: SubshiftProtocol | AutomatonProtocol,
        windows: Sequence[Sites],
        *,
        base: float = 2.0,
    ) -> npt.NDArray[np.float64]:
        """Return the entropy `log |X_F| / |F|` of a subshift or of the image of an automaton on every window."""
        entropies = []
        for window in windows:
            if hasattr(subject, 'violations'):
                entropies.append(np.log(subject.count(window)) / AutomatonAnalysis._size(window))
                continue
            automaton = type(subject)(window, subject.neighborhood, subject.rule, subject.states)
            interior = np.flatnonzero((automaton.table() < AutomatonAnalysis._size(window)).all(axis=1))
            images = automaton.step(
                AutomatonAnalysis._configurations(automaton.states, AutomatonAnalysis._size(window)),
            )[:, interior]
            count = np.unique(AutomatonAnalysis._codes(images, automaton.states)).size
            entropies.append(np.log(count) / max(interior.size, 1))
        return np.asarray(entropies) / np.log(base)

    @staticmethod
    def _linear_injective(automaton: LinearAutomatonProtocol, *, infinite: bool) -> bool:
        """Check whether a linear automaton is invertible: its matrix on its universe, or a monomial on `Z^d`."""
        if infinite:
            return len(automaton.laurent_polynomial().exponents()) == 1
        return bool(automaton.matrix().is_invertible())

    @staticmethod
    def _linear_surjective(automaton: LinearAutomatonProtocol, *, infinite: bool) -> bool:
        """Check whether a linear automaton is surjective: invertible on its universe, or not zero on `Z^d`."""
        if infinite:
            return not automaton.laurent_polynomial().is_zero()
        return AutomatonAnalysis._linear_injective(automaton, infinite=False)
