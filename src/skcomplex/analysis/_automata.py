"""Decide injectivity and surjectivity of cellular automata and subshifts, and measure their entropy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self, cast

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ..core.laws.utils import Table
from ..core.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace, MemoryNeighborhood
from ._base import BaseAnalysis, Source
from ._kinds import DataKind
from ._lazy import module

MAX_CONFIGURATIONS = 2**16
MAX_PATTERNS = 2**20
SAT_SOLVER = 'cadical153'
METHODS = ('auto', 'enumeration', 'sat')
IDENTITY = '1'

type Codes = npt.NDArray[np.int64]
type Indices = npt.NDArray[np.intp]
type Reach = LatticeNeighborhood | MemoryNeighborhood
type Sites = BaseSpace


class _Automaton(Protocol):
    """Cellular automaton the analysis reads by its methods, such as `Automaton` of the automata models."""

    space: Sites
    neighborhood: Reach
    rule: object
    states: int

    @property
    def reads(self) -> Reach:
        """Neighbourhood the rule reads."""

    @property
    def size(self) -> int:
        """Number of sites."""

    def table(self) -> Indices:
        """Return the site of every cell the rule reads at every site."""

    def local_map(self) -> Codes:
        """Return the next value of every pattern of the cells the rule reads."""

    def step(self, configuration: npt.ArrayLike) -> Codes:
        """Return the image of a configuration."""

    def run(self, configuration: npt.ArrayLike, steps: int) -> Codes:
        """Return the configurations of a number of steps."""


class _LinearAutomaton(Protocol):
    """Linear cellular automaton the analysis reads by its matrix and its Laurent polynomial."""

    def matrix(self) -> Any:  # noqa: ANN401
        """Return the Sage matrix of the automaton on its space."""

    def laurent_polynomial(self) -> Any:  # noqa: ANN401
        """Return the Sage Laurent polynomial of the automaton on `Z^d`."""


class _Subshift(Protocol):
    """Subshift of finite type the analysis reads by its count of admissible configurations and its violations."""

    def count(self, space: Sites) -> int:
        """Return the number of admissible configurations of a space."""

    def violations(self, configuration: npt.ArrayLike, space: Sites) -> npt.NDArray[np.bool_]:
        """Return the sites at which a forbidden pattern appears."""


def _fitted(space: Sites) -> Sites:
    """Return a space of sites, preparing it on its first use."""
    if not hasattr(space, 'size_'):
        space.check().prepare()
    return space


def _size(space: Sites) -> int:
    """Return the number of sites of a space."""
    return int(_fitted(space).size_ or 0)


def _patterns(states: int, width: int) -> Codes:
    """Return every pattern of values of a number of cells, in the order of their codes."""
    if states**width > MAX_PATTERNS:
        error_msg = f'A rule on {width} cells with {states} values has more than {MAX_PATTERNS} patterns.'
        raise AnalysisError(error_msg)
    return np.asarray(list(itertools.product(range(states), repeat=width)), dtype=np.int64).reshape(-1, width)


def _codes(values: Codes, states: int) -> Codes:
    """Return the code of every row of values, read as digits in the base of the number of states."""
    weights = states ** np.arange(values.shape[-1] - 1, -1, -1, dtype=np.int64)
    return values @ weights


def _configurations(states: int, size: int) -> Codes:
    """Return every configuration of a number of sites, or raise an error when there are too many."""
    if states**size > MAX_CONFIGURATIONS:
        error_msg = (
            f'{states}^{size} configurations exceed the enumeration limit of {MAX_CONFIGURATIONS}. Use the SAT method, '
            'or a smaller universe.'
        )
        raise AnalysisError(error_msg)
    return _patterns(states, size)


def _check_method(method: str, enumerable: bool) -> bool:
    """Return whether to enumerate configurations, after checking the method."""
    if method not in METHODS:
        error_msg = f'Unknown method {method!r}. Use one of {", ".join(METHODS)}.'
        raise AnalysisError(error_msg)
    return method == 'enumeration' or (method == 'auto' and enumerable)


def _shifted(element: Any, base: Any) -> Any:  # noqa: ANN401
    """Return the composite of an element of a memory set after an element of another, as a word or an offset."""
    if isinstance(element, str):
        parts = [part for part in (base, element) if part.strip() not in {'', IDENTITY}]
        return '*'.join(parts) or IDENTITY
    return tuple(first + second for first, second in zip(base, element, strict=True))


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
        solvers = module('pysat.solvers')
        with solvers.Solver(name=SAT_SOLVER, bootstrap_with=self.clauses) as solver:
            return solver.get_model() if solver.solve() else None


def _rule_clauses(
    clauses: _Clauses,
    automaton: _Automaton,
    cells: npt.NDArray[np.int64],
    sites: Sequence[int],
    output: Callable[[int, int], list[int] | None],
) -> None:
    """Add, for every site and pattern of its cells, the clause that the pattern implies the output of the rule."""
    table, values = automaton.table(), automaton.local_map()
    patterns = _patterns(automaton.states, table.shape[1])
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
                -int(cells[neighbour, item]) for neighbour, item in zip(row[~outside], pattern[~outside], strict=True)
            ]
            clauses.clauses.append(premise + consequence)


def _decode(model: list[int], variables: npt.NDArray[np.int64]) -> Codes:
    """Return the values of one-hot variables in a model."""
    true = set(model)
    return np.asarray(
        [next(value for value, item in enumerate(row) if item in true) for row in variables.tolist()],
        dtype=np.int64,
    )


def _collision(automaton: _Automaton, window: Sequence[int]) -> tuple[Codes, Codes] | None:
    """Return two configurations that differ inside a window, agree outside it, and have the same image, by SAT."""
    clauses = _Clauses()
    first = clauses.one_hot(automaton.size, automaton.states)
    second = clauses.one_hot(automaton.size, automaton.states)
    image = clauses.one_hot(automaton.size, automaton.states, exactly=False)
    for cells in (first, second):
        _rule_clauses(clauses, automaton, cells, range(automaton.size), lambda site, value: [int(image[site, value])])
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
    return None if model is None else (_decode(model, first), _decode(model, second))


def _preimage(automaton: _Automaton, target: Codes, window: Sequence[int]) -> Codes | None:
    """Return a configuration whose image equals a target on a window, or `None`, by SAT."""
    clauses = _Clauses()
    cells = clauses.one_hot(automaton.size, automaton.states)

    def forbid(site: int, value: int) -> list[int] | None:
        """Return an empty consequence, which forbids the pattern, when the rule misses the target."""
        return None if value == target[site] else []

    _rule_clauses(clauses, automaton, cells, window, forbid)
    model = clauses.solve()
    return None if model is None else _decode(model, cells)


def _images(automaton: _Automaton) -> tuple[Codes, Codes]:
    """Return every configuration and the code of its image."""
    configurations = _configurations(automaton.states, automaton.size)
    return configurations, _codes(automaton.step(configurations), automaton.states)


def _ball_neighborhood(automaton: _Automaton, radius: int) -> Reach:
    """Return the neighbourhood of every cell within a radius, a box of offsets or a ball of group elements."""
    if isinstance(automaton.space, LatticeSpace) and not isinstance(automaton.neighborhood, LatticeNeighborhood):
        error_msg = 'An inverse is searched on a lattice with a neighbourhood of offsets or on a group space.'
        raise AnalysisError(error_msg)
    return cast('Reach', _fitted(automaton.space)._ball(radius))


def _is_injective(automaton: _Automaton, *, method: str = 'auto') -> bool:
    """Check whether two different configurations never have the same image."""
    if _check_method(method, automaton.states**automaton.size <= MAX_CONFIGURATIONS):
        _, codes = _images(automaton)
        return bool(np.unique(codes).size == codes.size)
    return _collision(automaton, range(automaton.size)) is None


def _is_surjective(automaton: _Automaton, *, method: str = 'auto') -> bool:
    """Check whether every configuration has a preimage, as when it is injective."""
    return _is_injective(automaton, method=method)


def _is_pre_injective(automaton: _Automaton, window: Sequence[int] | None = None, *, method: str = 'auto') -> bool:
    """Check whether two configurations that differ only inside a window never have the same image."""
    sites = list(range(automaton.size)) if window is None else [int(site) for site in window]
    if not _check_method(method, automaton.states**automaton.size <= MAX_CONFIGURATIONS):
        return _collision(automaton, sites) is None
    configurations, codes = _images(automaton)
    outside = np.setdiff1d(np.arange(automaton.size), sites)
    keys = codes * automaton.states**outside.size + _codes(configurations[:, outside], automaton.states)
    return bool(np.unique(keys).size == keys.size)


def _garden_of_eden(
    automaton: _Automaton,
    window: Sequence[int] | None = None,
    *,
    tries: int = 200,
    random_state: int | np.random.Generator | None = None,
) -> Codes | None:
    """Return a pattern on a window that no configuration maps to, or `None`."""
    sites = list(range(automaton.size)) if window is None else [int(site) for site in window]
    if window is None and automaton.states**automaton.size <= MAX_CONFIGURATIONS:
        configurations, codes = _images(automaton)
        missing = np.setdiff1d(np.arange(automaton.states**automaton.size), codes)
        return configurations[missing[0]] if missing.size else None
    rng = np.random.default_rng(random_state)
    for _ in range(tries):
        target = rng.integers(automaton.states, size=automaton.size)
        if _preimage(automaton, target, sites) is None:
            return target
    return None


def _inverse(automaton: _Automaton, max_radius: int = 2) -> _Automaton | None:
    """Return the automaton whose composition with a reversible one is the identity, or `None`."""
    configurations, codes = _images(automaton)
    if np.unique(codes).size != codes.size:
        return None
    images = automaton.step(configurations)
    for radius in range(max_radius + 1):
        candidate = type(automaton)(automaton.space, _ball_neighborhood(automaton, radius), Table({}), automaton.states)
        padded = np.concatenate([images, np.zeros((images.shape[0], 1), dtype=np.int64)], axis=1)
        patterns = _codes(padded[:, candidate.table()], automaton.states).ravel()
        values = configurations.ravel()
        pairs = np.unique(np.column_stack([patterns, values]), axis=0)
        if np.unique(pairs[:, 0]).size != pairs.shape[0]:
            continue
        rows = _patterns(automaton.states, candidate.table().shape[1])
        mapping = {tuple(int(item) for item in rows[code]): int(value) for code, value in pairs}
        rule = Table(mapping, states=automaton.states)
        return type(automaton)(automaton.space, candidate.reads, rule, automaton.states)
    return None


def _compose(first: _Automaton, second: _Automaton) -> _Automaton:
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
            item = _shifted(element, base)
            if item not in composite:
                composite.append(item)
            row.append(composite.index(item))
        columns.append(tuple(row))
    reach: Reach = (
        MemoryNeighborhood(tuple(composite))
        if isinstance(inner, MemoryNeighborhood)
        else LatticeNeighborhood(tuple(composite))
    )
    rule = _ComposedRule(first.local_map(), second.local_map(), tuple(columns), first.states).evaluate
    return type(first)(first.space, reach, rule, first.states)


def _difference_pattern(
    automaton: _Automaton,
    configuration: npt.ArrayLike,
    site: int,
    steps: int,
) -> npt.NDArray[np.bool_]:
    """Return where two runs differ when one starts with one cell changed to its next value."""
    initial = np.asarray(configuration, dtype=np.int64)
    changed = initial.copy()
    changed[site] = (changed[site] + 1) % automaton.states
    return automaton.run(initial, steps) != automaton.run(changed, steps)


def _configuration_entropy(
    subject: _Subshift | _Automaton,
    windows: Sequence[Sites],
    *,
    base: float = 2.0,
) -> npt.NDArray[np.float64]:
    """Return the entropy `log |X_F| / |F|` of a subshift or of the image of an automaton on every window."""
    entropies = []
    for window in windows:
        if hasattr(subject, 'violations'):
            entropies.append(np.log(subject.count(window)) / _size(window))
            continue
        automaton = type(subject)(window, subject.neighborhood, subject.rule, subject.states)
        interior = np.flatnonzero((automaton.table() < _size(window)).all(axis=1))
        images = automaton.step(_configurations(automaton.states, _size(window)))[:, interior]
        count = np.unique(_codes(images, automaton.states)).size
        entropies.append(np.log(count) / max(interior.size, 1))
    return np.asarray(entropies) / np.log(base)


@dataclass(frozen=True)
class _ComposedRule:
    """Rule of the composition of two automata, on the composite neighbourhood."""

    first: Codes
    second: Codes
    columns: tuple[tuple[int, ...], ...]
    states: int

    def evaluate(self, *values: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
        """Return the value of the second rule on the values of the first rule around every cell."""
        cells = np.stack([np.asarray(value, dtype=np.int64) for value in values], axis=-1)
        inner = [self.first[_codes(cells[..., list(columns)], self.states)] for columns in self.columns]
        return self.second[_codes(np.stack(inner, axis=-1), self.states)]


def _linear_injective(automaton: _LinearAutomaton, *, infinite: bool) -> bool:
    """Check whether a linear automaton is invertible: its matrix on its universe, or a monomial on `Z^d`."""
    if infinite:
        return len(automaton.laurent_polynomial().exponents()) == 1
    return bool(automaton.matrix().is_invertible())


def _linear_surjective(automaton: _LinearAutomaton, *, infinite: bool) -> bool:
    """Check whether a linear automaton is surjective: invertible on its universe, or not zero on `Z^d`."""
    if infinite:
        return not automaton.laurent_polynomial().is_zero()
    return _linear_injective(automaton, infinite=False)


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
        >>> from skcomplex.core.laws.utils import Table, Totalistic
        >>> from skcomplex.core.spaces import BoxNeighborhood, LatticeNeighborhood, LatticeSpace
        >>> from skcomplex.models.computer_science.automata import Automaton, Subshift
        >>> shift = Automaton(LatticeSpace(8), LatticeNeighborhood(((1,),)), Table({(1,): 1}))
        >>> AutomatonAnalysis().check().prepare(shift).run().injective_
        True
        >>> majority = Automaton(LatticeSpace(5), BoxNeighborhood(), Totalistic(lambda total: total >= 2))
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
        second: _Automaton | None = None,
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
                _linear_injective(automaton, infinite=self.infinite)
                if injective
                else _linear_surjective(automaton, infinite=self.infinite)
            )
        elif self.kind == 'pre-injective':
            decision = _is_pre_injective(automaton, self.window, method=self.method)  # type: ignore[arg-type]
        else:
            decision = _is_injective(automaton, method=self.method)  # type: ignore[arg-type]
        setattr(self, f'{self.kind.replace("-", "_")}_', decision)

    def _run(self) -> Self:
        """Decide or measure the automaton."""
        automaton = self.automaton_
        if self.kind in {'injective', 'surjective', 'pre-injective'}:
            self._decide()
        elif self.kind == 'garden-of-eden':
            self.pattern_ = _garden_of_eden(automaton, self.window, tries=self.tries, random_state=self.random_state)
        elif self.kind == 'inverse':
            self.inverse_ = _inverse(automaton, self.max_radius)
        elif self.kind == 'compose':
            self.composition_ = _compose(automaton, self.second)  # type: ignore[arg-type]
        elif self.kind == 'difference':
            self.differences_ = _difference_pattern(automaton, self.configuration, self.site, self.steps)
        else:
            self.entropy_ = _configuration_entropy(automaton, self.windows, base=self.base)  # type: ignore[arg-type]
        return self
