"""Decide injectivity and surjectivity of cellular automata on finite universes, and measure their information."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import GF, QQ, LaurentPolynomialRing, matrix

from ..base import AnalysisError
from ..laws import OuterTotalistic, Table, Totalistic
from ..space import GroupSpace, LatticeSpace, Memory, Neighborhood
from ._information import entropy_rate
from ._kinds import DataKind, analysis
from ._lazy import module

MAX_CONFIGURATIONS = 2**16
MAX_PATTERNS = 2**20
SAT_SOLVER = 'cadical153'
METHODS = ('auto', 'enumeration', 'sat')
IDENTITY = '1'

type Codes = npt.NDArray[np.int64]
type Indices = npt.NDArray[np.intp]
type Reach = Neighborhood | Memory
type Sites = LatticeSpace | GroupSpace


def _rule_table(rule: object) -> object:
    """Return a rule code as its table, through its `table()` method, and any other rule unchanged."""
    convert = getattr(rule, 'table', None)
    return convert() if not callable(rule) and callable(convert) else rule


def _with_center(neighborhood: Reach) -> Reach:
    """Return a neighbourhood with the site itself first, as outer totalistic rules read it."""
    if isinstance(neighborhood, Memory):
        return Memory((IDENTITY, *neighborhood.elements))
    zero = (0,) * len(neighborhood.offsets[0])
    return Neighborhood((zero, *neighborhood.offsets))


def _evaluate(rule: object, patterns: Codes) -> npt.NDArray[Any]:
    """Return the next value of every pattern, one row per pattern in the order the rule reads its cells."""
    if isinstance(rule, Table):
        return np.asarray([rule.mapping.get(tuple(int(item) for item in row), rule.default) for row in patterns])
    if isinstance(rule, Totalistic):
        sums = patterns.sum(axis=1)
        if callable(rule.rule):
            return np.broadcast_to(np.asarray(rule.rule(sums)), sums.shape)
        return np.asarray([rule.rule.get(int(total), rule.default) for total in sums])
    if isinstance(rule, OuterTotalistic):
        centers, sums = patterns[:, 0], patterns[:, 1:].sum(axis=1)
        if callable(rule.rule):
            return np.broadcast_to(np.asarray(rule.rule(centers, sums)), sums.shape)
        pairs = zip(centers.tolist(), sums.tolist(), strict=True)
        return np.asarray([rule.rule.get((center, total), rule.default) for center, total in pairs])
    if callable(rule):
        return np.broadcast_to(np.asarray(rule(*patterns.T)), (patterns.shape[0],))
    error_msg = (
        f'The rule {rule!r} is not a Table, a totalistic rule, a rule code with a table() method, or a function.'
    )
    raise AnalysisError(error_msg)


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


@dataclass(frozen=True)
class Automaton:
    """Cellular automaton on a finite universe: a periodic lattice, a finite quotient of a group, or a ball.

    The automaton maps a configuration, one integer value from `0` to `states - 1` per site, to the configuration whose
    value at every site is the rule applied to the values of the neighbourhood of the site, as `LocalRule` laws do.
    Sites outside a lattice with fixed edges or outside a ball read the fill value of the space.

    Args:
        space:
            The lattice or the group space.

        neighborhood:
            The neighbourhood of offsets on a lattice, or the memory set on a group space.

        rule:
            A `Table`, a `Totalistic` rule, an `OuterTotalistic` rule, an object whose `table()` returns a `Table`,
            or a function of one array per neighbour that returns the next values.

        states:
            The number of values of a cell.

    Examples:
        >>> from skcomplex.core.analysis import Automaton
        >>> from skcomplex.core.laws import Totalistic
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> parity = Automaton(LatticeSpace(5), Neighborhood.radius(), Totalistic(lambda total: total % 2))
        >>> parity.step([0, 0, 1, 0, 0]).tolist()
        [0, 1, 1, 1, 0]
    """

    space: Sites
    neighborhood: Reach
    rule: Any
    states: int = 2

    @property
    def reads(self) -> Reach:
        """Neighbourhood the rule reads, with the site itself first for an outer totalistic rule."""
        rule = _rule_table(self.rule)
        return _with_center(self.neighborhood) if isinstance(rule, OuterTotalistic) else self.neighborhood

    @property
    def size(self) -> int:
        """Number of sites."""
        return int(self.space.size)

    def table(self) -> Indices:
        """Return the site of every cell the rule reads at every site, with the number of sites outside the space.

        Returns:
            table:
                One row per site and one column per cell of `reads`.
        """
        return self.space.index_table(self.reads)  # type: ignore[arg-type]

    def local_map(self) -> Codes:
        """Return the next value of every pattern of the cells the rule reads, indexed by the code of the pattern.

        Returns:
            values:
                One integer value per pattern.

        Raises:
            AnalysisError:
                The rule returns a value outside `0` to `states - 1`.
        """
        patterns = _patterns(self.states, len(self.reads))
        values = np.asarray(_evaluate(_rule_table(self.rule), patterns), dtype=float)
        if values.size and (values.min() < 0 or values.max() >= self.states or not np.all(values == np.rint(values))):
            error_msg = f'The rule returns values outside the integers 0 to {self.states - 1}.'
            raise AnalysisError(error_msg)
        return values.astype(np.int64)

    def step(self, configuration: npt.ArrayLike) -> Codes:
        """Return the image of a configuration, or of one configuration per row.

        Args:
            configuration:
                The values of the sites, or one row of values per configuration.

        Returns:
            image:
                The next values, of the same shape.
        """
        values = np.asarray(configuration, dtype=np.int64)
        fill = np.full((*values.shape[:-1], 1), int(getattr(self.space, 'fill', 0)), dtype=np.int64)
        padded = np.concatenate([values, fill], axis=-1)
        return self.local_map()[_codes(padded[..., self.table()], self.states)]

    def run(self, configuration: npt.ArrayLike, steps: int) -> Codes:
        """Return the configurations of a number of steps, the initial one first.

        Args:
            configuration:
                The initial values of the sites.

            steps:
                The number of steps.

        Returns:
            configurations:
                One row per step.
        """
        rows = [np.asarray(configuration, dtype=np.int64)]
        for _ in range(steps):
            rows.append(self.step(rows[-1]))
        return np.stack(rows)


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
    automaton: Automaton,
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


def _collision(automaton: Automaton, window: Sequence[int]) -> tuple[Codes, Codes] | None:
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


def _preimage(automaton: Automaton, target: Codes, window: Sequence[int]) -> Codes | None:
    """Return a configuration whose image equals a target on a window, or `None`, by SAT."""
    clauses = _Clauses()
    cells = clauses.one_hot(automaton.size, automaton.states)

    def forbid(site: int, value: int) -> list[int] | None:
        """Return an empty consequence, which forbids the pattern, when the rule misses the target."""
        return None if value == target[site] else []

    _rule_clauses(clauses, automaton, cells, window, forbid)
    model = clauses.solve()
    return None if model is None else _decode(model, cells)


def _images(automaton: Automaton) -> tuple[Codes, Codes]:
    """Return every configuration and the code of its image."""
    configurations = _configurations(automaton.states, automaton.size)
    return configurations, _codes(automaton.step(configurations), automaton.states)


def is_injective(automaton: Automaton, *, method: str = 'auto') -> bool:
    """Check whether two different configurations never have the same image.

    Small universes are decided by enumerating every configuration, and larger ones by a SAT encoding solved by PySAT.

    Args:
        automaton:
            The automaton on a finite universe.

        method:
            `'enumeration'`, `'sat'`, or `'auto'` to enumerate up to `MAX_CONFIGURATIONS` configurations.

    Returns:
        injective:
            Whether the automaton is injective on its universe.

    Raises:
        AnalysisError:
            The method is unknown, or enumeration is asked for too many configurations.

    Examples:
        >>> from skcomplex.core.analysis import Automaton, is_injective
        >>> from skcomplex.core.laws import Table
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> shift = Automaton(LatticeSpace(8), Neighborhood(((1,),)), Table({(1,): 1}))
        >>> is_injective(shift)
        True
    """
    if _check_method(method, automaton.states**automaton.size <= MAX_CONFIGURATIONS):
        _, codes = _images(automaton)
        return bool(np.unique(codes).size == codes.size)
    return _collision(automaton, range(automaton.size)) is None


def is_surjective(automaton: Automaton, *, method: str = 'auto') -> bool:
    """Check whether every configuration has a preimage, which on a finite universe holds exactly when it is injective.

    Args:
        automaton:
            The automaton on a finite universe.

        method:
            `'enumeration'`, `'sat'`, or `'auto'` to enumerate up to `MAX_CONFIGURATIONS` configurations.

    Returns:
        surjective:
            Whether the automaton is surjective on its universe.

    Raises:
        AnalysisError:
            The method is unknown, or enumeration is asked for too many configurations.

    Examples:
        >>> from skcomplex.core.analysis import Automaton, is_surjective
        >>> from skcomplex.core.laws import Totalistic
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> majority = Automaton(LatticeSpace(8), Neighborhood.radius(), Totalistic(lambda total: total >= 2))
        >>> is_surjective(majority)
        False
    """
    return is_injective(automaton, method=method)


def is_pre_injective(automaton: Automaton, window: Sequence[int] | None = None, *, method: str = 'auto') -> bool:
    """Check whether two configurations that differ only inside a window never have the same image.

    Args:
        automaton:
            The automaton on a finite universe.

        window:
            The sites where the configurations may differ, or `None` for every site.

        method:
            `'enumeration'`, `'sat'`, or `'auto'` to enumerate up to `MAX_CONFIGURATIONS` configurations.

    Returns:
        pre_injective:
            Whether no two configurations that differ only inside the window share their image.

    Raises:
        AnalysisError:
            The method is unknown, or enumeration is asked for too many configurations.

    Examples:
        >>> from skcomplex.core.analysis import Automaton, is_pre_injective
        >>> from skcomplex.core.laws import Totalistic
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> xor = Automaton(LatticeSpace(6), Neighborhood(((0,), (1,))), Totalistic(lambda total: total % 2))
        >>> is_pre_injective(xor, [2])
        True
    """
    sites = list(range(automaton.size)) if window is None else [int(site) for site in window]
    if not _check_method(method, automaton.states**automaton.size <= MAX_CONFIGURATIONS):
        return _collision(automaton, sites) is None
    configurations, codes = _images(automaton)
    outside = np.setdiff1d(np.arange(automaton.size), sites)
    keys = codes * automaton.states**outside.size + _codes(configurations[:, outside], automaton.states)
    return bool(np.unique(keys).size == keys.size)


def garden_of_eden(
    automaton: Automaton,
    window: Sequence[int] | None = None,
    *,
    tries: int = 200,
    random_state: int | np.random.Generator | None = None,
) -> Codes | None:
    """Return a pattern on a window that no configuration maps to, a Garden of Eden, or `None` when none is found.

    Small universes enumerate the images of every configuration. Larger ones draw random patterns and keep the first
    one whose preimage constraints a SAT solver finds unsatisfiable.

    Args:
        automaton:
            The automaton on a finite universe.

        window:
            The sites of the pattern, or `None` for every site.

        tries:
            The number of random patterns tried by SAT.

        random_state:
            The seed or generator of the random patterns.

    Returns:
        pattern:
            A configuration whose values on the window have no preimage, or `None`.

    Examples:
        >>> from skcomplex.core.analysis import Automaton, garden_of_eden
        >>> from skcomplex.core.laws import Totalistic
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> majority = Automaton(LatticeSpace(5), Neighborhood.radius(), Totalistic(lambda total: total >= 2))
        >>> garden_of_eden(majority).tolist()
        [0, 0, 1, 0, 1]
    """
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


def _ball_neighborhood(automaton: Automaton, radius: int) -> Reach:
    """Return the neighbourhood of every cell within a radius, a box of offsets or a ball of group elements."""
    if isinstance(automaton.space, GroupSpace):
        sizes = automaton.space.ball_sizes()
        return Memory(tuple(automaton.space.words[: sizes[min(radius, len(sizes) - 1)]]))
    if not isinstance(automaton.neighborhood, Neighborhood):
        error_msg = 'An inverse is searched on a lattice with a neighbourhood of offsets or on a group space.'
        raise AnalysisError(error_msg)
    return Neighborhood.radius(radius, n_dim=len(automaton.neighborhood.offsets[0]))


def inverse(automaton: Automaton, max_radius: int = 2) -> Automaton | None:
    """Return an automaton whose composition with a reversible automaton is the identity, with the smallest memory.

    The search tries the neighbourhoods of every radius up to the largest, a box of offsets on a lattice or the ball
    of group elements on a group space, and keeps the first on which the image determines every cell.

    Args:
        automaton:
            The automaton on a universe small enough to enumerate.

        max_radius:
            The largest radius of the memory of the inverse.

    Returns:
        inverse:
            The inverse automaton with a table rule, or `None` when the automaton is not injective or no memory up to
            the radius determines the cells.

    Raises:
        AnalysisError:
            The universe has too many configurations to enumerate.

    Examples:
        >>> from skcomplex.core.analysis import Automaton, inverse
        >>> from skcomplex.core.laws import Table
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> shift = Automaton(LatticeSpace(6), Neighborhood(((1,),)), Table({(1,): 1}))
        >>> inverse(shift).step([1, 0, 0, 0, 0, 0]).tolist()
        [0, 1, 0, 0, 0, 0]
    """
    configurations, codes = _images(automaton)
    if np.unique(codes).size != codes.size:
        return None
    images = automaton.step(configurations)
    for radius in range(max_radius + 1):
        candidate = Automaton(automaton.space, _ball_neighborhood(automaton, radius), Table({}), automaton.states)
        padded = np.concatenate([images, np.zeros((images.shape[0], 1), dtype=np.int64)], axis=1)
        patterns = _codes(padded[:, candidate.table()], automaton.states).ravel()
        values = configurations.ravel()
        pairs = np.unique(np.column_stack([patterns, values]), axis=0)
        if np.unique(pairs[:, 0]).size != pairs.shape[0]:
            continue
        rows = _patterns(automaton.states, len(candidate.reads))
        mapping = {tuple(int(item) for item in rows[code]): int(value) for code, value in pairs}
        return Automaton(automaton.space, candidate.reads, Table(mapping, states=automaton.states), automaton.states)
    return None


@dataclass(frozen=True)
class _ComposedRule:
    """Rule of the composition of two automata, on the composite neighbourhood."""

    first: Codes
    second: Codes
    columns: tuple[tuple[int, ...], ...]
    states: int

    def __call__(self, *values: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
        """Return the value of the second rule on the values of the first rule around every cell."""
        cells = np.stack([np.asarray(value, dtype=np.int64) for value in values], axis=-1)
        inner = [self.first[_codes(cells[..., list(columns)], self.states)] for columns in self.columns]
        return self.second[_codes(np.stack(inner, axis=-1), self.states)]


def compose(first: Automaton, second: Automaton) -> Automaton:
    """Return the automaton that applies one automaton and then another on the same space.

    Its neighbourhood holds the composite `t s` of every element `t` read by the second and `s` read by the first.

    Args:
        first:
            The automaton applied first.

        second:
            The automaton applied second.

    Returns:
        composition:
            The automaton of the composition, with a function rule.

    Raises:
        AnalysisError:
            The automata have different spaces or numbers of states.

    Examples:
        >>> from skcomplex.core.analysis import Automaton, compose
        >>> from skcomplex.core.laws import Table
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> space = LatticeSpace(5)
        >>> left = Automaton(space, Neighborhood(((1,),)), Table({(1,): 1}))
        >>> right = Automaton(space, Neighborhood(((-1,),)), Table({(1,): 1}))
        >>> compose(left, right).step([0, 1, 0, 0, 1]).tolist()
        [0, 1, 0, 0, 1]
    """
    if first.space is not second.space or first.states != second.states:
        error_msg = 'Composed automata share their space and their number of states.'
        raise AnalysisError(error_msg)
    inner, outer = first.reads, second.reads
    inner_items = inner.elements if isinstance(inner, Memory) else inner.offsets
    outer_items = outer.elements if isinstance(outer, Memory) else outer.offsets
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
    reach: Reach = Memory(tuple(composite)) if isinstance(inner, Memory) else Neighborhood(tuple(composite))
    rule = _ComposedRule(first.local_map(), second.local_map(), tuple(columns), first.states)
    return Automaton(first.space, reach, rule, first.states)


def difference_pattern(
    automaton: Automaton,
    configuration: npt.ArrayLike,
    site: int,
    steps: int,
) -> npt.NDArray[np.bool_]:
    """Return where two runs differ when one starts from a configuration and the other from it with one cell changed.

    The changed cell takes the next value modulo the number of states.

    Args:
        automaton:
            The automaton.

        configuration:
            The initial values of the sites.

        site:
            The site of the changed cell.

        steps:
            The number of steps.

    Returns:
        differences:
            One row per step, the initial one first, true where the runs differ.

    Examples:
        >>> from skcomplex.core.analysis import Automaton, difference_pattern
        >>> from skcomplex.core.laws import Totalistic
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> parity = Automaton(LatticeSpace(7), Neighborhood.radius(), Totalistic(lambda total: total % 2))
        >>> difference_pattern(parity, [0] * 7, 3, 2).sum(axis=1).tolist()
        [1, 3, 3]
    """
    initial = np.asarray(configuration, dtype=np.int64)
    changed = initial.copy()
    changed[site] = (changed[site] + 1) % automaton.states
    return automaton.run(initial, steps) != automaton.run(changed, steps)


@analysis(DataKind.FIELD)
def spatial_entropy_rates(
    configurations: npt.ArrayLike,
    max_m: int = 8,
    *,
    base: float = 2.0,
) -> npt.NDArray[np.float64]:
    """Return the spatial entropy rate of every configuration of a run of a one-dimensional automaton.

    The rate of every row is estimated from the block entropies up to the largest block length, as `entropy_rate`
    does. Lindgren's theorem states that it never increases under a cellular automaton.

    Args:
        configurations:
            One row of cell values per step.

        max_m:
            The largest block length.

        base:
            The base of the logarithm.

    Returns:
        rates:
            One entropy rate per row.

    Examples:
        >>> from skcomplex.core.analysis import spatial_entropy_rates
        >>> spatial_entropy_rates([[0, 1] * 50, [0, 0] * 50], 2).tolist()
        [0.0, 0.0]
    """
    rows = np.atleast_2d(np.asarray(configurations))
    return np.asarray([entropy_rate(row, max_m, base=base) for row in rows])


@dataclass(frozen=True)
class Subshift:
    """Subshift of finite type: the configurations in which no forbidden pattern appears around any site.

    A pattern is a tuple of values in the order of a neighbourhood. It appears at a site when the cells of the
    neighbourhood of the site hold its values, and patterns that reach outside a space are not checked, so a lattice
    with fixed edges holds the words of a window.

    Args:
        neighborhood:
            The neighbourhood of offsets on a lattice, or the memory set on a group space, that patterns cover.

        forbidden:
            The forbidden patterns.

        states:
            The number of values of a cell.

    Examples:
        >>> from skcomplex.core.analysis import Subshift
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> golden = Subshift(Neighborhood(((0,), (1,))), [(1, 1)])
        >>> golden.count(LatticeSpace(6, boundary='fixed'))
        21
    """

    neighborhood: Reach
    forbidden: Sequence[tuple[int, ...]]
    states: int = 2

    def _forbidden_codes(self) -> Codes:
        """Return the codes of the forbidden patterns."""
        patterns = np.asarray([tuple(pattern) for pattern in self.forbidden], dtype=np.int64).reshape(
            -1,
            len(self.neighborhood),
        )
        return _codes(patterns, self.states)

    def _complete(self, space: Sites) -> tuple[Indices, npt.NDArray[np.bool_]]:
        """Return the neighbour table of a space and the sites whose patterns lie inside it."""
        table = space.index_table(self.neighborhood)  # type: ignore[arg-type]
        return table, (table < space.size).all(axis=1)

    def violations(self, configuration: npt.ArrayLike, space: Sites) -> npt.NDArray[np.bool_]:
        """Return the sites at which a forbidden pattern appears.

        Args:
            configuration:
                The values of the sites.

            space:
                The lattice or the group space.

        Returns:
            violations:
                One flag per site.
        """
        table, complete = self._complete(space)
        values = np.append(np.asarray(configuration, dtype=np.int64), 0)
        codes = _codes(values[table], self.states)
        return complete & np.isin(codes, self._forbidden_codes())

    def is_admissible(self, configuration: npt.ArrayLike, space: Sites) -> bool:
        """Check whether no forbidden pattern appears in a configuration.

        Args:
            configuration:
                The values of the sites.

            space:
                The lattice or the group space.

        Returns:
            admissible:
                Whether the configuration belongs to the subshift.
        """
        return not self.violations(configuration, space).any()

    def count(self, space: Sites) -> int:
        """Return the number of admissible configurations of a space, by dynamic programming over its sites.

        The sites are assigned in order, and the counts of completions are shared between assignments that agree on
        the earlier sites that later patterns read, so a window of a one-dimensional subshift costs linear time.

        Args:
            space:
                The lattice or the group space, a window of the subshift.

        Returns:
            count:
                The number of configurations without a forbidden pattern.
        """
        table, complete = self._complete(space)
        forbidden = {tuple(int(item) for item in pattern) for pattern in self.forbidden}
        checks: list[list[int]] = [[] for _ in range(space.size)]
        for site in np.flatnonzero(complete):
            checks[int(table[site].max())].append(int(site))
        needed = [
            tuple(
                sorted(
                    {int(cell) for later in checks[site:] for anchor in later for cell in table[anchor] if cell < site},
                ),
            )
            for site in range(space.size + 1)
        ]
        values = [0] * space.size
        memo: dict[tuple[int, tuple[int, ...]], int] = {}

        def extend(site: int) -> int:
            """Return the number of admissible completions of the values before a site."""
            if site == space.size:
                return 1
            key = (site, tuple(values[cell] for cell in needed[site]))
            if key not in memo:
                total = 0
                for value in range(self.states):
                    values[site] = value
                    if all(tuple(values[cell] for cell in table[anchor]) not in forbidden for anchor in checks[site]):
                        total += extend(site + 1)
                memo[key] = total
            return memo[key]

        return extend(0)

    def sample(
        self,
        space: Sites,
        sweeps: int = 100,
        *,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> Codes:
        """Return an admissible configuration drawn by a Markov chain of single-cell moves that keep it admissible.

        Args:
            space:
                The lattice or the group space.

            sweeps:
                The number of sweeps, each proposing a random value at every site in a random order.

            initial:
                The admissible initial configuration, or `None` for every cell at 0.

            random_state:
                The seed or generator of the moves.

        Returns:
            configuration:
                The admissible configuration.

        Raises:
            AnalysisError:
                The initial configuration is not admissible.
        """
        rng = np.random.default_rng(random_state)
        values = np.zeros(space.size, dtype=np.int64) if initial is None else np.asarray(initial, dtype=np.int64).copy()
        if not self.is_admissible(values, space):
            error_msg = 'The initial configuration holds a forbidden pattern. Give an admissible one.'
            raise AnalysisError(error_msg)
        table, complete = self._complete(space)
        touching: list[list[int]] = [[] for _ in range(space.size)]
        for anchor in np.flatnonzero(complete):
            for cell in set(table[anchor].tolist()):
                touching[cell].append(int(anchor))
        forbidden = {tuple(int(item) for item in pattern) for pattern in self.forbidden}
        for _ in range(sweeps):
            for site in rng.permutation(space.size):
                previous = values[site]
                values[site] = rng.integers(self.states)
                if any(tuple(values[table[anchor]].tolist()) in forbidden for anchor in touching[site]):
                    values[site] = previous
        return values


def configuration_entropy(
    subject: Subshift | Automaton,
    windows: Sequence[Sites],
    *,
    base: float = 2.0,
) -> npt.NDArray[np.float64]:
    """Return the entropy `log |X_F| / |F|` of the configurations of a subshift or of the image of an automaton.

    `X_F` is the set of admissible configurations of a window `F`, or the set of images on the sites of `F` whose
    cells lie inside it. Along a Følner sequence of windows the values approach the topological entropy.

    Args:
        subject:
            The subshift, or the automaton whose rule and neighbourhood are applied on every window.

        windows:
            The windows, such as lattices with fixed edges of growing size.

        base:
            The base of the logarithm.

    Returns:
        entropies:
            One entropy per window.

    Examples:
        >>> from skcomplex.core.analysis import Subshift, configuration_entropy
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> golden = Subshift(Neighborhood(((0,), (1,))), [(1, 1)])
        >>> configuration_entropy(golden, [LatticeSpace(20, boundary='fixed')]).round(3).tolist()
        [0.706]
    """
    entropies = []
    for window in windows:
        if isinstance(subject, Subshift):
            entropies.append(np.log(subject.count(window)) / window.size)
            continue
        automaton = Automaton(window, subject.neighborhood, subject.rule, subject.states)
        interior = np.flatnonzero((automaton.table() < window.size).all(axis=1))
        images = automaton.step(_configurations(automaton.states, window.size))[:, interior]
        count = np.unique(_codes(images, automaton.states)).size
        entropies.append(np.log(count) / max(interior.size, 1))
    return np.asarray(entropies) / np.log(base)


def _field_of(field: str) -> Any:  # noqa: ANN401
    """Return the Sage field of `'GF(p)'` or `'QQ'`."""
    if field == 'QQ':
        return QQ
    text = field.replace(' ', '')
    if text.startswith('GF(') and text.endswith(')') and text[3:-1].isdigit():
        return GF(int(text[3:-1]))
    error_msg = f"Unknown field {field!r}. Use 'QQ' or 'GF(p)' for a prime p."
    raise AnalysisError(error_msg)


@dataclass(frozen=True)
class LinearAutomaton:
    """Linear cellular automaton, the element `sum_s a_s s` of the group ring `K[G]` acting on configurations.

    The value at the site `g` becomes `sum_s a_s x(g s)` over the field, for every element `s` of the neighbourhood
    with its coefficient `a_s`. On a finite universe it is a square matrix, and injectivity, surjectivity, and
    pre-injectivity all hold exactly when the matrix is invertible. On `Z^d` it is a Laurent polynomial, invertible
    when it is a monomial and surjective when it is not zero.

    Args:
        space:
            The periodic lattice or the group space of a finite group or quotient.

        neighborhood:
            The neighbourhood of offsets on a lattice, or the memory set on a group space.

        coefficients:
            The coefficient of every element of the neighbourhood, integers or rationals.

        field:
            `'GF(p)'` for the field of a prime number `p` of elements, or `'QQ'` for the rationals.

    Examples:
        >>> from skcomplex.core.analysis import LinearAutomaton
        >>> from skcomplex.core.space import LatticeSpace, Neighborhood
        >>> rule90 = LinearAutomaton(LatticeSpace(6), Neighborhood(((-1,), (1,))), (1, 1), 'GF(2)')
        >>> rule90.step([0, 0, 1, 0, 0, 0]).tolist()
        [0, 1, 0, 1, 0, 0]
        >>> rule90.is_invertible(), rule90.is_surjective(infinite=True)
        (False, True)
    """

    space: Sites
    neighborhood: Reach
    coefficients: Sequence[float]
    field: str = 'GF(2)'

    def step(self, configuration: npt.ArrayLike) -> npt.NDArray[Any]:
        """Return the image of a configuration.

        Args:
            configuration:
                The values of the sites.

        Returns:
            image:
                The next values, reduced modulo the field size over a prime field.
        """
        table = self.space.index_table(self.neighborhood)  # type: ignore[arg-type]
        values = np.append(np.asarray(configuration), getattr(self.space, 'fill', 0))
        image = values[table] @ np.asarray(self.coefficients)
        characteristic = int(_field_of(self.field).characteristic())
        return np.rint(image).astype(np.int64) % characteristic if characteristic else image

    def matrix(self) -> Any:  # noqa: ANN401
        """Return the matrix of the automaton over the field, whose row `g` gives the image at `g`.

        Returns:
            matrix:
                The Sage matrix.

        Raises:
            AnalysisError:
                A site reads a cell outside the space.
        """
        table = self.space.index_table(self.neighborhood)  # type: ignore[arg-type]
        if (table >= self.space.size).any():
            error_msg = 'A linear automaton has a matrix on a periodic lattice or a finite group, not on a ball.'
            raise AnalysisError(error_msg)
        field = _field_of(self.field)
        entries = matrix(field, self.space.size, self.space.size)
        for site, row in enumerate(table.tolist()):
            for neighbour, coefficient in zip(row, self.coefficients, strict=True):
                entries[site, neighbour] += field(QQ(coefficient))
        return entries

    def laurent_polynomial(self) -> Any:  # noqa: ANN401
        """Return the Laurent polynomial `sum a_s x^s` of an automaton on a lattice, its symbol on `Z^d`.

        Returns:
            polynomial:
                The Sage Laurent polynomial in the variables `x0` to `x(d-1)`.

        Raises:
            AnalysisError:
                The neighbourhood is not one of offsets.
        """
        if not isinstance(self.neighborhood, Neighborhood):
            error_msg = 'A Laurent polynomial describes a linear automaton on a lattice, with offsets.'
            raise AnalysisError(error_msg)
        field = _field_of(self.field)
        n_dim = len(self.neighborhood.offsets[0])
        ring = LaurentPolynomialRing(field, [f'x{axis}' for axis in range(n_dim)])
        variables = ring.gens()
        terms = (
            field(QQ(coefficient)) * np.prod([variable**step for variable, step in zip(variables, offset, strict=True)])
            for offset, coefficient in zip(self.neighborhood.offsets, self.coefficients, strict=True)
        )
        return sum(terms, ring.zero())

    def is_invertible(self, *, infinite: bool = False) -> bool:
        """Check whether the automaton is invertible, on its finite universe or on `Z^d`.

        Args:
            infinite:
                Whether to decide on `Z^d` from the Laurent polynomial instead of on the space.

        Returns:
            invertible:
                Whether the matrix is invertible, or whether the polynomial is a monomial.
        """
        if infinite:
            return len(self.laurent_polynomial().exponents()) == 1
        return bool(self.matrix().is_invertible())

    def is_surjective(self, *, infinite: bool = False) -> bool:
        """Check whether the automaton is surjective, on its finite universe or on `Z^d`.

        Args:
            infinite:
                Whether to decide on `Z^d` from the Laurent polynomial instead of on the space.

        Returns:
            surjective:
                Whether the matrix is invertible, or whether the polynomial is not zero.
        """
        if infinite:
            return not self.laurent_polynomial().is_zero()
        return self.is_invertible()

    def is_pre_injective(self, *, infinite: bool = False) -> bool:
        """Check whether the automaton is pre-injective, which for linear automata over fields means surjective.

        Args:
            infinite:
                Whether to decide on `Z^d` from the Laurent polynomial instead of on the space.

        Returns:
            pre_injective:
                Whether no nonzero configuration of finite support maps to zero.
        """
        return self.is_surjective(infinite=infinite)
