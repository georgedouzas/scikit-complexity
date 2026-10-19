"""Define lattices, neighbourhoods, local rules, relaxation, and Monte Carlo laws of lattice models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR
from sage.symbolic.function import BuiltinFunction

from ..base import Expression, SymbolicVector, vector
from ._spaces import Space

BOUNDARIES = ('periodic', 'fixed', 'reflecting')
SCHEDULES = ('synchronous', 'random-sequential', 'random-site', 'checkerboard')
SAMPLERS = ('metropolis', 'glauber', 'heat-bath')
MOVES = ('flip', 'random-value', 'rotate')

type Cell = tuple[int, ...]
type Offset = tuple[int, ...]


class LatticeSpace(Space):
    """Box of the integer lattice `Z^d` whose sites hold the members of lattice groups.

    Sites are numbered in row-major order, so site `(i, j)` of a lattice of shape `(n, m)` is member `i * m + j`.

    Args:
        shape:
            The number of sites along every dimension.

        boundary:
            `'periodic'` to join opposite edges, `'fixed'` to read the value `fill` outside the box, or `'reflecting'`
            to read the mirror image of the box outside it.

        fill:
            The value read outside a lattice with fixed edges.

        spacing:
            The distance between neighbouring sites, or one distance per dimension.

    Examples:
        >>> from skcomplex.core.model import LatticeSpace
        >>> space = LatticeSpace((10, 20))
        >>> space.size, space.n_dim
        (200, 2)
    """

    def __init__(
        self,
        shape: Sequence[int] | int,
        *,
        boundary: str = 'periodic',
        fill: float = 0.0,
        spacing: float | Sequence[float] = 1.0,
    ) -> None:
        self.shape = shape
        self.boundary = boundary
        self.fill = fill
        self.spacing = spacing

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Number of sites along every dimension."""
        return (int(self.shape),) if np.ndim(self.shape) == 0 else tuple(int(size) for size in self.shape)  # type: ignore[arg-type, union-attr]

    @property
    def spacings(self) -> tuple[float, ...]:
        """Distance between neighbouring sites along every dimension."""
        if np.ndim(self.spacing) == 0:
            return (float(self.spacing),) * self.n_dim  # type: ignore[arg-type]
        return tuple(float(item) for item in self.spacing)  # type: ignore[union-attr]

    @property
    def n_dim(self) -> int:  # type: ignore[override]
        """Number of dimensions."""
        return len(self.dimensions)

    @property
    def size(self) -> int:
        """Number of sites."""
        return int(np.prod(self.dimensions))

    def check(self) -> None:
        """Raise an error when the shape or the boundary is not valid.

        Raises:
            ValueError:
                The shape has a size that is not a positive integer, or the boundary is unknown.
        """
        if not self.dimensions or any(size < 1 for size in self.dimensions):
            error_msg = (
                f'A lattice has a positive number of sites along every dimension, but its shape is {self.shape}.'
            )
            raise ValueError(error_msg)
        if self.boundary not in BOUNDARIES:
            error_msg = f'Unknown lattice boundary {self.boundary!r}. Use one of {", ".join(BOUNDARIES)}.'
            raise ValueError(error_msg)

    def neighbor(self, cell: Cell, offset: Offset) -> Cell | None:
        """Return the site at an offset from a site, or `None` outside a lattice with fixed edges.

        Args:
            cell:
                The coordinates of the site.

            offset:
                The offset along every dimension.

        Returns:
            neighbor:
                The coordinates of the neighbouring site, wrapped or mirrored at the edges.
        """
        coordinates = []
        for position, step, size in zip(cell, offset, self.dimensions, strict=True):
            value = position + step
            if self.boundary == 'periodic':
                value %= size
            elif self.boundary == 'reflecting':
                period = 2 * size
                value %= period
                value = value if value < size else period - 1 - value
            elif not 0 <= value < size:
                return None
            coordinates.append(int(value))
        return tuple(coordinates)

    def neighbors(self, cell: Cell, neighborhood: 'Neighborhood') -> list[Cell | None]:
        """Return the sites of a neighbourhood of a site, in the neighbourhood's order.

        Args:
            cell:
                The coordinates of the site.

            neighborhood:
                The neighbourhood.

        Returns:
            neighbors:
                The coordinates of every neighbouring site, or `None` outside a lattice with fixed edges.

        Examples:
            >>> from skcomplex.core.model import LatticeSpace, Neighborhood
            >>> LatticeSpace((10, 10)).neighbors((0, 0), Neighborhood.moore())
            [(9, 9), (9, 0), (9, 1), (0, 9), (0, 1), (1, 9), (1, 0), (1, 1)]
        """
        return [self.neighbor(cell, offset) for offset in neighborhood.offsets]

    def index_table(self, neighborhood: 'Neighborhood') -> npt.NDArray[np.intp]:
        """Return the member index of every neighbour of every site, with `size` for sites outside fixed edges.

        Args:
            neighborhood:
                The neighbourhood.

        Returns:
            table:
                One row per site and one column per offset.
        """
        self.check()
        if neighborhood.offsets and len(neighborhood.offsets[0]) != self.n_dim:
            error_msg = (
                f'The neighbourhood has offsets of dimension {len(neighborhood.offsets[0])}, but the lattice has '
                f'{self.n_dim} dimensions.'
            )
            raise ValueError(error_msg)
        coordinates = np.indices(self.dimensions).reshape(self.n_dim, -1)
        sizes = np.asarray(self.dimensions)[:, None]
        table = np.empty((self.size, len(neighborhood.offsets)), dtype=np.intp)
        for column, offset in enumerate(neighborhood.offsets):
            shifted = coordinates + np.asarray(offset)[:, None]
            outside = np.zeros(self.size, dtype=bool)
            if self.boundary == 'periodic':
                shifted %= sizes
            elif self.boundary == 'reflecting':
                shifted %= 2 * sizes
                shifted = np.where(shifted < sizes, shifted, 2 * sizes - 1 - shifted)
            else:
                outside = ((shifted < 0) | (shifted >= sizes)).any(axis=0)
                shifted = np.clip(shifted, 0, sizes - 1)
            flat = np.ravel_multi_index(tuple(shifted), self.dimensions)
            table[:, column] = np.where(outside, self.size, flat)
        return table


@dataclass(frozen=True)
class Neighborhood:
    """Ordered offsets of the sites a local rule reads, with the site itself as the zero offset when included.

    Args:
        offsets:
            The offsets, one tuple of integers per neighbour, in the order rules receive their values.

    Examples:
        >>> from skcomplex.core.model import Neighborhood
        >>> Neighborhood.von_neumann().offsets
        ((-1, 0), (0, -1), (0, 1), (1, 0))
        >>> len(Neighborhood.moore(center=True))
        9
    """

    offsets: tuple[Offset, ...]

    def __post_init__(self) -> None:
        offsets = tuple(tuple(int(step) for step in np.atleast_1d(offset)) for offset in self.offsets)
        if len({len(offset) for offset in offsets}) > 1:
            error_msg = 'Every offset of a neighbourhood has the same number of dimensions.'
            raise ValueError(error_msg)
        object.__setattr__(self, 'offsets', offsets)

    def __len__(self) -> int:
        """Return the number of offsets."""
        return len(self.offsets)

    @classmethod
    def _of(cls, radius: int, n_dim: int, keep: Callable[[Offset], bool], center: bool) -> 'Neighborhood':
        """Return the offsets within a radius that a condition keeps, in lexicographic order."""
        offsets = [
            offset
            for offset in itertools.product(range(-radius, radius + 1), repeat=n_dim)
            if any(offset) and keep(offset)
        ]
        return cls(((0,) * n_dim, *offsets) if center else tuple(offsets))

    @classmethod
    def moore(cls, radius: int = 1, *, n_dim: int = 2, center: bool = False) -> 'Neighborhood':
        """Return the sites within a radius in every dimension, the 8 surrounding cells in 2D for radius 1.

        Args:
            radius:
                The largest offset along any dimension.

            n_dim:
                The number of dimensions.

            center:
                Whether the site itself comes first.

        Returns:
            neighborhood:
                The neighbourhood.
        """
        return cls._of(radius, n_dim, lambda _: True, center)

    @classmethod
    def von_neumann(cls, radius: int = 1, *, n_dim: int = 2, center: bool = False) -> 'Neighborhood':
        """Return the sites within a Manhattan distance, the 4 adjacent cells in 2D for radius 1.

        Args:
            radius:
                The largest Manhattan distance.

            n_dim:
                The number of dimensions.

            center:
                Whether the site itself comes first.

        Returns:
            neighborhood:
                The neighbourhood.
        """
        return cls._of(radius, n_dim, lambda offset: sum(map(abs, offset)) <= radius, center)

    @classmethod
    def radius(cls, radius: int = 1, *, n_dim: int = 1) -> 'Neighborhood':
        """Return the sites within a radius, the site itself included in place, as one-dimensional automata read them.

        For radius 1 in one dimension the order is left, centre, right, the order of Wolfram's rule numbers.

        Args:
            radius:
                The largest offset along any dimension.

            n_dim:
                The number of dimensions.

        Returns:
            neighborhood:
                The neighbourhood.

        Examples:
            >>> from skcomplex.core.model import Neighborhood
            >>> Neighborhood.radius().offsets
            ((-1,), (0,), (1,))
        """
        return cls(tuple(itertools.product(range(-radius, radius + 1), repeat=n_dim)))


@dataclass(frozen=True)
class Table:
    """Rule given by the next value of every neighbourhood pattern of integer states `0` to `states - 1`.

    Args:
        mapping:
            A mapping from pattern, a tuple of values in neighbourhood order, to the next value.

        default:
            The next value of patterns the mapping leaves out.

        states:
            The number of state values, or `None` for one more than the largest value in the patterns.

    Examples:
        >>> from skcomplex.core.model import Table
        >>> Table({(1, 1): 1}).states
        2
    """

    mapping: Mapping[tuple[int, ...], float]
    default: float = 0.0
    states: int | None = None

    def __post_init__(self) -> None:
        if self.states is None:
            largest = max((value for pattern in self.mapping for value in pattern), default=0)
            object.__setattr__(self, 'states', max(int(largest) + 1, 2))


@dataclass(frozen=True)
class Wolfram:
    """Elementary cellular automaton by its Wolfram number, on the left, centre, and right cells.

    Args:
        number:
            The rule number from 0 to 255.

    Examples:
        >>> from skcomplex.core.model import Wolfram
        >>> Wolfram(30).table().mapping[(1, 0, 0)]
        1
    """

    number: int

    def table(self) -> Table:
        """Return the rule as a table of the eight patterns.

        Returns:
            table:
                The table.
        """
        if not 0 <= self.number <= 255:  # noqa: PLR2004
            error_msg = f'A Wolfram rule number is between 0 and 255, but it is {self.number}.'
            raise ValueError(error_msg)
        patterns = itertools.product((0, 1), repeat=3)
        return Table(
            {pattern: (self.number >> (4 * pattern[0] + 2 * pattern[1] + pattern[2])) & 1 for pattern in patterns},
        )


@dataclass(frozen=True)
class Totalistic:
    """Rule given by the next value of every sum of the neighbourhood values.

    Args:
        rule:
            A mapping from sum to next value, or a function of the sums that returns the next values.

        default:
            The next value of sums a mapping leaves out.

        states:
            The number of state values `0` to `states - 1`, which bounds the sums.

    Examples:
        >>> from skcomplex.core.model import Totalistic
        >>> Totalistic(lambda total: total > 4).states
        2
    """

    rule: Mapping[int, float] | Callable[[Any], Any]
    default: float = 0.0
    states: int = 2


@dataclass(frozen=True)
class OuterTotalistic:
    """Rule given by the next value of every pair of the cell's value and the sum of its neighbours' values.

    Args:
        rule:
            A mapping from `(value, sum)` to next value, or a function of the values and the sums.

        default:
            The next value of pairs a mapping leaves out.

        states:
            The number of state values `0` to `states - 1`, which bounds the sums.

    Examples:
        >>> from skcomplex.core.model import OuterTotalistic
        >>> life = OuterTotalistic.life()
        >>> life.rule[(0, 3)], life.rule[(1, 2)], (1, 4) in life.rule
        (1, 1, False)
    """

    rule: Mapping[tuple[int, int], float] | Callable[[Any, Any], Any]
    default: float = 0.0
    states: int = 2

    @classmethod
    def life(cls, born: Iterable[int] = (3,), survive: Iterable[int] = (2, 3)) -> 'OuterTotalistic':
        """Return a life-like rule in the notation `B3/S23` of Conway's Game of Life.

        Args:
            born:
                The numbers of live neighbours that bring a dead cell to life.

            survive:
                The numbers of live neighbours that keep a live cell alive.

        Returns:
            rule:
                The rule.
        """
        return cls({**{(0, count): 1 for count in born}, **{(1, count): 1 for count in survive}})


type RuleSpec = Table | Wolfram | Totalistic | OuterTotalistic | Callable[..., Any]


def _targets(target: object) -> tuple[Expression, ...]:
    """Return the target symbols of a lattice law as a tuple."""
    return tuple(target) if isinstance(target, tuple | list) else (target,)


@dataclass(frozen=True)
class LocalRule:
    """Law that computes the next value of lattice cells from the ordered values of their neighbourhoods.

    A function rule receives one array per neighbour of every read state, in neighbourhood order, with one value per
    site, and keyword arguments for the parameters of the component that it names, plus `random`, an array of uniform
    numbers per site, when the rule is stochastic. It returns the next values of the target, or a tuple with one array
    per target.

    Args:
        target:
            The symbol of a lattice state, or a tuple of them that the rule updates together.

        neighborhood:
            The neighbourhood whose values the rule reads.

        rule:
            A function, a `Table`, a `Wolfram` number, a `Totalistic` rule, or an `OuterTotalistic` rule.

        reads:
            The lattice states whose neighbourhoods a function rule reads, by default the targets.

        schedule:
            `'synchronous'` to update every site from the previous configuration, `'random-sequential'` to update the
            sites one at a time in a random order, `'random-site'` to update as many randomly chosen sites, or
            `'checkerboard'` to update the two sublattices of even and odd coordinate sums in turn.

        stochastic:
            Whether a function rule receives the uniform random array `random`.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import LocalRule, Neighborhood, Wolfram
        >>> LocalRule(symbol('cell'), Neighborhood.radius(), Wolfram(30)).schedule
        'synchronous'
    """

    target: Expression | tuple[Expression, ...]
    neighborhood: Neighborhood
    rule: RuleSpec
    reads: tuple[Expression, ...] | None = None
    schedule: str = 'synchronous'
    stochastic: bool = False

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r}. Use one of {", ".join(SCHEDULES)}.'
            raise ValueError(error_msg)

    @property
    def targets(self) -> tuple[Expression, ...]:
        """Target symbols."""
        return _targets(self.target)


@dataclass(frozen=True)
class Drive:
    """Law that changes a whole lattice state at once, such as dropping a grain on a random site.

    The function receives the state as an array of the lattice shape, keyword arguments for the parameters of the
    component that it names, and `random`, the simulation's random generator, and returns the new array.

    Args:
        target:
            The symbol of a lattice state.

        function:
            The function `f(values, *, random, **parameters)`.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Drive
        >>> Drive(symbol('height'), lambda values, *, random: values + 1).target
        height
    """

    target: Expression
    function: Callable[..., Any]

    @property
    def targets(self) -> tuple[Expression, ...]:
        """Target symbols."""
        return (self.target,)


@dataclass(frozen=True)
class Relaxation:
    """Slow drive and fast relaxation: drive laws once per step, then relaxation laws until the lattice is stable.

    Every relaxation that changes the lattice is recorded as an event with its size, the number of site updates, its
    duration, the number of iterations, and its area, the number of distinct sites updated. A site update is a site
    that changes in an iteration, or a site that `activity` marks as active before it, such as a toppling site.

    Args:
        drive:
            The laws applied once per step, local rules or drives.

        relax:
            The laws applied in turn until the lattice is stable.

        stable:
            A function of the arrays of the relaxation laws' targets, in lattice shape, that returns whether the
            lattice is stable.

        max_iterations:
            The largest number of relaxation iterations in one step.

        label:
            The name of the relaxation events, or `None` for the component's path.

        activity:
            A function of the targets' arrays that returns a boolean array of the active sites of an iteration, or
            `None` to count the sites that change.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import LocalRule, Neighborhood, Relaxation
        >>> height = symbol('height')
        >>> topple = LocalRule(height, Neighborhood.von_neumann(center=True), lambda z, *n: z - 4 * (z >= 4))
        >>> Relaxation([], [topple], lambda z: bool((z < 4).all())).max_iterations
        100000
    """

    drive: Sequence[LocalRule | Drive]
    relax: Sequence[LocalRule | Drive]
    stable: Callable[..., bool]
    max_iterations: int = 100_000
    label: str | None = None
    activity: Callable[..., Any] | None = None

    @property
    def targets(self) -> tuple[Expression, ...]:
        """Target symbols of the relaxation laws, in order of first appearance."""
        return tuple(dict.fromkeys(item for law in self.relax for item in law.targets))


class _Neighbor(BuiltinFunction):
    """Value of a lattice state at the `k`-th offset of a neighbourhood, `neighbor(x, k)`."""

    def __init__(self) -> None:
        super().__init__('neighbor', nargs=2)


neighbor = _Neighbor()


def neighbors(state: Expression, neighborhood: Neighborhood) -> SymbolicVector:
    """Return the values of a lattice state at the offsets of a neighbourhood, for local energies.

    Args:
        state:
            The symbol of a lattice state.

        neighborhood:
            The neighbourhood.

    Returns:
        values:
            One symbolic value per offset, in neighbourhood order.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Neighborhood, neighbors
        >>> spin = symbol('spin')
        >>> values = neighbors(spin, Neighborhood.von_neumann())
        >>> len(values), values[0]
        (4, neighbor(spin, 0))
    """
    return vector(neighbor(state, SR(index)) for index in range(len(neighborhood)))


@dataclass(frozen=True)
class Energy:
    """Local energy of a lattice site: every term of the total energy that involves the site's value.

    The energy change of a move is the difference of the local energy at the proposed and the current value, derived
    symbolically. Neighbour values are read with `neighbors`.

    Args:
        expression:
            The local energy, a symbolic expression of the site's states, its neighbours' values, and parameters.

        neighborhood:
            The neighbourhood that `neighbors` reads in the expression.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Energy, Neighborhood, neighbors
        >>> spin, J = symbol('spin'), symbol('J')
        >>> nearest = Neighborhood.von_neumann()
        >>> len(Energy(-J * spin * sum(neighbors(spin, nearest)), nearest).neighborhood)
        4
    """

    expression: object
    neighborhood: Neighborhood


@dataclass(frozen=True)
class Move:
    """Proposed change of the value of a lattice site.

    Args:
        target:
            The symbol of the lattice state the move changes.

        kind:
            `'flip'` for the negative value, `'random-value'` for a value drawn from `values`, or `'rotate'` for the
            value plus a uniform number in `[-width, width]`.

        values:
            The values a random-value move draws from.

        width:
            The largest change of a rotation.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import Move
        >>> Move.random_value(symbol('color'), range(3)).values
        (0.0, 1.0, 2.0)
    """

    target: Expression
    kind: str = 'flip'
    values: tuple[float, ...] = ()
    width: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in MOVES:
            error_msg = f'Unknown move {self.kind!r}. Use one of {", ".join(MOVES)}.'
            raise ValueError(error_msg)

    @classmethod
    def flip(cls, target: Expression) -> 'Move':
        """Return the move that changes a value to its negative, the spin flip of Ising models.

        Args:
            target:
                The symbol of the lattice state.

        Returns:
            move:
                The move.
        """
        return cls(target, 'flip')

    @classmethod
    def random_value(cls, target: Expression, values: Iterable[float]) -> 'Move':
        """Return the move that draws a new value uniformly from a set, as Potts models do.

        Args:
            target:
                The symbol of the lattice state.

            values:
                The values to draw from.

        Returns:
            move:
                The move.
        """
        return cls(target, 'random-value', tuple(float(value) for value in values))

    @classmethod
    def rotate(cls, target: Expression, width: float) -> 'Move':
        """Return the move that adds a uniform number in `[-width, width]`, as XY models do with angles.

        Args:
            target:
                The symbol of the lattice state.

            width:
                The largest change.

        Returns:
            move:
                The move.
        """
        return cls(target, 'rotate', width=float(width))


@dataclass(frozen=True)
class Sampler:
    """Acceptance of Monte Carlo moves at a temperature, with one sweep of the lattice per unit of time.

    Args:
        kind:
            `'metropolis'` to accept with probability `min(1, exp(-dE / T))`, `'glauber'` with `1 / (1 + exp(dE / T))`,
            or `'heat-bath'` to draw the new value among all values of a discrete move with Boltzmann weights.

        temperature:
            The temperature, usually a parameter symbol so that it changes live.

        schedule:
            `'checkerboard'` to update the two sublattices of a nearest-neighbour energy in turn, vectorised, or
            `'random-sequential'` to visit the sites one at a time in a random order.

    Examples:
        >>> from skcomplex.core.model import Sampler
        >>> Sampler('glauber', 2.0).schedule
        'checkerboard'
    """

    kind: str = 'metropolis'
    temperature: object = 1.0
    schedule: str = 'checkerboard'

    def __post_init__(self) -> None:
        if self.kind not in SAMPLERS:
            error_msg = f'Unknown sampler {self.kind!r}. Use one of {", ".join(SAMPLERS)}.'
            raise ValueError(error_msg)
        if self.schedule not in {'checkerboard', 'random-sequential'}:
            error_msg = f"Unknown Monte Carlo schedule {self.schedule!r}. Use 'checkerboard' or 'random-sequential'."
            raise ValueError(error_msg)


LATTICE_LAWS = (LocalRule, Drive, Relaxation)
MONTE_CARLO_LAWS = (Energy, Move, Sampler)
