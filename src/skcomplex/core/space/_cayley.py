"""Define group spaces, the Cayley graphs of finitely generated groups on balls or finite quotients, and memory sets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ..base import Requires, SpaceKind, SpaceTooLargeError, ValidationError
from ._groups import FinitelyGeneratedGroup, Word, WordProblem, _column, _format_word, _letters, parse_word
from ._spaces import Space

MAX_SITES = 100_000
LACKING_OPTIONS = frozenset({'regular-geometry', 'image'})

type Table = npt.NDArray[np.int64]


def _bound_sizes(group: FinitelyGeneratedGroup, radius: int, limit: int) -> list[int]:
    """Return upper bounds of the ball sizes up to a radius, or until one exceeds a limit, without GAP.

    The bound counts the words whose consecutive letters never multiply to the identity or to a single letter, after
    merging letters that are equal, as the short keys of the group decide.
    """
    identity = group.short_key(())
    distinct: dict[Hashable, int] = {}
    for letter in _letters(len(group.generators)):
        key = group.short_key((letter,))
        if key != identity and key not in distinct:
            distinct[key] = letter
    letters = list(distinct.values())
    singles = {identity, *distinct}
    allowed = np.array([[group.short_key((first, second)) not in singles for second in letters] for first in letters])
    counts = [1] * len(letters)
    sizes = [1]
    for depth in range(1, radius + 1):
        if depth > 1:
            counts = [
                sum(count for count, keep in zip(counts, allowed[:, column], strict=True) if keep)
                for column in range(len(letters))
            ]
        sizes.append(sizes[-1] + sum(counts))
        if sizes[-1] > limit:
            break
    return sizes


def _find(
    problem: WordProblem,
    buckets: dict[Hashable, list[int]],
    elements: list[Any],
    element: Any,  # noqa: ANN401
) -> int | None:
    """Return the site of an element among the known ones, or `None`."""
    for site in buckets.get(problem.key(element), []):
        if problem.same(elements[site], element):
            return site
    return None


@dataclass(frozen=True)
class _Ball:
    """Sites of a group space, with the word of every site, its neighbours by generator, and the ball sizes."""

    words: tuple[Word, ...]
    table: Table
    sizes: tuple[int, ...]


def _enumerate(problem: WordProblem, n_generators: int, radius: int | None, limit: int) -> _Ball:
    """Return the sites of a ball of a radius, or of the whole group without one, by breadth-first search."""
    letters = _letters(n_generators)
    elements: list[Any] = [problem.identity()]
    words: list[Word] = [()]
    buckets: dict[Hashable, list[int]] = {problem.key(elements[0]): [0]}
    rows = [[-1] * len(letters)]
    frontier, sizes, depth = [0], [1], 0
    while frontier:
        grows = radius is None or depth < radius
        found: list[int] = []
        for site in frontier:
            for column, letter in enumerate(letters):
                if rows[site][column] != -1:
                    continue
                element = problem.multiply(elements[site], letter)
                neighbour = _find(problem, buckets, elements, element)
                if neighbour is None and not grows:
                    continue
                if neighbour is None:
                    if len(elements) >= limit:
                        error_msg = (
                            f'The group space has more than {limit} sites. Ask for a smaller ball or quotient, or '
                            'raise max_sites.'
                        )
                        raise SpaceTooLargeError(error_msg)
                    neighbour = len(elements)
                    elements.append(element)
                    words.append((*words[site], letter))
                    rows.append([-1] * len(letters))
                    buckets.setdefault(problem.key(element), []).append(neighbour)
                    found.append(neighbour)
                rows[site][column] = neighbour
                rows[neighbour][_column(-letter)] = site
        if not grows:
            break
        if found or radius is not None:
            sizes.append(len(elements))
        frontier, depth = found, depth + 1
    return _Ball(tuple(words), np.asarray(rows, dtype=np.int64), tuple(sizes))


@dataclass(frozen=True)
class Memory:
    """Ordered memory set of a cellular automaton on a group space: the group elements whose values a rule reads.

    A cell at the element `g` reads the cell at `g s` for every element `s` of the memory set, in its order, so the
    neighbourhood keeps its orientation. Elements are words in the generators, such as `'a'`, `'a^-1'`, or `'a*b'`,
    and `'1'` is the identity, the cell itself.

    Args:
        elements:
            The words of the memory set.

    Examples:
        >>> from skcomplex.core.space import Memory
        >>> len(Memory(('1', 'a', 'a^-1')))
        3
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.LATTICE})

    elements: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'elements', tuple(str(element) for element in self.elements))

    def __len__(self) -> int:
        """Return the number of elements."""
        return len(self.elements)

    @classmethod
    def generators(
        cls,
        group: FinitelyGeneratedGroup,
        *,
        center: bool = True,
        inverses: bool = True,
    ) -> 'Memory':
        """Return the memory set of the generators of a group, after the identity and with their inverses if asked.

        Args:
            group:
                The group.

            center:
                Whether the identity comes first.

            inverses:
                Whether every generator is followed by its inverse.

        Returns:
            memory:
                The memory set.

        Examples:
            >>> from skcomplex.core.space import Memory, finitely_presented_group
            >>> Memory.generators(finitely_presented_group(['a', 'b'])).elements
            ('1', 'a', 'a^-1', 'b', 'b^-1')
        """
        elements = [word for name in group.generators for word in ((name, f'{name}^-1') if inverses else (name,))]
        return cls(('1', *elements) if center else tuple(elements))


class GroupSpace(Space):
    """Cayley graph of a finitely generated group, on a ball of words or on a finite quotient, whose sites hold cells.

    Sites are the distinct elements of a ball of a radius in the generators and their inverses, or of a finite
    quotient, numbered by word length and then by order of discovery, so a ball is a prefix of every larger ball. A
    cell at the site `g` reads the site `g s` for every element `s` of a memory set, and a neighbour outside a ball, or
    reached through a site outside it, reads the value `fill`. Without a ball or a quotient the group must be finite,
    and the sites are all its elements. The space is built on first use, and running a rule reads only its neighbour
    table.

    Args:
        group:
            The group, from `finitely_presented_group`, `automaton_group`, `permutation_group`, or `matrix_group`.

        ball:
            The radius of the ball, or `None` for a finite quotient or a finite group.

        quotient:
            A level of the tree of an automaton group, whose action gives the quotient, or a finite group whose
            generators are the images of the generators of `group`, or `None`.

        fill:
            The value read outside a ball.

        max_sites:
            The largest number of sites.

    Raises:
        ValidationError:
            The space has both a ball and a quotient, or the ball has a negative radius.

    Examples:
        >>> from skcomplex.core.space import GroupSpace, permutation_group
        >>> space = GroupSpace(permutation_group([[1, 0, 2], [1, 2, 0]]), ball=1)
        >>> space.size, space.words
        (4, ['1', 'a', 'b', 'b^-1'])
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.LATTICE
    lacks: ClassVar[frozenset[str]] = LACKING_OPTIONS

    n_dim = 1

    def __init__(
        self,
        group: FinitelyGeneratedGroup,
        *,
        ball: int | None = None,
        quotient: int | FinitelyGeneratedGroup | None = None,
        fill: float = 0.0,
        max_sites: int = MAX_SITES,
    ) -> None:
        self.group = group
        self.ball = ball
        self.quotient = quotient
        self.fill = fill
        self.max_sites = max_sites
        self._built: _Ball | None = None
        if ball is not None and quotient is not None:
            error_msg = 'A group space is either a ball or a quotient. Give ball or quotient, not both.'
            raise ValidationError(error_msg)
        if ball is not None and ball < 0:
            error_msg = f'The radius of a ball is a nonnegative integer, but it is {ball}.'
            raise ValidationError(error_msg)

    def _sites_group(self) -> FinitelyGeneratedGroup:
        """Return the group whose elements are the sites, the quotient when there is one."""
        if self.quotient is None:
            return self.group
        if isinstance(self.quotient, int):
            return self.group.level_quotient(self.quotient)
        if len(self.quotient.generators) != len(self.group.generators):
            error_msg = (
                f'The quotient has {len(self.quotient.generators)} generators, but the group has '
                f'{len(self.group.generators)}. Give one image per generator.'
            )
            raise ValidationError(error_msg)
        problem = self.quotient.word_problem()
        identity = problem.identity()
        for relator in getattr(self.group, 'relators', ()):
            element = identity
            for letter in relator:
                element = problem.multiply(element, letter)
            if problem.key(element) != problem.key(identity) or not problem.same(element, identity):
                word = _format_word(relator, self.group.generators)
                error_msg = f'The relator {word} is not the identity in the quotient, so it is not a quotient.'
                raise ValidationError(error_msg)
        return self.quotient

    def _ball(self) -> _Ball:
        """Return the sites, building them on first use."""
        if self._built is None:
            self.check()
            group = self._sites_group()
            self._built = _enumerate(group.word_problem(), len(group.generators), self.ball, self.max_sites)
        return self._built

    def check(self) -> None:
        """Raise an error when the ball would exceed the site limit, from a bound that needs no GAP.

        Raises:
            SpaceTooLargeError:
                The bound of the ball size exceeds `max_sites`.
        """
        if self.ball is None:
            return
        sizes = _bound_sizes(self.group, self.ball, self.max_sites)
        if sizes[-1] <= self.max_sites:
            return
        fits = len(sizes) - 2
        error_msg = (
            f'The ball of radius {self.ball} exceeds the limit of {self.max_sites} sites. The largest radius that fits '
            f'is {fits}, with at most {sizes[fits]} sites.'
        )
        raise SpaceTooLargeError(error_msg)

    @property
    def size(self) -> int:
        """Number of sites."""
        return len(self._ball().words)

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Number of sites, as the shape of the arrays of cells."""
        return (self.size,)

    @property
    def words(self) -> list[str]:
        """Shortest word of every site, in site order."""
        generators = self._sites_group().generators
        return [_format_word(word, generators) for word in self._ball().words]

    @property
    def lengths(self) -> npt.NDArray[np.int64]:
        """Word length of every site, its distance from the identity in the Cayley graph."""
        return np.asarray([len(word) for word in self._ball().words], dtype=np.int64)

    @property
    def generator_table(self) -> Table:
        """Site of `g s` for every site `g` and letter `s`, in the order `a, a^-1, b, b^-1, ...`, and -1 outside."""
        return self._ball().table

    @property
    def edges(self) -> list[tuple[int, int, int]]:
        """Edge from every site `g` to `g s` for every generator `s` inside the space, with the generator's number."""
        table = self._ball().table
        return [
            (site, int(table[site, 2 * generator]), generator)
            for generator in range(table.shape[1] // 2)
            for site in range(table.shape[0])
            if table[site, 2 * generator] >= 0
        ]

    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """Position of every site on rings by word length, spread evenly around each ring.

        Examples:
            >>> from skcomplex.core.space import GroupSpace, permutation_group
            >>> GroupSpace(permutation_group([[1, 0]]), ball=1).positions.round(3).tolist()
            [[0.0, 0.0], [1.0, 0.0]]
        """
        lengths = self.lengths
        positions = np.zeros((lengths.size, 2))
        for length in np.unique(lengths):
            sites = np.flatnonzero(lengths == length)
            angles = 2 * np.pi * np.arange(sites.size) / sites.size
            positions[sites] = length * np.column_stack([np.cos(angles), np.sin(angles)])
        return positions

    def ball_sizes(self) -> list[int]:
        """Return the number of sites within every radius, from 0 to the radius of the ball or of the quotient.

        Returns:
            sizes:
                The sizes of the balls.

        Examples:
            >>> from skcomplex.core.space import GroupSpace, permutation_group
            >>> GroupSpace(permutation_group([[1, 0, 2], [1, 2, 0]]), ball=3).ball_sizes()
            [1, 4, 6, 6]
        """
        return list(self._ball().sizes)

    def index_table(self, neighborhood: Memory) -> npt.NDArray[np.intp]:
        """Return the site `g s` of every site `g` and element `s` of a memory set, with `size` outside the space.

        Args:
            neighborhood:
                The memory set.

        Returns:
            table:
                One row per site and one column per element.

        Raises:
            ValidationError:
                The neighbourhood is not a memory set, or one of its words has a factor that is not a generator.

        Examples:
            >>> from skcomplex.core.space import GroupSpace, Memory, permutation_group
            >>> space = GroupSpace(permutation_group([[1, 2, 0]]))
            >>> space.index_table(Memory(('1', 'a'))).tolist()
            [[0, 1], [1, 2], [2, 0]]
        """
        if not isinstance(neighborhood, Memory):
            error_msg = f'A group space reads a memory set of group elements, but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
        ball = self._ball()
        generators = self._sites_group().generators
        size = len(ball.words)
        table = np.empty((size, len(neighborhood)), dtype=np.intp)
        for position, element in enumerate(neighborhood.elements):
            sites = np.arange(size)
            for letter in parse_word(element, generators):
                inside = sites >= 0
                sites = np.where(inside, ball.table[np.where(inside, sites, 0), _column(letter)], -1)
            table[:, position] = np.where(sites >= 0, sites, size)
        return table
