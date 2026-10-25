"""Define group spaces, the Cayley graphs of finitely generated groups on balls or finite quotients."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ...core.base import BaseSpace, SpaceKind, SpaceTooLargeError, ValidationError
from ...core.space import Memory
from ._groups import FinitelyGeneratedGroup, Word, WordProblem, _column, _format_word, _letters, parse_word

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


class CayleySpace(BaseSpace):
    """Cayley graph of a finitely generated group, on a ball of words or on a finite quotient, whose sites hold cells.

    Sites are the distinct elements of a ball of a radius in the generators and their inverses, or of a finite
    quotient, numbered by word length and then by order of discovery, so a ball is a prefix of every larger ball. A
    cell at the site `g` reads the site `g s` for every element `s` of a memory set, and a neighbour outside a ball, or
    reached through a site outside it, reads the value `fill`. Without a ball or a quotient the group must be finite,
    and the sites are all its elements. Preparing the space enumerates its sites, and running a rule reads only its
    neighbour table.

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

    Attributes:
        generators_:
            The names of the generators of the group of the sites, the quotient when there is one.

        words_:
            The shortest word of every site, in site order.

        lengths_:
            The word length of every site, its distance from the identity in the Cayley graph.

        ball_sizes_:
            The number of sites within every radius, from 0 to the radius of the ball or of the quotient.

        generator_table_:
            The site of `g s` for every site `g` and letter `s`, in the order `a, a^-1, b, b^-1, ...`, and -1 outside.

        edges_:
            The edge from every site `g` to `g s` for every generator `s` inside the space, with the generator's
            number.

    Examples:
        >>> from skcomplex.formalisms.groups import CayleySpace, permutation_group
        >>> space = CayleySpace(permutation_group([[1, 0, 2], [1, 2, 0]]), ball=3).prepare()
        >>> space.size_, space.words_[:4], space.ball_sizes_
        (6, ['1', 'a', 'b', 'b^-1'], [1, 4, 6, 6])
        >>> CayleySpace(permutation_group([[1, 2, 0]])).prepare().run(0).tolist()
        [1, 2]
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.LATTICE
    lacks: ClassVar[frozenset[str]] = LACKING_OPTIONS

    size_: int
    generators_: tuple[str, ...]
    words_: list[str]
    lengths_: npt.NDArray[np.int64]
    ball_sizes_: list[int]
    generator_table_: Table
    edges_: list[tuple[int, int, int]]

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

    def _check(self) -> None:
        """Raise an error for both a ball and a quotient, a negative radius, or a ball beyond the site limit.

        The limit is checked from a bound of the ball size that needs no GAP.

        Raises:
            ValidationError:
                The space has both a ball and a quotient, or the ball has a negative radius.

            SpaceTooLargeError:
                The bound of the ball size exceeds `max_sites`.
        """
        if self.ball is not None and self.quotient is not None:
            error_msg = 'A group space is either a ball or a quotient. Give ball or quotient, not both.'
            raise ValidationError(error_msg)
        if self.ball is None:
            return
        if self.ball < 0:
            error_msg = f'The radius of a ball is a nonnegative integer, but it is {self.ball}.'
            raise ValidationError(error_msg)
        sizes = _bound_sizes(self.group, self.ball, self.max_sites)
        if sizes[-1] <= self.max_sites:
            return
        prepares = len(sizes) - 2
        error_msg = (
            f'The ball of radius {self.ball} exceeds the limit of {self.max_sites} sites. The largest radius that prepares '
            f'is {prepares}, with at most {sizes[prepares]} sites.'
        )
        raise SpaceTooLargeError(error_msg)

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

    def _prepare(self) -> None:
        """Enumerate the sites, and set their words, their neighbours by generator, their edges, and their positions."""
        group = self._sites_group()
        built = _enumerate(group.word_problem(), len(group.generators), self.ball, self.max_sites)
        self.generators_ = group.generators
        self.words_ = [_format_word(word, group.generators) for word in built.words]
        self.lengths_ = np.asarray([len(word) for word in built.words], dtype=np.int64)
        self.ball_sizes_ = list(built.sizes)
        self.generator_table_ = built.table
        self.edges_ = [
            (site, int(built.table[site, 2 * generator]), generator)
            for generator in range(built.table.shape[1] // 2)
            for site in range(built.table.shape[0])
            if built.table[site, 2 * generator] >= 0
        ]
        self.size_ = len(self.words_)
        self.n_dim_ = 1
        self.shape_ = (self.size_,)
        self.periodic_ = self.ball is None
        self.default_visual_ = 'cayley'
        self.positions_ = np.zeros((self.size_, 2))
        for length in np.unique(self.lengths_):
            sites = np.flatnonzero(self.lengths_ == length)
            angles = 2 * np.pi * np.arange(sites.size) / sites.size
            self.positions_[sites] = length * np.column_stack([np.cos(angles), np.sin(angles)])

    def _default_neighborhood(self) -> Memory:
        """Return the memory set of the generators and their inverses, without the site itself."""
        return Memory(tuple(word for name in self.generators_ for word in (name, f'{name}^-1')))

    def _index_table(self, neighborhood: Memory) -> npt.NDArray[np.intp]:
        """Return the site `g s` of every site `g` and element `s` of a memory set, with `size_` outside the space."""
        neighborhood = neighborhood.check()
        if not isinstance(neighborhood, Memory):
            error_msg = f'A group space reads a memory set of group elements, but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
        size = self.size_
        table = np.empty((size, len(neighborhood.elements)), dtype=np.intp)
        for position, element in enumerate(neighborhood.elements):
            sites = np.arange(size)
            for letter in parse_word(element, self.generators_):
                inside = sites >= 0
                sites = np.where(inside, self.generator_table_[np.where(inside, sites, 0), _column(letter)], -1)
            table[:, position] = np.where(sites >= 0, sites, size)
        return table

    def _ball(self, radius: int) -> Memory:
        """Return the memory set of the shortest words of the sites within a radius, the identity first."""
        sizes = self.ball_sizes_
        return Memory(tuple(self.words_[: sizes[min(radius, len(sizes) - 1)]]))

    def _neighbor_pairs(self, layer: str | None = None) -> list[tuple[int, int]]:
        """Return the pairs of distinct sites joined by an edge of the Cayley graph, each pair once."""
        found = {frozenset((first, second)): (first, second) for first, second, _ in self.edges_ if first != second}
        return list(found.values())
