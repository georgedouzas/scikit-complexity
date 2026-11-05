"""Define Cayley spaces, the Cayley graphs of finitely generated groups on balls or on whole finite groups."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_gap import libgap
from sage.libs.gap.element import GapElement

from ...errors import SpaceError, SpaceTooLargeError, ValidationError
from ._base import BaseNeighborhood, BaseSpace
from ._memory import MemoryNeighborhood

MAX_SITES = 100_000
LEVEL_POINTS = 2**14
ORDERINGS = ('shortlex', 'recursive')
KNUTH_BENDIX_LIMITS = (('maxeqns', 5000), ('maxstates', 50000))
IDENTITY_WORDS = ('', '1')
IDENTIFIER = re.compile(r'^[A-Za-z_]\w*$')
TOKEN = re.compile(r'^([A-Za-z_]\w*)(?:\^(-?\d+))?$')


class CayleySpace(BaseSpace):
    """Cayley graph of a finitely generated group, on a ball of words or a whole finite group, whose sites hold cells.

    The group is a native Sage or GAP group: a Sage `PermutationGroup` or `MatrixGroup`, or a GAP group from `libgap`,
    such as the free group `libgap.FreeGroup(['a', 'b'])`, a finitely presented group `free / relators`, an automaton
    group `libgap.AutomatonGroup('t = (1, t)(1, 2)')` of the AutomGrp package, or the finite quotient of one, such as
    `libgap.PermGroupOnLevel(group, 3)`. Finitely presented groups are solved by Knuth-Bendix completion with KBMAG.

    Sites are the distinct elements of a ball of a radius in the generators and their inverses, numbered by word length
    and then by order of discovery, so a ball is a prefix of every larger ball. Without a ball the group must be finite,
    and the sites are all its elements. A cell at the site `g` reads the site `g s` for every element `s` of a memory
    set, and a neighbour outside a ball, or reached through a site outside it, reads the value `fill`.

    Args:
        group:
            The group.

        ball:
            The radius of the ball, or `None` for a finite group.

        fill:
            The value read outside a ball.

        max_sites:
            The largest number of sites.

        label:
            The label of the space in its system.

    Attributes:
        generators_:
            The names of the generators, their GAP names when these are identifiers, or `a`, `b`, and so on.

        words_:
            The shortest word of every site, in site order.

        lengths_:
            The word length of every site, its distance from the identity in the Cayley graph.

        ball_sizes_:
            The number of sites within every radius, from 0 to the radius of the ball or of the finite group.

        generator_table_:
            The site of `g s` for every site `g` and letter `s`, in the order `a, a^-1, b, b^-1, ...`, and -1 outside.

        edges_:
            The edge from every site `g` to `g s` for every generator `s` inside the space, with the generator's
            number.

        level_:
            The level of the tree whose action keys the elements of an automaton group, or 0 for other groups.

    Examples:
        >>> from sage.all__sagemath_gap import PermutationGroup
        >>> from skcomplex.core.spaces import CayleySpace
        >>> space = CayleySpace(PermutationGroup([[(1, 2)], [(1, 2, 3)]]), ball=3).prepare()
        >>> space.size_, space.words_[:4], space.ball_sizes_
        (6, ['1', 'a', 'b', 'b^-1'], [1, 4, 6, 6])
    """

    def __init__(
        self,
        group: object,
        *,
        ball: int | None = None,
        fill: float = 0.0,
        max_sites: int = MAX_SITES,
        label: str = 'space',
    ) -> None:
        super().__init__(label)
        self.group = group
        self.ball = ball
        self.fill = fill
        self.max_sites = max_sites

    def _get_gap_group(self) -> GapElement:
        """Return the group as a GAP group, with the AutomGrp and KBMAG packages loaded."""
        libgap.LoadPackage('automgrp')
        libgap.LoadPackage('kbmag')
        return self.group if isinstance(self.group, GapElement) else libgap(self.group)

    def _get_rewriting(self, group: GapElement) -> GapElement:
        """Return a confluent rewriting system of a finitely presented group, by the shortlex or recursive ordering.

        Args:
            group:
                The finitely presented GAP group.

        Returns:
            system:
                The confluent KBMAG rewriting system.

        Raises:
            SpaceError:
                Knuth-Bendix completion finds no confluent rewriting system.
        """
        limit = libgap.eval('function(system, name, value) system!.options.(name) := value; end')
        for ordering in ORDERINGS:
            system = libgap.KBMAGRewritingSystem(group)
            libgap.SetOrderingOfKBMAGRewritingSystem(system, ordering)
            for name, value in KNUTH_BENDIX_LIMITS:
                limit(system, name, value)
            if libgap.KnuthBendix(system):
                return system
        error_msg = (
            'Knuth-Bendix completion found no confluent rewriting system for the group, so its Cayley ball cannot be '
            'built. Give a finite quotient, or the group as an automaton, permutation, or matrix group.'
        )
        raise SpaceError(error_msg)

    def _get_level(self, group: GapElement) -> int:
        """Return the level of the tree of an automaton group whose action keys its elements, or 0 for other groups."""
        if not libgap.IsTreeAutomorphismGroup(group):
            return 0
        degree = int(libgap.DegreeOfTree(group))
        return max(1, int(np.log(LEVEL_POINTS) / np.log(degree)))

    def _get_key(self, element: GapElement | npt.NDArray[np.intp]) -> str | bytes:
        """Return the key of an element: its canonical text, or for an automaton its action on a level of the tree."""
        return element.tobytes() if self.level_ else str(element)

    def _get_letters(self, group: GapElement) -> list[GapElement | npt.NDArray[np.intp]]:
        """Return every generator and its inverse, as permutations of the points of a level for an automaton group."""
        letters = [generator**sign for generator in libgap.GeneratorsOfGroup(group) for sign in (1, -1)]
        if not self.level_:
            return letters
        points = int(libgap.DegreeOfTree(group)) ** self.level_
        return [
            np.array(libgap.ListPerm(libgap.PermOnLevel(letter, self.level_), points).sage(), dtype=np.intp) - 1
            for letter in letters
        ]

    def _multiply(
        self,
        element: GapElement | npt.NDArray[np.intp],
        letter: GapElement | npt.NDArray[np.intp],
    ) -> GapElement | npt.NDArray[np.intp]:
        """Return the product of an element and a letter, composing the actions on the level for an automaton."""
        return letter[element] if self.level_ else element * letter

    def _parse_word(self, text: str) -> list[int]:
        """Return a word written as `a*b^-1*a^2` as signed generator numbers, `i + 1` for generator `i`.

        Args:
            text:
                The word as text.

        Returns:
            word:
                The signed generator numbers, empty for the identity.

        Raises:
            ValidationError:
                A factor is not a generator name with an optional integer exponent.
        """
        if text.strip() in IDENTITY_WORDS:
            return []
        letters: list[int] = []
        for factor in text.replace(' ', '').split('*'):
            match = TOKEN.match(factor)
            if match is None or match.group(1) not in self.generators_:
                error_msg = (
                    f'The factor {factor!r} of the word {text!r} is not a generator of {", ".join(self.generators_)} '
                    'with an optional exponent, such as a^-1.'
                )
                raise ValidationError(error_msg)
            letter = self.generators_.index(match.group(1)) + 1
            exponent = int(match.group(2) or 1)
            letters += [letter if exponent > 0 else -letter] * abs(exponent)
        return letters

    def _format_word(self, word: list[int]) -> str:
        """Return a word of signed generator numbers as text, `1` for the identity."""
        if not word:
            return '1'
        return '*'.join(self.generators_[abs(letter) - 1] + ('' if letter > 0 else '^-1') for letter in word)

    def _add_site(
        self,
        sites: dict[str | bytes, int],
        elements: list[GapElement | npt.NDArray[np.intp]],
        element: GapElement | npt.NDArray[np.intp],
    ) -> int:
        """Return the new site of an element, after adding it to the known ones.

        Args:
            sites:
                The site of every known element, by key, which the new element joins.

            elements:
                The known elements in the order of their sites, which the new element joins.

            element:
                The element to add.

        Returns:
            site:
                The site of the element.

        Raises:
            SpaceTooLargeError:
                The space has more than `max_sites` sites.
        """
        if len(elements) >= self.max_sites:
            error_msg = (
                f'The group space has more than {self.max_sites} sites. Ask for a smaller ball or a finite quotient, '
                'or raise max_sites.'
            )
            raise SpaceTooLargeError(error_msg)
        elements.append(element)
        sites[self._get_key(element)] = len(elements) - 1
        return len(elements) - 1

    def _enumerate(self, group: GapElement) -> tuple[list[list[int]], list[list[int]], list[int]]:
        """Return the word and the neighbours by letter of every site, and the ball sizes, by breadth-first search."""
        rewriting = self._get_rewriting(group) if libgap.IsFpGroup(group) else None
        words_group = libgap.FreeGroupOfFpGroup(group) if rewriting is not None else group
        letters = self._get_letters(words_group)
        identity = np.arange(len(letters[0])) if self.level_ else libgap.One(words_group)
        elements: list[GapElement | npt.NDArray[np.intp]] = [identity]
        words: list[list[int]] = [[]]
        rows = [[-1] * len(letters)]
        sites = {self._get_key(identity): 0}
        frontier, sizes, depth = [0], [1], 0
        while frontier:
            grows = self.ball is None or depth < self.ball
            found: list[int] = []
            for site, column in ((site, column) for site in frontier for column in range(len(letters))):
                if rows[site][column] != -1:
                    continue
                element = self._multiply(elements[site], letters[column])
                element = element if rewriting is None else libgap.ReducedWord(rewriting, element)
                neighbour = sites.get(self._get_key(element))
                if neighbour is None and grows:
                    neighbour = self._add_site(sites, elements, element)
                    words.append([*words[site], (column // 2 + 1) * (-1 if column % 2 else 1)])
                    rows.append([-1] * len(letters))
                    found.append(neighbour)
                if neighbour is not None:
                    rows[site][column] = neighbour
                    rows[neighbour][column + 1 if column % 2 == 0 else column - 1] = site
            if not grows:
                break
            if found or self.ball is not None:
                sizes.append(len(elements))
            frontier, depth = found, depth + 1
        return words, rows, sizes

    def _check(self) -> None:
        """Raise an error for a group that is not a Sage or GAP group, or a ball of negative radius.

        Raises:
            ValidationError:
                The group is neither a Sage nor a GAP group, or the ball has a negative radius.
        """
        if not isinstance(self.group, GapElement) and not hasattr(self.group, '_libgap_'):
            error_msg = f'A Cayley space takes a Sage or GAP group, but it was given {type(self.group).__name__}.'
            raise ValidationError(error_msg)
        if self.ball is not None and (isinstance(self.ball, bool) or not isinstance(self.ball, int) or self.ball < 0):
            error_msg = f'The radius of a ball is a nonnegative integer, but it is {self.ball!r}.'
            raise ValidationError(error_msg)

    def _prepare(self, context: None = None) -> None:
        """Enumerate the sites, and set their words, their neighbours by generator, their edges, and their positions."""
        group = self._get_gap_group()
        names = [str(generator) for generator in libgap.GeneratorsOfGroup(group)]
        if not all(IDENTIFIER.match(name) for name in names) or len(set(names)) != len(names):
            names = [chr(ord('a') + index) for index in range(len(names))]
        self.generators_ = tuple(names)
        self.level_ = self._get_level(group)
        words, rows, sizes = self._enumerate(group)
        table = np.asarray(rows, dtype=np.int64)
        self.words_ = [self._format_word(word) for word in words]
        self.lengths_ = np.asarray([len(word) for word in words], dtype=np.int64)
        self.ball_sizes_ = sizes
        self.generator_table_ = table
        self.edges_ = [
            (site, int(table[site, 2 * generator]), generator)
            for generator in range(table.shape[1] // 2)
            for site in range(table.shape[0])
            if table[site, 2 * generator] >= 0
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

    def _get_lacks(self) -> frozenset[str]:
        """Return the options lattices offer that a Cayley graph does not: image visuals."""
        return frozenset({'image'})

    def _default_neighborhood(self) -> MemoryNeighborhood:
        """Return the memory set of the generators and their inverses, without the site itself."""
        return MemoryNeighborhood(tuple(word for name in self.generators_ for word in (name, f'{name}^-1')))

    def _index_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
        """Return the site `g s` of every site `g` and element `s` of a memory set, with `size_` outside the space."""
        if not isinstance(neighborhood, MemoryNeighborhood):
            error_msg = f'A Cayley space reads a memory set of group elements, but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
        elements = neighborhood.prepare().elements_
        size = self.size_
        table = np.empty((size, len(elements)), dtype=np.intp)
        for position, element in enumerate(elements):
            sites = np.arange(size)
            for letter in self._parse_word(element):
                column = 2 * (abs(letter) - 1) + (letter < 0)
                inside = sites >= 0
                sites = np.where(inside, self.generator_table_[np.where(inside, sites, 0), column], -1)
            table[:, position] = np.where(sites >= 0, sites, size)
        return table

    def _ball(self, radius: int) -> MemoryNeighborhood:
        """Return the memory set of the shortest words of the sites within a radius, the identity first."""
        sizes = self.ball_sizes_
        return MemoryNeighborhood(tuple(self.words_[: sizes[min(radius, len(sizes) - 1)]]))

    def _neighbor_pairs(self, layer: str | None = None) -> list[tuple[int, int]]:
        """Return the pairs of distinct sites joined by an edge of the Cayley graph, each pair once."""
        found = {frozenset((first, second)): (first, second) for first, second, _ in self.edges_ if first != second}
        return list(found.values())
