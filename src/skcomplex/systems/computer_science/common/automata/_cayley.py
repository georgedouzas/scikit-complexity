"""Define the Cayley graph of a group, the memory sets that read it, and a visual."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb
from sage.all__sagemath_gap import libgap
from sage.libs.gap.element import GapElement

from .....core.components.spaces import BaseNeighborhood, BaseSpace
from .....core.engine.problems import BaseProblemSpec, CompiledElementSpec, FrameSpec
from .....core.engine.viz import BaseVisual
from .....core.errors import SpaceError, SpaceTooLargeError, ValidationError

MAX_SITES = 100_000
MAX_PRODUCTS = 10_000_000
LEVEL_POINTS = 2**14
ORDERINGS = ('shortlex', 'recursive')
KNUTH_BENDIX_LIMITS = (('maxeqns', 5000), ('maxstates', 50000))
IDENTITY = '1'
IDENTITY_WORDS = ('', IDENTITY)
IDENTIFIER = re.compile(r'^[A-Za-z_]\w*$')
TOKEN = re.compile(r'^([A-Za-z_]\w*)(?:\^(-?\d+))?$')
NODE_SHARE = 0.15
NODE_SPACING = 40.0
UPWARD = np.array([1.0, -1.0])


class MemoryNeighborhood(BaseNeighborhood):
    """A neighbourhood of the group elements whose values a cellular automaton reads.

    The memory set is a list of elements of a group. A cell at the element `g` reads the cell at `g s` for every element
    `s` of the memory set, in its order, so the neighbourhood keeps its orientation. Elements are words in the
    generators, such as `'a'`, `'a^-1'`, or `'a*b'`, and `'1'` is the identity, the cell itself.

    Unlike `LatticeNeighborhood`, whose offsets are integer vectors, it takes words of a group. Choose it when cells
    live on a `CayleySpace`.

    It appears in mathematics and computer science, in cellular automata on free groups and on groups of automata.

    Args:
        elements:
            The words of the memory set.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Attributes:
        elements_:
            The words of the memory set as strings.

    Examples:
        >>> from skcomplex.systems.computer_science.common.automata import MemoryNeighborhood
        >>> MemoryNeighborhood(['1', 'a', 'a^-1']).prepare().elements_
        ('1', 'a', 'a^-1')
    """

    def __init__(self, elements: Sequence[str], *, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.elements = elements

    def _get_elements(self) -> tuple[str, ...]:
        """Return the words of the memory set as strings."""
        return tuple(str(element) for element in self.elements)

    def _get_slots(self) -> tuple[str, ...]:
        """Return the words of the memory set, in the order a rule reads them."""
        return self.prepare().elements_

    def _join_slots(self, base: Hashable, slot: Hashable) -> str:
        """Return the word `b s` of a word `s` read after a word `b`, `1` for the identity.

        Args:
            base:
                The word read first, such as `'a'`.

            slot:
                The word read after it, such as `'b^-1'`.

        Returns:
            word:
                The product of the two words, such as `'a*b^-1'`.
        """
        parts = [str(part) for part in (base, slot) if str(part).strip() not in IDENTITY_WORDS]
        return '*'.join(parts) or IDENTITY

    def _get_from_slots(self, slots: Sequence[Hashable]) -> MemoryNeighborhood:
        """Return the memory set of other words."""
        return MemoryNeighborhood(tuple(str(slot) for slot in slots))

    def _get_centered(self) -> MemoryNeighborhood:
        """Return the memory set with the identity, the cell itself, first."""
        return MemoryNeighborhood((IDENTITY, *self.elements))

    def _count_centers(self, n_dim: int | None = None) -> int:
        """Return the number of words of the memory set that read the cell itself."""
        return sum(element.strip() in IDENTITY_WORDS for element in self.prepare().elements_)

    def _get_spaces(self) -> tuple[type, ...]:
        """Return the classes of the spaces a memory set is read on, the Cayley graphs."""
        return (CayleySpace,)

    def _prepare(self, context: None = None) -> None:
        """Set `elements_`, the words of the memory set as strings."""
        super()._prepare(context)
        self.elements_ = self._get_elements()


class StateTranslationNeighborhood(BaseNeighborhood):
    """A neighbourhood that moves with the value each cell of a group space holds.

    The cell at `g` reads the cells at `g x(g) s` for every word `s`, where `x(g)` is the group element that the first
    read state holds at `g`, as the number of its site. The sites read move with the configuration, so on an infinite
    group the map has no finite memory set, as for `x(g x(g))` of Ceccherini-Silberstein and Coornaert (2023, Example
    1.8.2), which is continuous and equivariant but no cellular automaton. A run truncates it to a ball or a finite
    quotient, and a product outside the ball reads the fill value of the space.

    Unlike `MemoryNeighborhood`, whose memory set is fixed, it shifts the memory set by the value of the cell. Choose it
    when studying maps on groups that are not cellular automata.

    It appears in mathematics, in symbolic dynamics on groups, as the example of Ceccherini-Silberstein and Coornaert
    shows.

    Args:
        elements:
            The words `s` of the memory set after the translation, `'1'` for the translated site alone.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Attributes:
        elements_:
            The words of the memory set as strings.

    Examples:
        >>> from skcomplex.systems.computer_science.common.automata import StateTranslationNeighborhood
        >>> StateTranslationNeighborhood().prepare().elements_
        ('1',)
    """

    def __init__(self, elements: Sequence[str] = (IDENTITY,), *, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.elements = elements

    def _is_translated(self) -> bool:
        """Check whether the sites read move with the values of the cells, which they do."""
        return True

    def _prepare(self, context: None = None) -> None:
        """Set `elements_`, the words of the memory set as strings."""
        super()._prepare(context)
        self.elements_ = tuple(str(element) for element in self.elements)


class CayleyVisual(BaseVisual):
    """A visual that logs every state of a lattice group on a group space as a graph.

    The states are logged as the nodes of its Cayley graph. Sites sit on rings by word length, with one ring per radius
    and the identity at the centre, colored by state. Every chosen state is logged under `<path>/<state>` as
    `GraphNodes` at every frame, with the edges from every site `g` to `g s` for every generator `s` logged once as
    static `GraphEdges`.

    Unlike `SiteVisual`, which draws the sites of a lattice or a network, it draws the words of a group on rings.
    Choose it for cells on a `CayleySpace`, which draws them with it by default.

    It appears in mathematics and computer science, in pictures of cellular automata on free groups and on groups of
    automata.

    Args:
        states:
            The states to log, or `None` for every state.

        low:
            The value of the darkest color, or `None` for the smallest value of every frame.

        high:
            The value of the brightest color, or `None` for the largest value of every frame.

        path:
            The entity path of the element or agent group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        edges_:
            The entity paths whose static edges are logged.

    Examples:
        >>> from skcomplex.systems.computer_science.common.automata import CayleyVisual
        >>> CayleyVisual(low=0, high=1)
        CayleyVisual(high=1, low=0)
    """

    numeric_options = ('low', 'high')

    def __init__(
        self,
        states: Sequence[str] | None = None,
        low: float | None = None,
        high: float | None = None,
        *,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.states = states
        self.low = low
        self.high = high
        self.path = path
        self.recording = recording

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the group, forget the edges logged in earlier runs, and set the views."""
        self.edges_: set[str] = set()
        super()._prepare(context)

    def _log_edges(self, recording: rr.RecordingStream, path: str, space: CayleySpace) -> None:
        """Log the edges of every generator once, as static data."""
        edges = [(str(site), str(neighbour)) for site, neighbour, _ in space.edges_ if site != neighbour]
        recording.log(path, rr.GraphEdges(edges), static=True)
        self.edges_.add(path)

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log every chosen state of a lattice group on a group space as graph nodes.

        The nodes are colored.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled lattice group, with its path and the dotted names of its states.

            frame:
                The frame.
        """
        space = getattr(element.element, 'space', None)
        if not isinstance(space, CayleySpace):
            return
        positions = space.positions_ * NODE_SPACING * UPWARD
        nodes = [str(site) for site in range(space.size_)]
        for state, name in element.states.items():
            if self.states is not None and state not in self.states:
                continue
            path = f'{element.path}/{state}'
            if path not in self.edges_:
                self._log_edges(recording, path, space)
            values = np.ravel(np.asarray(frame.get_value(name), dtype=float))
            drawn = rr.GraphNodes(
                nodes,
                positions=positions,
                colors=self._get_colors(values, self.low, self.high),
                radii=NODE_SHARE * NODE_SPACING,
                show_labels=False,
            )
            recording.log(path, drawn)

    def _get_views(self, element: object, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return one graph view per chosen state, bounded by the outer ring.

        Args:
            element:
                The compiled lattice group, with its path and the dotted names of its states.

            problem:
                The compiled problem the group belongs to.

        Returns:
            views:
                One graph view per chosen state, with `<path>/<state>` as origin.
        """
        space = getattr(getattr(element, 'element', None), 'space', None)
        if not isinstance(element, CompiledElementSpec) or not isinstance(space, CayleySpace):
            return []
        extent = (float(space.lengths_.max(initial=0)) + 1.0) * NODE_SPACING
        bounds = rrb.VisualBounds2D(x_range=(-extent, extent), y_range=(-extent, extent))
        states = [state for state in element.states if self.states is None or state in self.states]
        return [rrb.GraphView(origin=f'{element.path}/{state}', name=state, visual_bounds=bounds) for state in states]


class CayleySpace(BaseSpace):
    """A space whose sites are the elements of a group, joined by its generators.

    A Cayley graph has one node per element of a group and an edge from `g` to `g s` for every generator `s`. The sites
    are the elements in a ball of words of length up to `ball`, numbered by word length and then by order of discovery,
    so a ball is a prefix of every larger ball. Without a ball the group must be finite, and the sites are all its
    elements. The group is a Sage `PermutationGroup` or `MatrixGroup`, or a GAP group from `libgap`, such as a free
    group, a finitely presented group, which KBMAG solves by Knuth-Bendix completion, or an automaton group.

    A cell at the site `g` reads the site `g s` for every element `s` of a memory set, and a neighbour outside the ball
    reads the value `fill`. A `StateTranslationNeighborhood` reads the site `g x(g) s`, where `x(g)` is the group
    element that the cell holds as the number of its site.

    Unlike `LatticeSpace`, which is the Cayley graph of the integers `Z^d`, it takes any finitely generated group, such
    as a free group whose graph is a tree. Choose it when cells live on the elements of a group.

    It appears in mathematics and computer science, in cellular automata on groups and in symbolic dynamics, as in the
    book of Ceccherini-Silberstein and Coornaert.

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
            The name of the space in its system, such as `'grid'`.

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
        >>> from skcomplex.systems.computer_science.common.automata import CayleySpace
        >>> space = CayleySpace(PermutationGroup([[(1, 2)], [(1, 2, 3)]]), ball=3).prepare()
        >>> space.size_, space.words_[:4], space.ball_sizes_
        (6, ['1', 'a', 'b', 'b^-1'], [1, 4, 6, 6])
    """

    space_tags: ClassVar[tuple[str, ...]] = ('lattice',)

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
        """Return a confluent rewriting system of a finitely presented group.

        It uses the shortlex or recursive ordering.

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
        """Return the level of the tree of an automaton group whose action keys elements.

        It returns 0 for other groups.
        """
        if not libgap.IsTreeAutomorphismGroup(group):
            return 0
        degree = int(libgap.DegreeOfTree(group))
        return max(1, int(np.log(LEVEL_POINTS) / np.log(degree)))

    def _get_key(self, element: GapElement | npt.NDArray[np.intp]) -> str | bytes:
        """Return the key of an element.

        It is its canonical text, or for an automaton its action on a level of the tree.
        """
        return element.tobytes() if self.level_ else str(element)

    def _get_letters(self, group: GapElement) -> list[GapElement | npt.NDArray[np.intp]]:
        """Return every generator and its inverse.

        For an automaton group they are permutations of the points of a level.
        """
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
        """Return the product of an element and a letter.

        For an automaton it composes the actions on the level.
        """
        return letter[element] if self.level_ else element * letter

    def _parse_word(self, text: str) -> list[int]:
        """Return a word written as `a*b^-1*a^2` as signed generator numbers.

        Generator `i` is numbered `i + 1`.

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
        """Return the word and the neighbours by letter of every site, and the ball sizes.

        They come from a breadth-first search.
        """
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

    def _prepare(self, context: None = None) -> None:
        """Enumerate the sites and set their words, neighbours, edges, and positions.

        It first raises an error for a group not a Sage or GAP group, or a negative ball radius.

        The neighbours are by generator.

        Args:
            context:
                Nothing, since a space is prepared alone.

        Raises:
            ValidationError:
                The group is neither a Sage nor a GAP group, or the ball has a negative radius.
        """
        super()._prepare(context)
        if not isinstance(self.group, GapElement) and not hasattr(self.group, '_libgap_'):
            error_msg = f'A Cayley space takes a Sage or GAP group, but it was given {type(self.group).__name__}.'
            raise ValidationError(error_msg)
        if self.ball is not None and (isinstance(self.ball, bool) or not isinstance(self.ball, int) or self.ball < 0):
            error_msg = f'The radius of a ball is a nonnegative integer, but it is {self.ball!r}.'
            raise ValidationError(error_msg)
        libgap.ChangeDirectoryCurrent(str(Path.cwd()))
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
        self.default_visual_ = CayleyVisual
        self.positions_ = np.zeros((self.size_, 2))
        for length in np.unique(self.lengths_):
            sites = np.flatnonzero(self.lengths_ == length)
            angles = 2 * np.pi * np.arange(sites.size) / sites.size
            self.positions_[sites] = length * np.column_stack([np.cos(angles), np.sin(angles)])

    def _has_sites(self) -> bool:
        """Check whether the space has sites joined by neighbours, its group elements."""
        return True

    def _get_lacks(self) -> frozenset[str]:
        """Return the options lattices offer that a Cayley graph does not: image visuals."""
        return frozenset({'image'})

    def _get_default_neighborhood(self) -> MemoryNeighborhood:
        """Return the memory set of the generators and their inverses.

        It leaves out the site itself.
        """
        return MemoryNeighborhood(tuple(word for name in self.generators_ for word in (name, f'{name}^-1')))

    def _get_index_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
        """Return the site `g s` of every site `g` and element `s` of a memory set.

        A site outside the space gets `size_`.
        """
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

    def _get_translation_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
        """Return the site of `g h s` for every site `g`, site `h`, and word `s`.

        A site outside the space gets `size_`. The product walks the shortest word of `h` and then the word `s` from
        `g`, so on a ball a product reached through a site outside it lies outside, as for a memory set.

        Args:
            neighborhood:
                The memory set translated by the value of a cell.

        Returns:
            table:
                One row per site `g`, one column per site `h`, and one layer per word `s`.

        Raises:
            ValidationError:
                The neighbourhood is no state translation neighbourhood.

            SpaceTooLargeError:
                The table would hold more than ten million entries.
        """
        if not isinstance(neighborhood, StateTranslationNeighborhood):
            error_msg = f'A Cayley space translates a memory set by group elements, but it was given {neighborhood!r}.'
            raise ValidationError(error_msg)
        size = self.size_
        words = neighborhood.prepare().elements_
        if size * size * len(words) > MAX_PRODUCTS:
            error_msg = (
                f'The products of the {size} sites of the group space with each other and {len(words)} words exceed '
                f'{MAX_PRODUCTS} entries. Ask for a smaller ball or a finite quotient.'
            )
            raise SpaceTooLargeError(error_msg)
        products = self._get_index_table(MemoryNeighborhood(tuple(self.words_)))
        table = np.empty((size, size, len(words)), dtype=np.intp)
        for layer, word in enumerate(words):
            right = np.append(self._get_index_table(MemoryNeighborhood((word,)))[:, 0], size)
            table[:, :, layer] = right[products]
        return table

    def _get_ball(self, radius: int) -> MemoryNeighborhood:
        """Return the memory set of the shortest words of the sites within a radius.

        The identity comes first.
        """
        sizes = self.ball_sizes_
        return MemoryNeighborhood(tuple(self.words_[: sizes[min(radius, len(sizes) - 1)]]))

    def _get_neighbor_pairs(self, layer: str | None = None) -> list[tuple[int, int]]:
        """Return the pairs of distinct sites joined by an edge of the Cayley graph.

        Each pair comes once.
        """
        found = {frozenset((first, second)): (first, second) for first, second, _ in self.edges_ if first != second}
        return list(found.values())
