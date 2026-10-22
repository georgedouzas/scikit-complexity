"""Define finitely generated groups by presentations, automata, permutations, and matrices, with their word problems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import GF, matrix

from ..base import SpaceTooLargeError, ValidationError, WordProblemError

GAP_MODULE = 'sage.all__sagemath_gap'
GAP_PACKAGE = 'passagemath-gap'
KNUTH_BENDIX_LIMITS = (('maxeqns', 5000), ('maxstates', 50000))
ORDERINGS = ('shortlex', 'recursive')
HASH_POINTS = 2**14
INT64_HEADROOM = 2**62
RINGS = ('ZZ', 'GF')
IDENTITY_WORDS = ('', '1')
TOKEN = re.compile(r'^([A-Za-z_]\w*)(?:\^(-?\d+))?$')

type Word = tuple[int, ...]
type Permutation = npt.NDArray[np.int64]


def _libgap() -> Any:  # noqa: ANN401
    """Return the GAP library, imported on first use and started on its first call."""
    try:
        module = import_module(GAP_MODULE)
    except ImportError as error:
        error_msg = (
            'Finitely presented and automaton groups are built with GAP. Install the dependency with '
            f'`pip install {GAP_PACKAGE}`.'
        )
        raise ImportError(error_msg) from error
    return module.libgap


def _letters(n_generators: int) -> Word:
    """Return the letters of the generators and their inverses, in the order `a, a^-1, b, b^-1, ...`."""
    return tuple(sign * (index + 1) for index in range(n_generators) for sign in (1, -1))


def _column(letter: int) -> int:
    """Return the column of a letter in the order of `_letters`."""
    return 2 * (abs(letter) - 1) + (letter < 0)


def _inverse_word(word: Word) -> Word:
    """Return the inverse of a word."""
    return tuple(-letter for letter in reversed(word))


def _reduce_freely(word: Word) -> Word:
    """Return a word with every letter next to its inverse cancelled."""
    reduced: list[int] = []
    for letter in word:
        if reduced and reduced[-1] == -letter:
            reduced.pop()
        else:
            reduced.append(letter)
    return tuple(reduced)


def _cycles(action: Sequence[int]) -> str:
    """Return a permutation of the alphabet in GAP's cycle notation on the points 1 to n."""
    seen, cycles = set(), []
    for start in range(len(action)):
        cycle, point = [], start
        while point not in seen:
            seen.add(point)
            cycle.append(point + 1)
            point = action[point]
        if len(cycle) > 1:
            cycles.append(f'({",".join(map(str, cycle))})')
    return ''.join(cycles)


def _check_names(generators: Sequence[str]) -> tuple[str, ...]:
    """Return generator names after checking that they are distinct identifiers other than 1."""
    names = tuple(generators)
    if not names or len(set(names)) != len(names) or not all(name.isidentifier() for name in names):
        error_msg = f'A group needs distinct generator names that are identifiers, but it has {names}.'
        raise ValidationError(error_msg)
    return names


def _modulus_of(ring: str) -> int | None:
    """Return the prime of a ring `'GF(p)'`, or `None` for `'ZZ'`."""
    if ring == 'ZZ':
        return None
    match = re.fullmatch(r'GF\((\d+)\)', ring.replace(' ', ''))
    prime = int(match.group(1)) if match else 0
    if prime < 2 or any(prime % divisor == 0 for divisor in range(2, math.isqrt(prime) + 1)):  # noqa: PLR2004
        error_msg = f"Unknown ring {ring!r}. Use 'ZZ' or 'GF(p)' for a prime p."
        raise ValidationError(error_msg)
    return prime


def _names_of(names: Sequence[str] | None, count: int) -> tuple[str, ...]:
    """Return the given names of generators, or the letters `a`, `b`, and so on."""
    if names is None:
        return tuple(chr(ord('a') + index) for index in range(count))
    return _check_names(names)


def parse_word(text: str, generators: Sequence[str]) -> Word:
    """Return a word of generators written as `a*b^-1*a^2`, with `1` for the identity, as signed generator numbers.

    Generator `i` is the letter `i + 1` and its inverse the letter `-(i + 1)`.

    Args:
        text:
            The word, generator names joined by `*`, each with an optional integer exponent after `^`.

        generators:
            The names of the generators.

    Returns:
        word:
            The letters of the word.

    Raises:
        ValidationError:
            A factor is not a generator name with an optional integer exponent.

    Examples:
        >>> from skcomplex.core.space import parse_word
        >>> parse_word('a*b^-1*a^2', ('a', 'b'))
        (1, -2, 1, 1)
        >>> parse_word('1', ('a', 'b'))
        ()
    """
    if text.strip() in IDENTITY_WORDS:
        return ()
    letters: list[int] = []
    for factor in text.replace(' ', '').split('*'):
        match = TOKEN.match(factor)
        if match is None or match.group(1) not in generators:
            error_msg = (
                f'The factor {factor!r} of the word {text!r} is not a generator of {", ".join(generators)} with an '
                'optional exponent, such as a^-1.'
            )
            raise ValidationError(error_msg)
        letter = generators.index(match.group(1)) + 1
        exponent = int(match.group(2) or 1)
        letters += [letter if exponent > 0 else -letter] * abs(exponent)
    return tuple(letters)


def _format_word(word: Word, generators: Sequence[str]) -> str:
    """Return a word as text, `1` for the identity."""
    if not word:
        return '1'
    return '*'.join(generators[abs(letter) - 1] + ('' if letter > 0 else '^-1') for letter in word)


def _composed(permutations: Sequence[Permutation], word: Word, points: int) -> Permutation:
    """Return the right action of a word on points, the first letter acting first."""
    images = np.arange(points)
    for letter in word:
        images = permutations[_column(letter)][images]
    return images


def _inverse_permutation(images: Permutation) -> Permutation:
    """Return the inverse of a permutation given by its images."""
    inverse = np.empty_like(images)
    inverse[images] = np.arange(images.size)
    return inverse


class WordProblem:
    """Word problem of a group: its elements, right multiplication by a letter, and equality of elements."""

    def identity(self) -> Any:  # noqa: ANN401
        """Return the identity element."""
        raise NotImplementedError

    def multiply(self, element: Any, letter: int) -> Any:  # noqa: ANN401
        """Return the product of an element with a letter on its right."""
        raise NotImplementedError

    def key(self, element: Any) -> Hashable:  # noqa: ANN401
        """Return a key that equal elements share."""
        raise NotImplementedError

    def same(self, first: Any, second: Any) -> bool:  # noqa: ANN401
        """Check whether two elements with the same key are equal, which they are when keys are exact."""
        return True


class FinitelyGeneratedGroup:
    """Base class of finitely generated groups, given by the names of their generators.

    Subclasses solve the word problem of their presentation. The constructors `finitely_presented_group`,
    `automaton_group`, `permutation_group`, and `matrix_group` return them.

    Args:
        generators:
            The names of the generators.
    """

    generators: tuple[str, ...]

    def word_problem(self) -> WordProblem:
        """Return the word problem of the group, which may start GAP.

        Returns:
            word_problem:
                The elements, their right multiplication by generators, and their equality.
        """
        raise NotImplementedError

    def short_key(self, word: Word) -> Hashable:
        """Return a key of a short word that equal words share, without GAP, possibly merging different words.

        Args:
            word:
                The letters of the word.

        Returns:
            key:
                The key.
        """
        raise NotImplementedError

    def level_quotient(self, level: int) -> 'FinitelyGeneratedGroup':
        """Return the finite quotient of a level of the group.

        Args:
            level:
                The level.

        Returns:
            quotient:
                The quotient, a permutation group with the images of the generators.

        Raises:
            ValidationError:
                The group has no levels.
        """
        error_msg = f'Only automaton groups have quotients by a level of their tree, not {type(self).__name__}.'
        raise ValidationError(error_msg)


class _FreeSolver(WordProblem):
    """Word problem of a free group, by free reduction."""

    def identity(self) -> Word:
        """Return the empty word."""
        return ()

    def multiply(self, element: Word, letter: int) -> Word:
        """Return the freely reduced product."""
        return element[:-1] if element and element[-1] == -letter else (*element, letter)

    def key(self, element: Word) -> Hashable:
        """Return the reduced word."""
        return element


class _RewritingSolver(WordProblem):
    """Word problem of a finitely presented group, by a confluent rewriting system of KBMAG in GAP."""

    def __init__(self, generators: tuple[str, ...], relators: tuple[Word, ...]) -> None:
        gap = _libgap()
        gap.LoadPackage('kbmag')
        free = gap.FreeGroup(list(generators))
        self._generators = list(gap.GeneratorsOfGroup(free))
        self._identity = gap.One(free)
        group = free / [self._word(relator) for relator in relators]
        for ordering in ORDERINGS:
            system = gap.KBMAGRewritingSystem(group)
            gap.SetOrderingOfKBMAGRewritingSystem(system, ordering)
            limit = gap.eval('function(system, name, value) system!.options.(name) := value; end')
            for name, value in KNUTH_BENDIX_LIMITS:
                limit(system, name, value)
            if gap.KnuthBendix(system):
                self._system = system
                return
        error_msg = (
            'Knuth-Bendix completion found no confluent rewriting system for the group, so its Cayley ball cannot be '
            'built. Give a finite quotient, or the group as an automaton, permutation, or matrix group.'
        )
        raise WordProblemError(error_msg)

    def _word(self, word: Word) -> Any:  # noqa: ANN401
        """Return a word as a GAP word."""
        element = self._identity
        for letter in word:
            generator = self._generators[abs(letter) - 1]
            element = element * (generator if letter > 0 else generator**-1)
        return element

    def identity(self) -> Any:  # noqa: ANN401
        """Return the identity as a GAP word."""
        return self._identity

    def multiply(self, element: Any, letter: int) -> Any:  # noqa: ANN401
        """Return the reduced word of the product."""
        generator = self._generators[abs(letter) - 1]
        return _libgap().ReducedWord(self._system, element * (generator if letter > 0 else generator**-1))

    def key(self, element: Any) -> Hashable:  # noqa: ANN401
        """Return the reduced word as a tuple."""
        return tuple(int(item) for item in _libgap().ExtRepOfObj(element))


class _PermutationSolver(WordProblem):
    """Word problem of a permutation group, by composing images."""

    def __init__(self, permutations: Sequence[Permutation]) -> None:
        self._permutations = list(permutations)
        self._points = int(permutations[0].size) if permutations else 0

    def identity(self) -> Permutation:
        """Return the identity permutation."""
        return np.arange(self._points)

    def multiply(self, element: Permutation, letter: int) -> Permutation:
        """Return the permutation that acts by the element and then by the letter."""
        return self._permutations[_column(letter)][element]

    def key(self, element: Permutation) -> Hashable:
        """Return the bytes of the images."""
        return element.tobytes()


class _MatrixSolver(WordProblem):
    """Word problem of a matrix group over the integers or a prime field, by matrix products."""

    def __init__(self, matrices: Sequence[npt.NDArray[np.int64]], modulus: int | None) -> None:
        self._matrices = list(matrices)
        self._modulus = modulus
        self._size = int(matrices[0].shape[0]) if matrices else 0

    def identity(self) -> npt.NDArray[np.int64]:
        """Return the identity matrix."""
        return np.eye(self._size, dtype=np.int64)

    def multiply(self, element: npt.NDArray[np.int64], letter: int) -> npt.NDArray[np.int64]:
        """Return the product of the element with the matrix of the letter, reduced modulo the field size."""
        factor = self._matrices[_column(letter)]
        if self._modulus is not None:
            return (element @ factor) % self._modulus
        largest = int(np.abs(element).max()) * int(np.abs(factor).max()) * self._size
        if largest >= INT64_HEADROOM:
            error_msg = (
                'The entries of the integer matrices overflow 64-bit integers. Ask for a ball of smaller radius.'
            )
            raise SpaceTooLargeError(error_msg)
        return element @ factor

    def key(self, element: npt.NDArray[np.int64]) -> Hashable:
        """Return the bytes of the entries."""
        return element.tobytes()


class _AutomatonSolver(WordProblem):
    """Word problem of an automaton group: tree-level actions as keys, and AutomGrp in GAP to resolve equal keys."""

    def __init__(self, group: '_AutomatonGroup') -> None:
        self._group = group
        self._permutations = group.level_permutations(group.hash_level)
        self._points = int(self._permutations[0].size)
        self._gap_generators: list[Any] = []

    def _gap_word(self, word: Word) -> Any:  # noqa: ANN401
        """Return a word as an element of the automaton group in GAP."""
        if not self._gap_generators:
            gap = _libgap()
            gap.LoadPackage('automgrp')
            group = gap.AutomatonGroup(self._group.gap_recursion())
            self._gap_generators = list(gap.GeneratorsOfGroup(group))
            self._gap_identity = gap.One(group)
        element = self._gap_identity
        for letter in word:
            generator = self._gap_generators[abs(letter) - 1]
            element = element * (generator if letter > 0 else generator**-1)
        return element

    def identity(self) -> tuple[Permutation, Word]:
        """Return the identity action and the empty word."""
        return np.arange(self._points), ()

    def multiply(self, element: tuple[Permutation, Word], letter: int) -> tuple[Permutation, Word]:
        """Return the action and the word of the product."""
        images, word = element
        return self._permutations[_column(letter)][images], (*word, letter)

    def key(self, element: tuple[Permutation, Word]) -> Hashable:
        """Return the bytes of the action on the hash level."""
        return element[0].tobytes()

    def same(self, first: tuple[Permutation, Word], second: tuple[Permutation, Word]) -> bool:
        """Check with AutomGrp whether two elements that act alike on the hash level are equal."""
        return bool(_libgap().IsOne(self._gap_word(first[1]) * self._gap_word(second[1]) ** -1))


@dataclass(frozen=True)
class _PresentedGroup(FinitelyGeneratedGroup):
    """Finitely presented group, by generators and relators."""

    generators: tuple[str, ...]
    relators: tuple[Word, ...]

    def word_problem(self) -> WordProblem:
        """Return free reduction without relators, and a confluent rewriting system otherwise."""
        if not self.relators:
            return _FreeSolver()
        return _RewritingSolver(self.generators, self.relators)

    def short_key(self, word: Word) -> Hashable:
        """Return the freely reduced word, with generators whose square is a relator taken as their own inverses."""
        involutions = {relator[0] for relator in self.relators if relator in {(letter, letter) for letter in relator}}
        involutions |= {-letter for letter in involutions}
        folded = _reduce_freely(tuple(abs(letter) if letter in involutions else letter for letter in word))
        changed = True
        while changed:
            changed = False
            for position in range(len(folded) - 1):
                if folded[position] == folded[position + 1] and folded[position] in involutions:
                    folded = folded[:position] + folded[position + 2 :]
                    changed = True
                    break
        return folded


@dataclass(frozen=True)
class _PermutationGroup(FinitelyGeneratedGroup):
    """Permutation group, by the images of its generators."""

    generators: tuple[str, ...]
    images: tuple[tuple[int, ...], ...]

    def permutations(self) -> list[Permutation]:
        """Return the permutations of the letters."""
        found = []
        for images in self.images:
            forward = np.asarray(images, dtype=np.int64)
            found += [forward, _inverse_permutation(forward)]
        return found

    def word_problem(self) -> WordProblem:
        """Return the composition of images."""
        return _PermutationSolver(self.permutations())

    def short_key(self, word: Word) -> Hashable:
        """Return the bytes of the permutation of the word."""
        return _composed(self.permutations(), word, len(self.images[0])).tobytes()


@dataclass(frozen=True)
class _MatrixGroup(FinitelyGeneratedGroup):
    """Matrix group over the integers or a prime field, by its generators."""

    generators: tuple[str, ...]
    matrices: tuple[tuple[tuple[int, ...], ...], ...]
    modulus: int | None

    def letter_matrices(self) -> list[npt.NDArray[np.int64]]:
        """Return the matrices of the letters, each generator followed by its inverse."""
        found = []
        for entries in self.matrices:
            forward = np.asarray(entries, dtype=np.int64)
            if self.modulus is None:
                inverse = np.rint(np.linalg.inv(forward)).astype(np.int64)
            else:
                inverse = np.asarray(matrix(GF(self.modulus), entries).inverse().rows(), dtype=np.int64)
            found += [forward % self.modulus if self.modulus else forward, inverse]
        return found

    def word_problem(self) -> WordProblem:
        """Return matrix products."""
        return _MatrixSolver(self.letter_matrices(), self.modulus)

    def short_key(self, word: Word) -> Hashable:
        """Return the bytes of the product of the word."""
        solver = self.word_problem()
        element = solver.identity()
        for letter in word:
            element = solver.multiply(element, letter)
        return solver.key(element)


@dataclass(frozen=True)
class _AutomatonGroup(FinitelyGeneratedGroup):
    """Automaton group, by the wreath recursion of a Mealy automaton whose states are the generators."""

    generators: tuple[str, ...]
    sections: tuple[tuple[int | None, ...], ...]
    actions: tuple[tuple[int, ...], ...]

    @property
    def degree(self) -> int:
        """Size of the alphabet."""
        return len(self.actions[0])

    @property
    def hash_level(self) -> int:
        """Level of the tree whose action keys the elements, the deepest with at most `HASH_POINTS` leaves."""
        return max(1, int(math.log(HASH_POINTS) / math.log(self.degree)))

    def level_permutations(self, level: int) -> list[Permutation]:
        """Return the action of every letter on the leaves of a level, each generator followed by its inverse."""
        memo: dict[tuple[int | None, int], Permutation] = {}

        def act(state: int | None, depth: int) -> Permutation:
            """Return the action of a state on the leaves of a depth."""
            if state is None or depth == 0:
                return np.arange(self.degree**depth)
            if (state, depth) not in memo:
                below = self.degree ** (depth - 1)
                memo[state, depth] = np.concatenate(
                    [
                        image * below + act(self.sections[state][letter], depth - 1)
                        for letter, image in enumerate(self.actions[state])
                    ],
                )
            return memo[state, depth]

        found = []
        for state in range(len(self.generators)):
            forward = act(state, level)
            found += [forward, _inverse_permutation(forward)]
        return found

    def gap_recursion(self) -> str:
        """Return the wreath recursion in the notation of AutomGrp."""
        parts = []
        for name, sections, action in zip(self.generators, self.sections, self.actions, strict=True):
            states = ', '.join('1' if state is None else self.generators[state] for state in sections)
            cycles = _cycles(action)
            parts.append(f'{name} = ({states}){cycles}')
        return ', '.join(parts)

    def word_problem(self) -> WordProblem:
        """Return the tree-level keys resolved by AutomGrp."""
        return _AutomatonSolver(self)

    def short_key(self, word: Word) -> Hashable:
        """Return the bytes of the action of the word on the hash level."""
        return _composed(self.level_permutations(self.hash_level), word, self.degree**self.hash_level).tobytes()

    def level_quotient(self, level: int) -> FinitelyGeneratedGroup:
        """Return the permutation group of the action on a level of the tree.

        Args:
            level:
                The level, at least 1.

        Returns:
            quotient:
                The permutation group on the leaves of the level.

        Raises:
            ValidationError:
                The level is not positive.
        """
        if level < 1:
            error_msg = f'A level of the tree is a positive integer, but it is {level}.'
            raise ValidationError(error_msg)
        images = self.level_permutations(level)[::2]
        return _PermutationGroup(self.generators, tuple(tuple(int(item) for item in image) for image in images))


def finitely_presented_group(generators: Sequence[str], relators: Sequence[str] = ()) -> FinitelyGeneratedGroup:
    """Return the group of generators and relators, whose balls are built by free reduction or Knuth-Bendix completion.

    Without relators the group is free, and its words are reduced without GAP. With relators, KBMAG in GAP looks for a
    confluent rewriting system, by the shortlex and then the recursive ordering.

    Args:
        generators:
            The names of the generators.

        relators:
            The words that equal the identity, such as `'a*b*a^-1*b^-1'`.

    Returns:
        group:
            The group.

    Raises:
        ValidationError:
            A generator name is repeated or not an identifier, or a relator has a factor that is not a generator.

    Examples:
        >>> from skcomplex.core.space import finitely_presented_group
        >>> finitely_presented_group(['a', 'b']).generators
        ('a', 'b')
    """
    names = _check_names(generators)
    return _PresentedGroup(names, tuple(parse_word(relator, names) for relator in relators))


def automaton_group(
    recursion: Mapping[str, tuple[Sequence[str | None], Sequence[int] | None]],
) -> FinitelyGeneratedGroup:
    """Return the self-similar group of a Mealy automaton, given by the wreath recursion of its states.

    Every state is a generator, mapped to its sections, one state or `None` for the identity per letter of the
    alphabet, and to its permutation of the alphabet, the images of the letters `0` to `n - 1`, or `None` for the
    identity. Elements act on the right, so a word acts by its first letter first.

    Args:
        recursion:
            The sections and the permutation of every state.

    Returns:
        group:
            The group, whose balls are built from its action on a level of the tree and AutomGrp in GAP.

    Raises:
        ValidationError:
            A section names an unknown state, the states have alphabets of different sizes, or a permutation is not a
            permutation of the alphabet.

    Examples:
        >>> from skcomplex.core.space import automaton_group
        >>> adding = automaton_group({'t': ((None, 't'), (1, 0))})
        >>> adding.generators
        ('t',)
    """
    names = _check_names(list(recursion))
    degrees = {len(sections) for sections, _ in recursion.values()}
    if len(degrees) != 1 or min(degrees) < 2:  # noqa: PLR2004
        error_msg = f'Every state of an automaton has one section per letter of one alphabet, but they have {degrees}.'
        raise ValidationError(error_msg)
    degree = degrees.pop()
    sections, actions = [], []
    for name, (states, action) in recursion.items():
        unknown = [state for state in states if state is not None and state not in names]
        images = tuple(range(degree)) if action is None else tuple(int(image) for image in action)
        if unknown or sorted(images) != list(range(degree)):
            error_msg = (
                f'The state {name} has the sections {tuple(states)} and the permutation {action}. Sections are states '
                f'of {", ".join(names)} or None, and the permutation reorders 0 to {degree - 1}.'
            )
            raise ValidationError(error_msg)
        sections.append(tuple(None if state is None else names.index(state) for state in states))
        actions.append(images)
    return _AutomatonGroup(names, tuple(sections), tuple(actions))


def permutation_group(
    generators: Sequence[Sequence[int]],
    names: Sequence[str] | None = None,
) -> FinitelyGeneratedGroup:
    """Return the group of permutations of the points `0` to `n - 1`, given by the images of the points.

    Elements act on the right, so the product `g h` acts by `g` first.

    Args:
        generators:
            The images of the points under every generator.

        names:
            The names of the generators, or `None` for `a`, `b`, and so on.

    Returns:
        group:
            The group.

    Raises:
        ValidationError:
            A generator is not a permutation of the same points as the others.

    Examples:
        >>> from skcomplex.core.space import permutation_group
        >>> permutation_group([[1, 0, 2], [1, 2, 0]]).generators
        ('a', 'b')
    """
    images = tuple(tuple(int(point) for point in generator) for generator in generators)
    sizes = {len(image) for image in images}
    if len(sizes) != 1 or any(sorted(image) != list(range(len(image))) for image in images):
        error_msg = f'Every generator permutes the same points 0 to n - 1, but the generators are {images}.'
        raise ValidationError(error_msg)
    return _PermutationGroup(_names_of(names, len(images)), images)


def matrix_group(
    generators: Sequence[npt.ArrayLike],
    ring: str = 'ZZ',
    names: Sequence[str] | None = None,
) -> FinitelyGeneratedGroup:
    """Return the group of invertible square matrices over the integers or a prime field, given by its generators.

    Args:
        generators:
            The integer matrices.

        ring:
            `'ZZ'` for the integers, or `'GF(p)'` for the field of a prime number `p` of elements.

        names:
            The names of the generators, or `None` for `a`, `b`, and so on.

    Returns:
        group:
            The group.

    Raises:
        ValidationError:
            The ring is unknown, a generator is not a square integer matrix of the same size as the others, or it is
            not invertible over the ring.

    Examples:
        >>> from skcomplex.core.space import matrix_group
        >>> matrix_group([[[1, 2], [0, 1]], [[1, 0], [2, 1]]]).generators
        ('a', 'b')
    """
    modulus = _modulus_of(ring)
    arrays = [np.asarray(generator) for generator in generators]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1 or len(arrays[0].shape) != 2 or arrays[0].shape[0] != arrays[0].shape[1]:  # noqa: PLR2004
        error_msg = f'The generators of a matrix group are square matrices of one size, but their shapes are {shapes}.'
        raise ValidationError(error_msg)
    for array in arrays:
        determinant = round(float(np.linalg.det(array)))
        invertible = abs(determinant) == 1 if modulus is None else determinant % modulus != 0
        if not np.array_equal(array, np.rint(array)) or not invertible:
            error_msg = f'The generator {array.tolist()} is not an invertible integer matrix over {ring}.'
            raise ValidationError(error_msg)
    entries = tuple(tuple(tuple(int(item) for item in row) for row in array) for array in arrays)
    return _MatrixGroup(_names_of(names, len(arrays)), entries, modulus)
