"""Compile string productions into vectorised passes over words, and interpret words as geometry."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import Expression, RewritingError, symbols_of
from ..components import Word
from ..laws import Module, Production
from ..laws._rewriting import parse_word
from ._printing import numpy_function
from ._rewriting_kernels import (
    ACTION_CODES,
    DRAW,
    LEFT,
    MOVE,
    PITCH_DOWN,
    PITCH_UP,
    POP,
    PUSH,
    REVERSE,
    RIGHT,
    ROLL_LEFT,
    ROLL_RIGHT,
    neighbors,
    walk_word,
)
from ._structures import WordStructure

if TYPE_CHECKING:
    from ..viz import WordGeometry

IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
PROBABILITY_TOLERANCE = 1e-9
TURNS = (LEFT, RIGHT, REVERSE, PITCH_DOWN, PITCH_UP, ROLL_LEFT, ROLL_RIGHT)

type Arrays = list[npt.NDArray[Any]]


@dataclass(frozen=True)
class Parameters:
    """Parameter symbols of the rewriting interactions, in the order of the parameter vector.

    Args:
        symbols:
            The symbols.

        time:
            The generation symbol.
    """

    symbols: tuple[Expression, ...]
    time: Expression

    def function(
        self,
        variables: Sequence[Expression],
        expressions: Sequence[Any],
        owner: str,
    ) -> Callable[..., Arrays]:
        """Return a NumPy function of the generation, the variables, and the parameters that evaluates expressions.

        Args:
            variables:
                The symbols bound to module parameters.

            expressions:
                The expressions.

            owner:
                The production, named in errors.

        Returns:
            function:
                The function `f(generation, *variables, *parameters)`.

        Raises:
            RewritingError:
                An expression uses a symbol that is neither a variable nor a parameter.
        """
        values = [SR(item) for item in expressions]
        free = symbols_of(values) - set(variables) - set(self.symbols) - {self.time}
        if free:
            error_msg = (
                f'{owner} uses {", ".join(sorted(map(str, free)))}, which is neither a parameter of its predecessor '
                'nor a parameter of its interaction. Declare it with `parameter` on the interaction.'
            )
            raise RewritingError(error_msg)
        return numpy_function(self.time, [*variables, *self.symbols], values)


@dataclass
class _Group:
    """Productions of one predecessor, context, and condition, one of which a matching module takes."""

    code: int
    left: tuple[int, ...]
    right: tuple[int, ...]
    condition: Callable[..., Arrays] | None
    variables: int
    successors: list[int] = field(default_factory=list)
    probabilities: list[Callable[..., Arrays] | None] = field(default_factory=list)
    rank: tuple[int, int, int] = (0, 0, 0)


@dataclass
class _Successor:
    """Successor of a production, as symbol codes and the function of its parameters."""

    codes: npt.NDArray[np.int64]
    slots: list[tuple[int, int]]
    values: Callable[..., Arrays] | None
    variables: int


def _alphabet_of(word: Word, productions: Sequence[Production]) -> dict[str, int]:
    """Return the arity of every symbol, declared or taken from the axiom and the productions."""
    if isinstance(word.alphabet, Mapping):
        return {str(name): int(arity) for name, arity in word.alphabet.items()}
    declared = dict.fromkeys(word.alphabet, 0) if isinstance(word.alphabet, str) else {}
    found: dict[str, int] = {}
    parts: list[Any] = [word.axiom, *word.ignore]
    for production in productions:
        parts += [production.predecessor, production.successor, production.left or '', production.right or '']
    for part in parts:
        for module in parse_word(part, list(declared)):
            found[module.name] = max(found.get(module.name, 0), len(module.parameters))
    if word.alphabet is None:
        return found
    unknown = sorted(set(found) - set(declared))
    if unknown:
        error_msg = (
            f'Word {word.label!r} uses the symbol {unknown[0]!r}, which is not in its alphabet {word.alphabet!r}.'
        )
        raise RewritingError(error_msg)
    return {name: found.get(name, 0) for name in declared}


class CompiledWord:
    """String rewriting system of one word: its alphabet, productions, and geometry, compiled to array passes.

    Args:
        path:
            The path of the word.

        word:
            The word element.

        productions:
            The productions that target it.

        parameters:
            The parameter symbols of the system.

        values:
            The initial parameter values, used to check the probabilities.
    """

    def __init__(
        self,
        path: str,
        word: Word,
        productions: Sequence[Production],
        parameters: Parameters,
        values: npt.NDArray[np.float64],
    ) -> None:
        self.path = path
        self.max_size = int(word.max_size)
        self.parameters = parameters
        arities = _alphabet_of(word, productions)
        self.names = tuple(arities)
        self.arities = arities
        self.width = max(arities.values(), default=0)
        self.counted = tuple(name for name in self.names if IDENTIFIER.match(name))
        self.codes, self.values = self._encode(parse_word(word.axiom, self.names), 'the axiom')
        ignored = parse_word(word.ignore, self.names) if word.ignore else []
        self.ignored = np.zeros(len(self.names) + 1, dtype=np.bool_)
        self.ignored[[self.names.index(module.name) for module in ignored]] = True
        self.opening = self.names.index('[') if '[' in self.names else -2
        self.closing = self.names.index(']') if ']' in self.names else -2
        self.geometry = word.geometry
        if word.geometry is not None:
            meanings = word.geometry.meanings()
            self.actions = np.array(
                [ACTION_CODES.get(meanings.get(name, ''), 0) for name in self.names],
                dtype=np.int64,
            )
        self.successors: list[_Successor] = []
        self.groups = self._groups(productions)
        self._check_probabilities(values)

    def _encode(self, modules: Sequence[Module], where: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Return the codes and the parameter table of numeric modules."""
        codes = np.array([self._code(module, where) for module in modules], dtype=np.int64)
        table = np.full((len(modules), self.width), np.nan)
        for row, module in enumerate(modules):
            table[row, : len(module.parameters)] = [float(item) for item in module.parameters]
        return codes, table

    def _code(self, module: Module, where: str) -> int:
        """Return the code of a module after checking its number of parameters."""
        arity = self.arities.get(module.name)
        if arity is None or len(module.parameters) > arity:
            error_msg = (
                f'{module!r} in {where} of word {self.path!r} does not fit the alphabet, whose symbols and numbers of '
                f'parameters are {self.arities}.'
            )
            raise RewritingError(error_msg)
        return self.names.index(module.name)

    def _successor(self, production: Production, variables: list[Expression]) -> int:
        """Compile the successor of a production and return its index."""
        modules = parse_word(production.successor, self.names)
        codes = np.array([self._code(module, repr(production)) for module in modules], dtype=np.int64)
        slots = [(row, column) for row, module in enumerate(modules) for column in range(len(module.parameters))]
        expressions = [modules[row].parameters[column] for row, column in slots]
        values = self.parameters.function(variables, expressions, repr(production)) if slots else None
        self.successors.append(_Successor(codes, slots, values, len(variables)))
        return len(self.successors) - 1

    def _groups(self, productions: Sequence[Production]) -> list[_Group]:
        """Return the groups of productions in the order they are tried."""
        groups: dict[tuple[str, ...] | int, _Group] = {}
        for order, production in enumerate(productions):
            predecessor = production.predecessor
            module = predecessor if isinstance(predecessor, Module) else Module(predecessor)
            variables = [SR(item) for item in module.parameters]
            if any(not item.is_symbol() for item in variables) or len(set(variables)) < len(variables):
                error_msg = f'The predecessor of {production!r} must have distinct symbols as parameters.'
                raise RewritingError(error_msg)
            code = self._code(module, repr(production))
            left = parse_word(production.left, self.names) if production.left is not None else []
            right = parse_word(production.right, self.names) if production.right is not None else []
            condition = production.condition
            key: tuple[str, ...] | int = (
                (module.name, repr(left), repr(right), str(condition)) if production.probability is not None else -order
            )
            if key not in groups:
                groups[key] = _Group(
                    code=code,
                    left=tuple(self._code(item, repr(production)) for item in left),
                    right=tuple(self._code(item, repr(production)) for item in right),
                    condition=(
                        None
                        if condition is None
                        else self.parameters.function(variables, [condition], repr(production))
                    ),
                    variables=len(variables),
                    rank=(0 if left or right else 1, 0 if condition is not None else 1, order),
                )
            group = groups[key]
            group.successors.append(self._successor(production, variables))
            probability = production.probability
            group.probabilities.append(
                None if probability is None else self.parameters.function([], [probability], repr(production)),
            )
        return sorted(groups.values(), key=lambda group: group.rank)

    def _weights(self, group: _Group, values: npt.NDArray[np.float64], generation: int) -> npt.NDArray[np.float64]:
        """Return the probabilities of the productions of a group, after checking that they sum to one."""
        weights = np.array(
            [1.0 if item is None else float(item(generation, *values)[0]) for item in group.probabilities],
        )
        if len(weights) > 1 and abs(weights.sum() - 1.0) > PROBABILITY_TOLERANCE:
            error_msg = (
                f'The stochastic productions of {self.names[group.code]!r} in word {self.path!r} have probabilities '
                f'that sum to {weights.sum():.6g}, not 1.'
            )
            raise RewritingError(error_msg)
        return weights

    def _check_probabilities(self, values: npt.NDArray[np.float64]) -> None:
        """Raise an error when the probabilities of a stochastic group do not sum to one."""
        for group in self.groups:
            self._weights(group, values, 0)

    def initial(self, values: npt.NDArray[np.float64]) -> WordStructure:
        """Return the word at generation 0.

        Args:
            values:
                The parameter values.

        Returns:
            word:
                The axiom with its geometry.
        """
        del values
        return self._structure(self.codes, self.values, 0)

    def counts(self, word: WordStructure) -> list[float]:
        """Return the count of every counted symbol and the size of a word.

        Args:
            word:
                The word.

        Returns:
            counts:
                The counts, in the order of the states.
        """
        totals = np.bincount(word.codes, minlength=len(self.names))
        return [float(totals[self.names.index(name)]) for name in self.counted] + [float(len(word))]

    def _context(
        self,
        codes: npt.NDArray[np.int64],
        links: npt.NDArray[np.int64],
        rows: npt.NDArray[np.intp],
        context: Sequence[int],
    ) -> npt.NDArray[np.bool_]:
        """Return which rows have neighbours that spell a context, following the neighbour links."""
        index = rows.astype(np.int64)
        matched = np.ones(len(rows), dtype=bool)
        for code in context:
            index = np.where(index >= 0, links[np.maximum(index, 0)], -1)
            matched &= (index >= 0) & (codes[np.maximum(index, 0)] == code)
        return matched

    def _choose(
        self,
        word: WordStructure,
        values: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.int64]:
        """Return the successor of every module, or -1 for modules no production rewrites."""
        codes, table, generation = word.codes, word.parameters, word.generation
        chosen = np.full(len(codes), -1, dtype=np.int64)
        links = None
        if any(group.left or group.right for group in self.groups):
            links = neighbors(codes, self.ignored, self.opening, self.closing)
        draws = rng.random(len(codes)) if any(len(group.successors) > 1 for group in self.groups) else None
        for group in self.groups:
            rows = np.flatnonzero((codes == group.code) & (chosen < 0))
            if not len(rows):
                continue
            keep = np.ones(len(rows), dtype=bool)
            if group.left and links is not None:
                keep &= self._context(codes, links[0], rows, group.left[::-1])
            if group.right and links is not None:
                keep &= self._context(codes, links[1], rows, group.right)
            if group.condition is not None:
                columns = [table[rows, column] for column in range(group.variables)]
                keep &= np.broadcast_to(np.asarray(group.condition(generation, *columns, *values)[0], bool), keep.shape)
            rows = rows[keep]
            successors = np.asarray(group.successors)
            if len(successors) == 1 or draws is None:
                chosen[rows] = successors[0]
                continue
            bins = np.cumsum(self._weights(group, values, generation))
            picks = np.minimum(np.searchsorted(bins, draws[rows], side='right'), len(successors) - 1)
            chosen[rows] = successors[picks]
        return chosen

    def rewrite(self, word: WordStructure, values: npt.NDArray[np.float64], rng: np.random.Generator) -> WordStructure:
        """Return the word of the next generation, every module rewritten in parallel.

        Args:
            word:
                The word.

            values:
                The parameter values.

            rng:
                The random stream of stochastic productions.

        Returns:
            word:
                The next word with its geometry.

        Raises:
            RewritingError:
                The next word would exceed the maximum size, or stochastic probabilities do not sum to one.
        """
        codes, table, generation = word.codes, word.parameters, word.generation
        chosen = self._choose(word, values, rng)
        lengths = np.array([len(item.codes) for item in self.successors] + [1], dtype=np.int64)[chosen]
        size = int(lengths.sum())
        if size > self.max_size:
            error_msg = (
                f'Word {self.path!r} would reach {size} symbols at generation {generation + 1}, above its maximum size '
                f'{self.max_size}. Raise max_size or run fewer generations.'
            )
            raise RewritingError(error_msg)
        offsets = np.cumsum(lengths) - lengths
        new_codes = np.empty(size, dtype=np.int64)
        new_table = np.full((size, self.width), np.nan)
        kept = np.flatnonzero(chosen < 0)
        new_codes[offsets[kept]] = codes[kept]
        new_table[offsets[kept]] = table[kept]
        for index, successor in enumerate(self.successors):
            rows = np.flatnonzero(chosen == index)
            if not len(rows):
                continue
            starts = offsets[rows]
            for position, code in enumerate(successor.codes):
                new_codes[starts + position] = code
            if successor.values is None:
                continue
            columns = [table[rows, column] for column in range(successor.variables)]
            results = successor.values(generation, *columns, *values)
            for (position, column), result in zip(successor.slots, results, strict=True):
                new_table[starts + position, column] = result
        return self._structure(new_codes, new_table, generation + 1)

    def _structure(
        self,
        codes: npt.NDArray[np.int64],
        table: npt.NDArray[np.float64],
        generation: int,
    ) -> WordStructure:
        """Return a word with the segments its geometry draws."""
        if self.geometry is None:
            return WordStructure(self.names, codes, table, generation)
        segments, depths = interpret(self.geometry, self.actions[codes], table, generation)
        return WordStructure(self.names, codes, table, generation, segments=segments, depths=depths)


def interpret(
    geometry: 'WordGeometry',
    actions: npt.NDArray[np.int64],
    table: npt.NDArray[np.float64],
    generation: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return the segments and branch depths a word geometry draws for the actions of the modules of a word.

    Args:
        geometry:
            The word geometry.

        actions:
            The action code of every module.

        table:
            The parameters of every module, whose first column overrides the step or the angle.

        generation:
            The generation, which scales the step by the contraction.

    Returns:
        segments:
            The segments of shape `(k, 2, d)`.

        depths:
            The branch depth of every segment.
    """
    step = geometry.step * geometry.contraction**generation
    angle, heading = np.deg2rad(geometry.angle), np.deg2rad(geometry.heading)
    values = table[:, 0].copy() if table.shape[1] else np.full(len(actions), np.nan)
    turns = np.isin(actions, TURNS)
    values[turns] = np.deg2rad(values[turns])
    if geometry.dimension == 3 or np.isin(actions, (PUSH, POP)).any():  # noqa: PLR2004
        segments, depths = walk_word(actions, values, step, angle, heading)
        return segments[..., : geometry.dimension], depths
    turn = np.where(np.isnan(values), angle, values)
    turn = np.select([actions == LEFT, actions == RIGHT, actions == REVERSE], [turn, -turn, np.pi], 0.0)
    headings = heading + np.cumsum(turn)
    moves = (actions == DRAW) | (actions == MOVE)
    lengths = np.where(moves, np.where(np.isnan(values), step, values), 0.0)
    steps = lengths[:, None] * np.stack([np.cos(headings), np.sin(headings)], axis=1)
    ends = np.cumsum(steps, axis=0)
    drawn = actions == DRAW
    segments = np.stack([(ends - steps)[drawn], ends[drawn]], axis=1)
    return segments, np.zeros(len(segments), dtype=np.int64)
