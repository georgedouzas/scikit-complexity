"""String rewriting system of one word: its alphabet, productions, and geometry, compiled to array passes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression

from ....engine._utils import symbols_of
from ....engine.problems import BaseKernel, BaseProblemSpec, ExpressionPrinter
from ....specs import BaseStructureSpec, ParametersSpec, RunStateSpec
from ._geometry_spec import TURTLE_SYMBOLS, GeometrySpec
from ._group_spec import _GroupSpec
from ._module_spec import Letters, ModuleSpec
from ._production_law import ProductionLaw
from ._rewriting_error import RewritingError
from ._rewriting_loops import (
    _ACTION_CODES,
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
    _neighbors,
    _walk_word,
)
from ._successor_spec import _SuccessorSpec
from ._word_element import WordElement
from ._word_structure_spec import WordStructureSpec
from ._words import parse_word
from .types import Arrays

IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


PROBABILITY_TOLERANCE = 1e-9
SPACE_DIMENSION = 3


_TURNS = (LEFT, RIGHT, REVERSE, PITCH_DOWN, PITCH_UP, ROLL_LEFT, ROLL_RIGHT)


class WordKernel(BaseKernel):
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

    `check()` compiles the productions, `prepare(problem)` starts the word at its axiom, and `run(state)` rewrites it
    once with the parameters and the random stream of the state.
    """

    def __init__(
        self,
        path: str,
        word: WordElement,
        productions: Sequence[ProductionLaw],
        parameters: ParametersSpec,
        values: npt.NDArray[np.float64],
    ) -> None:
        self.path = path
        self.word = word
        self.productions = productions
        self.parameters = parameters
        self.values = values

    def _check(self) -> None:
        """Compile the alphabet, the axiom, and the productions, checking them and the probabilities of stochastic ones.

        Raises:
            RewritingError:
                A module does not fit the alphabet, a predecessor has repeated or non-symbol parameters, or the
                probabilities of stochastic productions do not sum to one.
        """
        word, productions, values = self.word, self.productions, self.values
        self.max_size = int(word.max_size)
        arities = self._alphabet_of(word, productions)
        self.names = tuple(arities)
        self.arities = arities
        self.width = max(arities.values(), default=0)
        self.counted = tuple(name for name in self.names if IDENTIFIER.match(name))
        self.codes, self.table = self._encode(parse_word(word.axiom, self.names), 'the axiom')
        ignored = parse_word(word.ignore, self.names) if word.ignore else []
        self.ignored = np.zeros(len(self.names) + 1, dtype=np.bool_)
        self.ignored[[self.names.index(module.name) for module in ignored]] = True
        self.opening = self.names.index('[') if '[' in self.names else -2
        self.closing = self.names.index(']') if ']' in self.names else -2
        self.geometry = word.geometry
        if word.geometry is not None:
            meanings = {**TURTLE_SYMBOLS, **(word.geometry.symbols or {})}
            self.actions = np.array(
                [_ACTION_CODES.get(meanings.get(name, ''), 0) for name in self.names],
                dtype=np.int64,
            )
        self.successors: list[_SuccessorSpec] = []
        self.groups = self._groups(productions)
        self._check_probabilities(values)

    def _encode(
        self,
        modules: Sequence[ModuleSpec],
        where: str,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Return the codes and the parameter table of numeric modules."""
        codes = np.array([self._code(module, where) for module in modules], dtype=np.int64)
        table = np.full((len(modules), self.width), np.nan)
        for row, module in enumerate(modules):
            table[row, : len(module.parameters)] = [float(item) for item in module.parameters]
        return codes, table

    def _code(self, module: ModuleSpec, where: str) -> int:
        """Return the code of a module after checking its number of parameters."""
        arity = self.arities.get(module.name)
        if arity is None or len(module.parameters) > arity:
            error_msg = (
                f'{self._written(module)} in {where} of word {self.path!r} does not fit the alphabet, whose symbols '
                f'and numbers of parameters are {self.arities}.'
            )
            raise RewritingError(error_msg)
        return self.names.index(module.name)

    def _successor(self, production: ProductionLaw, variables: list[Expression]) -> int:
        """Compile the successor of a production and return its index."""
        modules = parse_word(production.successor, self.names)
        text = self._production_text(production)
        codes = np.array([self._code(module, text) for module in modules], dtype=np.int64)
        slots = [(row, column) for row, module in enumerate(modules) for column in range(len(module.parameters))]
        expressions = [modules[row].parameters[column] for row, column in slots]
        values = self._function(variables, expressions, text) if slots else None
        self.successors.append(_SuccessorSpec(codes, tuple(slots), values, len(variables)))
        return len(self.successors) - 1

    def _function(
        self,
        variables: Sequence[Expression],
        expressions: Sequence[object],
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
        symbols, time = self.parameters.symbols, self.parameters.time
        free = symbols_of(values) - set(variables) - set(symbols) - {time}
        if free:
            error_msg = (
                f'{owner} uses {", ".join(sorted(map(str, free)))}, which is neither a parameter of its predecessor '
                'nor a parameter of its element. Declare it as a `ParameterSpec` argument of the word, patch, or host '
                'graph.'
            )
            raise RewritingError(error_msg)
        printer = ExpressionPrinter(time=time, compiled=True).check().prepare([*variables, *symbols])
        compiled = ExpressionPrinter._function_of(printer.run(values))

        def evaluate(*arguments: object) -> Arrays:
            result = compiled(*arguments)
            if not isinstance(result, list):
                error_msg = f'{owner} evaluates to {type(result).__name__}, not a list of arrays.'
                raise RewritingError(error_msg)
            return [np.asarray(item) for item in result]

        return evaluate

    def _groups(self, productions: Sequence[ProductionLaw]) -> list[_GroupSpec]:
        """Return the groups of productions in the order they are tried."""
        groups: dict[tuple[str, ...] | int, _GroupSpec] = {}
        members: dict[tuple[str, ...] | int, tuple[list[int], list[Callable[..., Arrays] | None]]] = {}
        for order, production in enumerate(productions):
            text = self._production_text(production)
            predecessor = production.predecessor
            module = predecessor if isinstance(predecessor, ModuleSpec) else ModuleSpec(predecessor)
            variables = [SR(item) for item in module.parameters]
            if any(not item.is_symbol() for item in variables) or len(set(variables)) < len(variables):
                error_msg = f'The predecessor of {text} must have distinct symbols as parameters.'
                raise RewritingError(error_msg)
            code = self._code(module, text)
            left = parse_word(production.left, self.names) if production.left is not None else []
            right = parse_word(production.right, self.names) if production.right is not None else []
            condition = production.condition
            key: tuple[str, ...] | int = (
                (
                    module.name,
                    ' '.join(self._written(item) for item in left),
                    ' '.join(self._written(item) for item in right),
                    str(condition),
                )
                if production.probability is not None
                else -order
            )
            if key not in groups:
                members[key] = ([], [])
                groups[key] = _GroupSpec(
                    code=code,
                    left=tuple(self._code(item, text) for item in left),
                    right=tuple(self._code(item, text) for item in right),
                    condition=(None if condition is None else self._function(variables, [condition], text)),
                    variables=len(variables),
                    rank=(0 if left or right else 1, 0 if condition is not None else 1, order),
                )
            successors, probabilities = members[key]
            successors.append(self._successor(production, variables))
            probability = production.probability
            probabilities.append(None if probability is None else self._function([], [probability], text))
        found = [
            replace(group, successors=tuple(members[key][0]), probabilities=tuple(members[key][1]))
            for key, group in groups.items()
        ]
        return sorted(found, key=lambda group: group.rank)

    def _weights(self, group: _GroupSpec, values: npt.NDArray[np.float64], generation: int) -> npt.NDArray[np.float64]:
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

    def _prepare(self, context: object) -> None:
        """Start the word at generation 0 for a run of the problem, as `structure_`."""
        if not isinstance(context, BaseProblemSpec):
            error_msg = f'Word {self.path!r} starts from the initial parameters of a problem, but got none.'
            raise RewritingError(error_msg)
        self.structure_ = self._initial(context.initial_parameters)

    def _run(self, inputs: RunStateSpec | None) -> WordStructureSpec:
        """Rewrite the current word once, every module in parallel, with the parameters and the random stream."""
        if inputs is None or inputs.rng is None:
            error_msg = f'Word {self.path!r} rewrites at a state with a random stream, but got {inputs!r}.'
            raise RewritingError(error_msg)
        self.structure_ = self._rewrite(self.structure_, inputs.p, inputs.rng)
        return self.structure_

    def _initial(self, values: npt.NDArray[np.float64]) -> WordStructureSpec:
        """Return the word at generation 0, the axiom with its geometry."""
        del values
        return self._structure(self.codes, self.table, 0)

    def counts(self, word: WordStructureSpec) -> list[float]:
        """Return the count of every counted symbol and the size of a word.

        Args:
            word:
                The word.

        Returns:
            counts:
                The counts, in the order of the states.
        """
        totals = np.bincount(word.codes, minlength=len(self.names))
        return [float(totals[self.names.index(name)]) for name in self.counted] + [float(word.parts)]

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
        word: WordStructureSpec,
        values: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.int64]:
        """Return the successor of every module, or -1 for modules no production rewrites."""
        codes, table, generation = word.codes, word.parameters, word.generation
        chosen = np.full(len(codes), -1, dtype=np.int64)
        links = None
        if any(group.left or group.right for group in self.groups):
            links = _neighbors(codes, self.ignored, self.opening, self.closing)
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

    def _rewrite(
        self,
        word: WordStructureSpec,
        values: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> WordStructureSpec:
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
    ) -> WordStructureSpec:
        """Return a word with the segments its geometry draws."""
        frozen_codes = BaseStructureSpec._frozen(codes, np.int64)
        frozen_table = BaseStructureSpec._frozen(table, np.float64)
        if self.geometry is None:
            return WordStructureSpec(self.names, frozen_codes, frozen_table, generation)
        segments, depths = self._interpret(self.geometry, self.actions[codes], table, generation)
        strokes = BaseStructureSpec._frozen(segments, np.float64)
        frozen_depths = BaseStructureSpec._frozen(depths, np.int64)
        return WordStructureSpec(
            self.names,
            frozen_codes,
            frozen_table,
            generation,
            strokes=strokes,
            depths=frozen_depths,
        )

    @staticmethod
    def _alphabet_of(word: WordElement, productions: Sequence[ProductionLaw]) -> dict[str, int]:
        """Return the arity of every symbol, declared or taken from the axiom and the productions."""
        if isinstance(word.alphabet, Mapping):
            return {str(name): int(arity) for name, arity in word.alphabet.items()}
        declared = dict.fromkeys(word.alphabet, 0) if isinstance(word.alphabet, str) else {}
        found: dict[str, int] = {}
        parts: list[Letters] = [word.axiom, *word.ignore]
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

    @staticmethod
    def _written(module: ModuleSpec) -> str:
        """Return a module as written, such as `A(x, 2)`."""
        if not module.parameters:
            return module.name
        return f'{module.name}({", ".join(str(item) for item in module.parameters)})'

    @staticmethod
    def _production_text(production: ProductionLaw) -> str:
        """Return a production as written, such as `B < A → B`, for error messages."""
        head = WordKernel._written(parse_word(production.predecessor)[0])
        if production.left is not None:
            head = f'{"".join(WordKernel._written(module) for module in parse_word(production.left))} < {head}'
        if production.right is not None:
            head = f'{head} > {"".join(WordKernel._written(module) for module in parse_word(production.right))}'
        modules = parse_word(production.successor)
        separator = ' ' if any(module.parameters for module in modules) else ''
        return f'{head} → {separator.join(WordKernel._written(module) for module in modules)}'

    @staticmethod
    def _interpret(
        geometry: GeometrySpec,
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
        turns = np.isin(actions, _TURNS)
        values[turns] = np.deg2rad(values[turns])
        if geometry.dimension == SPACE_DIMENSION or np.isin(actions, (PUSH, POP)).any():
            segments, depths = _walk_word(actions, values, step, angle, heading)
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
