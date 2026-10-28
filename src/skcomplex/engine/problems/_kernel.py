"""Laws of one component and one law kind, evaluated over members or pairs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ...types import Array, Indices, OptionalPairs, Reader
from ._layout import Layout
from ._noise_process import NoiseProcess
from ._output import Output
from ._pair_search import PairSearch
from ._printing import numpy_function
from ._problem import _JumpChannels
from ._reference import Reference, Search
from ._source import Source
from ._wiener_block import WienerBlock

MODES = {'periodic': 0, 'zero-flux': 1, 'fixed': 2}


LATTICE_BOUNDARIES = {'periodic': 'periodic', 'zero-flux': 'reflecting', 'fixed': 'fixed'}


@dataclass
class Kernel:
    """Laws of one component and one law kind, evaluated over members or pairs.

    Args:
        owner:
            The path of the component that returned the laws.

        time:
            The time symbol.

        references:
            A mapping from every symbol the laws may use to the component it stands for.

        rows:
            The number of evaluation indices in member mode.

        layout_states:
            The layout of the state vector, used to build the values passed to callables.

        layout_parameters:
            The layout of the parameter vector, used to build the values passed to callables.

        pairs:
            The neighbour search of a pair interaction, or the edges of an interaction over neighbours, or `None` for
            member mode.

        outputs:
            The outputs of the laws.
    """

    owner: str
    time: Expression
    references: Mapping[Expression, Reference]
    rows: int
    layout_states: Layout
    layout_parameters: Layout
    pairs: Search | None = None
    outputs: list[Output] = field(default_factory=list)

    @functools.cached_property
    def _plan(self) -> tuple[list[tuple[int, int]], list[Expression]]:
        """Return the slot of every output among the compiled expressions, and the expressions, found once.

        Outputs that call the same function on the same arguments share one slot.
        """
        slots: list[tuple[int, int]] = []
        expressions: list[Expression] = []
        calls: dict[tuple[int, tuple[Expression, ...]], tuple[int, int]] = {}
        for output in self.outputs:
            key = (id(output.function), output.arguments) if output.arguments is not None else None
            if key is not None and key in calls:
                slots.append(calls[key])
                continue
            parts = [output.expression] if output.expression is not None else list(output.arguments or ())
            slots.append((len(expressions), len(parts)))
            if key is not None:
                calls[key] = slots[-1]
            expressions.extend(parts)
        return slots, expressions

    @functools.cached_property
    def _slots(self) -> list[tuple[int, int]]:
        """Start and count of the compiled values of every output."""
        return self._plan[0]

    @functools.cached_property
    def _arguments(self) -> list[Expression]:
        """Symbols the compiled expressions read besides time, sorted by name."""
        return sorted(
            {item for expression in self._plan[1] for item in SR(expression).variables()} - {self.time},
            key=str,
        )

    @functools.cached_property
    def _function(self) -> Callable[..., Any] | None:
        """NumPy function of the symbolic outputs and the arguments of calls, or `None` without expressions."""
        expressions = self._plan[1]
        return numpy_function(self.time, self._arguments, expressions) if expressions else None

    @functools.cached_property
    def _readers(self) -> dict[Expression, Reader]:
        """Reader of every symbol resolved so far."""
        return {}

    @functools.cached_property
    def _argument_readers(self) -> list[Reader]:
        """Readers of the arguments of the compiled function, in order."""
        return [self._reader(item) for item in self._arguments]

    def check(self) -> 'Kernel':
        """Check that the outputs read only declared symbols, and neighbour symbols only inside a reduction.

        Returns:
            kernel:
                The kernel itself.

        Raises:
            ValidationError:
                If an output reads an undeclared symbol, or a neighbour symbol outside a neighbourhood reduction.
        """
        used = set(self._arguments)
        unknown = used - set(self.references)
        if unknown:
            names = ', '.join(sorted(map(str, unknown)))
            error_msg = (
                f'A law of {self.owner} uses the symbols {names}, which are not declared. Declare them with `state` '
                'or `parameter` and use them through the namespace `s` that `laws` receives.'
            )
            raise ValidationError(error_msg)
        others = sorted(str(item) for item in used if self.references[item].source is Source.OTHER)
        if others:
            error_msg = (
                f'A law of {self.owner} reads the neighbour symbols {", ".join(others)} outside a neighbourhood '
                'reduction. Read `s.other` inside `neighbors`, which reduces over the neighbours of every member.'
            )
            raise ValidationError(error_msg)
        return self

    @property
    def arguments(self) -> list[Expression]:
        """Symbols the outputs read, in the order the compiled function takes them after time."""
        return list(self._arguments)

    def gather(self, t: float, x: Array, p: Array) -> list[Any]:
        """Return the values of the symbols the outputs read, for every evaluation index of a member kernel.

        Args:
            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector, extended with the reductions when the kernel reads them.

        Returns:
            values:
                One value or one array of values per symbol of `arguments`.
        """
        return [read(x, p, None, t) for read in self._argument_readers]

    def read(self, symbol: Expression, t: float, x: Array, p: Array) -> Any:  # noqa: ANN401
        """Return the values of one symbol the kernel knows, for every evaluation index of a member kernel.

        Args:
            symbol:
                A symbol of `references`, such as an operand symbol of a derived read.

            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector, extended with the reductions when the kernel reads them.

        Returns:
            values:
                One value or one array of values.
        """
        return self._reader(symbol)(x, p, None, t)

    @property
    def has_python_functions(self) -> bool:
        """Whether an output is computed by a function, either a callable law or a `Call`."""
        return any(output.function is not None for output in self.outputs)

    def _gather_around(
        self,
        reference: Reference,
        x: Array,
        p: Array,
        pairs: tuple[Indices, Indices] | None,
        t: float,
    ) -> Any:  # noqa: ANN401
        """Return the values of a symbol read at neighbouring sites or derived by a global operator."""
        read = reference.neighbor
        if reference.source is Source.NEIGHBOR:
            return read.read(reference.column_of(x, p))
        values = [self._gather(item, x, p, pairs, t) for item in read.symbols]
        return read.evaluate(t, values, reference.entry.rows)

    def _reader(self, symbol: Expression) -> Reader:
        """Return the function that reads the values of a symbol for every evaluation index, resolved once."""
        if symbol in self._readers:
            return self._readers[symbol]
        reference = self.references[symbol]
        source = reference.source
        read: Reader
        if source is Source.DERIVED or source is Source.NEIGHBOR:
            if source is Source.DERIVED:
                for item in reference.neighbor.symbols:
                    self._reader(item)

            def read(x: Array, p: Array, pairs: OptionalPairs, t: float) -> Any:  # noqa: ANN401
                return self._gather_around(reference, x, p, pairs, t)

        elif source is Source.SINGLE or (source is Source.REDUCED and reference.entry.members is None):
            read = self._single_reader(reference.is_state, reference.index)
        else:
            read = self._member_reader(reference)
        self._readers[symbol] = read
        return read

    def _member_reader(self, reference: Reference) -> Reader:
        """Return the function that reads a member symbol, indexed by the pairs of a pair kernel."""
        is_state, members, source = reference.is_state, reference.members, reference.source
        images = isinstance(self.pairs, PairSearch) and reference.entry == self.pairs.second and is_state
        column = reference.column

        def read(x: Array, p: Array, pairs: OptionalPairs, t: float) -> Any:  # noqa: ANN401
            values = (x if is_state else p)[members]
            if source is Source.MEMBER or pairs is None:
                return values
            first, second = pairs
            if source is Source.FIRST:
                return values[first]
            if images:
                return self.pairs.second_images(x, first, second)[:, column]  # type: ignore[union-attr]
            return values[second]

        return read

    def _gather(self, symbol: Expression, x: Array, p: Array, pairs: OptionalPairs, t: float = 0.0) -> Any:  # noqa: ANN401
        """Return the values of a symbol for every evaluation index."""
        return self._reader(symbol)(x, p, pairs, t)

    def evaluate(self, t: float, x: Array, p: Array) -> tuple[int, tuple[Indices, Indices] | None, list[Array]]:
        """Return the values of every output for every evaluation index.

        Args:
            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            evaluation:
                The number of evaluation indices, the pairs in pair mode, and one array of values per output.
        """
        pairs = self.pairs.find(x, p) if self.pairs is not None else None
        size = len(pairs[0]) if pairs is not None else self.rows
        computed: list[Any] = []
        if self._function is not None:
            computed = list(self._function(t, *[read(x, p, pairs, t) for read in self._argument_readers]))
        values: list[Array] = []
        function_results: dict[tuple[int, int], Array] = {}
        for output, (start, count) in zip(self.outputs, self._slots, strict=True):
            if output.expression is not None:
                values.append(self._broadcast(computed[start], size))
                continue
            key = (id(output.function), start)
            if key not in function_results:
                arguments = [
                    np.broadcast_to(np.asarray(value, dtype=float), (size,))
                    for value in computed[start : start + count]
                ]
                function_results[key] = self._call(output, t, x, p, size=size, arguments=arguments)
            values.append(function_results[key][:, output.component])
        return size, pairs, values

    def _call(self, output: Output, t: float, x: Array, p: Array, *, size: int, arguments: list[Array]) -> Array:
        """Return the result of a function as one row per evaluation index and one column per component."""
        function: Callable[..., Any] = output.function  # type: ignore[assignment]
        if output.arguments is not None:
            rows = [np.atleast_1d(np.asarray(function(t, *row), dtype=float)) for row in zip(*arguments, strict=True)]
            result = np.asarray(rows, dtype=float).reshape(size, -1) if rows else np.zeros((0, output.components))
            return np.broadcast_to(result, (size, output.components))
        values = {**self.layout_states.values(x), **self.layout_parameters.values(p)}
        result = np.asarray(function(t, values), dtype=float)
        if result.size == size * output.components:
            return result.reshape(size, output.components)
        return np.broadcast_to(result.reshape(-1)[: output.components], (size, output.components))

    @staticmethod
    def _single_reader(is_state: bool, index: int) -> Reader:
        """Return the function that reads one component shared by every evaluation index."""
        if is_state:
            return lambda x, p, pairs, t: x[index]
        return lambda x, p, pairs, t: p[index]

    @staticmethod
    def _broadcast(value: Any, size: int) -> Array:  # noqa: ANN401
        """Return the values of an output as one float per evaluation index."""
        if np.ndim(value) == 0:
            return np.full(size, value, dtype=float)
        return np.broadcast_to(np.asarray(value, dtype=float), (size,))

    @staticmethod
    def _lower_noise(kernels: list['Kernel']) -> tuple[Indices, Indices, int, tuple[WienerBlock, ...]]:
        """Return the state and the Wiener process of every amplitude of noise kernels, with the processes they draw.

        Processes are numbered in order of first use. An output without a named process takes one new process per
        evaluation index, so noise without named processes draws one process per amplitude in evaluation order.

        Args:
            kernels:
                The noise kernels in evaluation order.

        Returns:
            rows:
                One state index per amplitude.

            columns:
                One process index per amplitude.

            processes:
                The number of processes.

            blocks:
                The blocks of correlated processes with their covariance factors.
        """
        rows, columns, processes = [], [], _Processes()
        for kernel in kernels:
            for item in kernel.outputs:
                indices = item.target._member_indices(kernel.rows, kernel.pairs)  # type: ignore[arg-type]
                rows.append(indices)
                columns.append(processes.columns(item.process, indices.size))
        empty = np.zeros(0, dtype=np.intp)
        return (
            np.concatenate([empty, *rows]).astype(np.intp),
            np.concatenate([empty, *columns]).astype(np.intp),
            processes.count,
            tuple(processes.blocks),
        )

    @staticmethod
    def _jump_channels(
        kernels: list['Kernel'],
        changes: Mapping[int, list[tuple[Reference, float]]],
    ) -> tuple[_JumpChannels, ...]:
        """Return the channels of every jump law of jump kernels.

        Args:
            kernels:
                The jump kernels in evaluation order.

            changes:
                The changes of every jump output, keyed by the output's identity.

        Returns:
            channels:
                One set of channels per jump law.
        """
        channels, offset = [], 0
        for kernel in kernels:
            for item in kernel.outputs:
                law_changes = changes[id(item)]
                targets = np.array(
                    [reference._member_indices(kernel.rows, kernel.pairs) for reference, _ in law_changes],
                    dtype=np.intp,
                ).reshape(len(law_changes), kernel.rows)
                increments = np.array([increment for _, increment in law_changes], dtype=float)
                channels.append(_JumpChannels(offset, targets, increments))
                offset += kernel.rows
        return tuple(channels)


@dataclass
class _Processes:
    """Wiener processes of noise outputs, numbered in order of first use."""

    count: int = 0
    local: dict[object, Indices] = field(default_factory=dict)
    starts: dict[object, int] = field(default_factory=dict)
    blocks: list[WienerBlock] = field(default_factory=list)
    silent: int | None = None

    def _fresh(self, size: int) -> Indices:
        """Return the columns of new processes."""
        columns = np.arange(self.count, self.count + size, dtype=np.intp)
        self.count += size
        return columns

    def _silent(self) -> int:
        """Return the column of a process of zero variance, which links without a process draw."""
        if self.silent is None:
            self.silent = self.count
            self.blocks.append(WienerBlock(self.count, 1, np.zeros((1, 1))))
            self.count += 1
        return self.silent

    def _links(self, process: NoiseProcess) -> Indices:
        """Return the link process of every evaluation index of a conserved noise output."""
        links = cast('Indices', process.links)
        if process.key not in self.starts:
            self.starts[process.key] = self.count
            self.count += process.dimension
        start = self.starts[process.key]
        if np.all(links >= 0):
            return (start + links).astype(np.intp)
        return np.where(links >= 0, start + links, self._silent()).astype(np.intp)

    def columns(self, process: NoiseProcess | None, size: int) -> Indices:
        """Return the process of every evaluation index of a noise output."""
        if process is None:
            return self._fresh(size)
        if process.links is not None:
            return self._links(process)
        if not process.dimension:
            if process.key not in self.local:
                self.local[process.key] = self._fresh(size)
            columns = self.local[process.key]
            if columns.size != size:
                error_msg = (
                    f'The noise process {process.key!r} drives members of groups of different sizes. Name one process '
                    'per group.'
                )
                raise ValidationError(error_msg)
            return columns
        if process.key not in self.starts:
            self.starts[process.key] = self.count
            if process.factor is not None:
                self.blocks.append(WienerBlock(self.count, process.dimension, process.factor))
            self.count += process.dimension
        return np.full(size, self.starts[process.key] + process.index, dtype=np.intp)
