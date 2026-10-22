"""Evaluate the laws of one component over its members or its pairs with array operations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..base import Expression, ValidationError, symbols_of
from ..laws import NumericFunction
from ._layout import Entry, Layout
from ._printing import numpy_function
from ._problem import JumpChannels, Vector

type Indices = npt.NDArray[np.intp]
type Pairs = tuple[Indices, Indices] | None
type Reader = Callable[[Vector, Vector, Pairs, float], Any]


def _single_reader(is_state: bool, index: int) -> Reader:
    """Return the function that reads one component shared by every evaluation index."""
    if is_state:
        return lambda x, p, pairs, t: x[index]
    return lambda x, p, pairs, t: p[index]


def _broadcast(value: Any, size: int) -> Vector:  # noqa: ANN401
    """Return the values of an output as one float per evaluation index."""
    if np.ndim(value) == 0:
        return np.full(size, value, dtype=float)
    return np.broadcast_to(np.asarray(value, dtype=float), (size,))


class Source(Enum):
    """How a symbol is read for each evaluation index of a kernel."""

    MEMBER = 'member'
    SINGLE = 'single'
    FIRST = 'first'
    SECOND = 'second'
    NEIGHBOR = 'neighbor'
    DERIVED = 'derived'
    REDUCED = 'reduced'


@dataclass(frozen=True)
class Reference:
    """Component of a state or parameter that a symbol stands for.

    Args:
        entry:
            The layout entry of the variable.

        is_state:
            Whether the variable is a state, as opposed to a parameter.

        column:
            The component of the variable of one member, or for a single read of one group member, the component
            counted from the group's first component.

        source:
            How the component is read for each evaluation index.

        neighbor:
            How a neighbour read finds the neighbour of every member, for the neighbour source.
    """

    entry: Entry
    is_state: bool
    column: int
    source: Source = Source.SINGLE
    neighbor: Any = field(default=None, compare=False, repr=False)
    _index: int = field(init=False, compare=False, repr=False)
    _members: slice = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Compute the flat index of a single read and the strided slice of a member read once."""
        entry = self.entry
        object.__setattr__(self, '_index', entry.start + self.column)
        object.__setattr__(self, '_members', slice(entry.start + self.column, entry.start + entry.size, entry.columns))

    @property
    def index(self) -> int:
        """Flat index of the component of a single read, or of the first member."""
        return self._index

    @property
    def members(self) -> slice:
        """Strided slice of the component of every member in the flat vector."""
        return self._members

    def column_of(self, x: Vector, p: Vector) -> Vector:
        """Return the component of every member as a view into the state or parameter vector.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            column:
                A view with one value per member, or with the one value of a single read.
        """
        vector = x if self.is_state else p
        if self.source is Source.SINGLE:
            return vector[self._index : self._index + 1]
        return vector[self._members]

    @property
    def flat_indices(self) -> Indices:
        """Indices of the component of every member in the flat vector."""
        return self.entry.start + np.arange(self.entry.rows) * self.entry.columns + self.column


@dataclass
class PairSearch:
    """Neighbour search that returns the pairs of members closer than a cutoff.

    Args:
        first:
            The entry of the positions of the first members.

        second:
            The entry of the positions of the second members.

        cutoff:
            The parameter entry of the cutoff distance.

        period:
            The side of a periodic box, or `None` for an open space.
    """

    first: Entry
    second: Entry
    cutoff: Entry
    period: float | None

    def find(self, x: Vector, p: Vector) -> tuple[Indices, Indices]:
        """Return the indices of the first and second members of every pair within the cutoff.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            pairs:
                The first and second member indices.
        """
        cutoff = float(self.cutoff.view(p)[0, 0])
        first, second = self._positions(self.first.view(x)), self._positions(self.second.view(x))
        boxsize = self.period
        first_tree = cKDTree(first, boxsize=boxsize)
        if self.first == self.second:
            pairs = first_tree.query_pairs(cutoff, output_type='ndarray')
            return pairs[:, 0].astype(np.intp), pairs[:, 1].astype(np.intp)
        matrix = first_tree.sparse_distance_matrix(cKDTree(second, boxsize=boxsize), cutoff, output_type='ndarray')
        return matrix['i'].astype(np.intp), matrix['j'].astype(np.intp)

    def _positions(self, positions: Vector) -> Vector:
        """Return positions wrapped into the periodic box when there is one."""
        return np.mod(positions, self.period) if self.period is not None else positions

    def second_images(self, x: Vector, first: Indices, second: Indices) -> Vector:
        """Return the positions of the second members as seen from the first, with minimum image distances.

        Args:
            x:
                The state vector.

            first:
                The first member of every pair.

            second:
                The second member of every pair.

        Returns:
            positions:
                One row of second member positions per pair.
        """
        first_positions, second_positions = self.first.view(x)[first], self.second.view(x)[second]
        if self.period is None:
            return second_positions
        difference = first_positions - second_positions
        return first_positions - (difference - self.period * np.rint(difference / self.period))


@dataclass
class Output:
    """Value a kernel produces for one target component.

    Args:
        target:
            The target component, or `None` for the propensity of a jump.

        expression:
            The symbolic value, or `None` when a callable produces it.

        function:
            The callable or `Call` function that produces the values of every component of its target, or `None`.

        arguments:
            The argument expressions of a `Call` function, or `None` for a callable law.

        component:
            The component of the callable's result that this output holds.

        components:
            The number of components the callable returns per evaluation index.
    """

    target: Reference | None
    expression: Expression | None = None
    function: NumericFunction | Callable[..., Any] | None = None
    arguments: tuple[Expression, ...] | None = None
    component: int = 0
    components: int = 1


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
            The neighbour search of a pair interaction, or `None` for member mode.

        outputs:
            The outputs of the laws.
    """

    owner: str
    time: Expression
    references: Mapping[Expression, Reference]
    rows: int
    layout_states: Layout
    layout_parameters: Layout
    pairs: PairSearch | None = None
    outputs: list[Output] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compile the symbolic outputs and the arguments of calls into one NumPy function."""
        self._slots: list[tuple[int, int]] = []
        expressions: list[Expression] = []
        for output in self.outputs:
            parts = [output.expression] if output.expression is not None else list(output.arguments or ())
            self._slots.append((len(expressions), len(parts)))
            expressions.extend(parts)
        used = symbols_of(expressions) - {self.time}
        unknown = used - set(self.references)
        if unknown:
            names = ', '.join(sorted(map(str, unknown)))
            error_msg = (
                f'A law of {self.owner} uses the symbols {names}, which are not declared. Declare them with `state` '
                'or `parameter` and use them through the namespace `s` that `laws` receives.'
            )
            raise ValidationError(error_msg)
        self._arguments = sorted(used, key=str)
        self._function: Callable[..., Any] | None = (
            numpy_function(self.time, self._arguments, expressions) if expressions else None
        )
        self._readers: dict[Expression, Reader] = {}
        self._argument_readers = [self._reader(item) for item in self._arguments]

    @property
    def arguments(self) -> list[Expression]:
        """Symbols the outputs read, in the order the compiled function takes them after time."""
        return list(self._arguments)

    def gather(self, t: float, x: Vector, p: Vector) -> list[Any]:
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

    def read(self, symbol: Expression, t: float, x: Vector, p: Vector) -> Any:  # noqa: ANN401
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
        x: Vector,
        p: Vector,
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

            def read(x: Vector, p: Vector, pairs: Pairs, t: float) -> Any:  # noqa: ANN401
                return self._gather_around(reference, x, p, pairs, t)

        elif source is Source.SINGLE or (source is Source.REDUCED and reference.entry.members is None):
            read = _single_reader(reference.is_state, reference.index)
        else:
            read = self._member_reader(reference)
        self._readers[symbol] = read
        return read

    def _member_reader(self, reference: Reference) -> Reader:
        """Return the function that reads a member symbol, indexed by the pairs of a pair kernel."""
        is_state, members, source = reference.is_state, reference.members, reference.source
        images = self.pairs is not None and reference.entry == self.pairs.second and is_state
        column = reference.column

        def read(x: Vector, p: Vector, pairs: Pairs, t: float) -> Any:  # noqa: ANN401
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

    def _gather(self, symbol: Expression, x: Vector, p: Vector, pairs: Pairs, t: float = 0.0) -> Any:  # noqa: ANN401
        """Return the values of a symbol for every evaluation index."""
        return self._reader(symbol)(x, p, pairs, t)

    def evaluate(self, t: float, x: Vector, p: Vector) -> tuple[int, tuple[Indices, Indices] | None, list[Vector]]:
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
        values: list[Vector] = []
        function_results: dict[tuple[int, int], Vector] = {}
        for output, (start, count) in zip(self.outputs, self._slots, strict=True):
            if output.expression is not None:
                values.append(_broadcast(computed[start], size))
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

    def _call(self, output: Output, t: float, x: Vector, p: Vector, *, size: int, arguments: list[Vector]) -> Vector:
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


def add_into(output: Vector, target: Reference, values: Vector, pairs: tuple[Indices, Indices] | None) -> None:
    """Add the values of one output to the target component of a flat vector.

    Args:
        output:
            The flat vector to add to.

        target:
            The target component.

        values:
            One value per evaluation index.

        pairs:
            The pairs of a pair kernel, or `None` in member mode.
    """
    if target.source is Source.SINGLE:
        output[target.index] += values.sum()
        return
    column = output[target.members]
    if target.source is Source.MEMBER:
        column += values
    else:
        assert pairs is not None
        np.add.at(column, pairs[0] if target.source is Source.FIRST else pairs[1], values)


def assign_into(output: Vector, target: Reference, values: Vector) -> None:
    """Set the target component of a flat vector to the values of one output.

    Args:
        output:
            The flat vector to write to.

        target:
            The target component, read per member or as a single element.

        values:
            One value per evaluation index.
    """
    if target.source is not Source.MEMBER:
        output[target.index] = values[0]
        return
    output[target.members] = values


def member_indices(reference: Reference, rows: int) -> Indices:
    """Return the flat index of a target component for every evaluation index of a member kernel.

    Args:
        reference:
            The target component.

        rows:
            The number of evaluation indices of the kernel.

    Returns:
        indices:
            One flat index per evaluation index.
    """
    if reference.source is Source.MEMBER:
        return reference.flat_indices
    if reference.source is Source.NEIGHBOR:
        table = np.asarray(reference.neighbor.table, dtype=np.intp)
        inside = table < reference.entry.rows
        flat = reference.entry.start + table * reference.entry.columns + reference.column
        return np.where(inside, flat, -1).astype(np.intp)
    return np.full(rows, reference.entry.start + reference.column, dtype=np.intp)


def noise_rows(kernels: list[Kernel]) -> Indices:
    """Return the state index driven by every Wiener process of noise kernels.

    Args:
        kernels:
            The noise kernels in evaluation order.

    Returns:
        rows:
            One state index per Wiener process.
    """
    rows = [member_indices(item.target, kernel.rows) for kernel in kernels for item in kernel.outputs]  # type: ignore[arg-type]
    return np.concatenate([np.zeros(0, dtype=np.intp), *rows]).astype(np.intp)


def jump_channels(
    kernels: list[Kernel],
    changes: Mapping[int, list[tuple[Reference, float]]],
) -> tuple[JumpChannels, ...]:
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
                [member_indices(reference, kernel.rows) for reference, _ in law_changes],
                dtype=np.intp,
            ).reshape(len(law_changes), kernel.rows)
            increments = np.array([increment for _, increment in law_changes], dtype=float)
            channels.append(JumpChannels(offset, targets, increments))
            offset += kernel.rows
    return tuple(channels)
