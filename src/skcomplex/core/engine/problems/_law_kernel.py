"""Laws of one component and one law kind, evaluated over members or pairs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression

from ...components.laws import CallSpec
from ...errors import ValidationError
from ...types import Array, Indices
from ..enums import ReadKind
from ..types import NumericFunction, OptionalPairs, ReadFunction
from ._base import BaseKernel
from ._edge_search import EdgeSearch
from ._expression_printer import ExpressionPrinter
from ._layout_spec import LayoutSpec
from ._pair_search import PairSearch
from ._random_stream import RandomStream
from ._reference_spec import PairsSearch, ReferenceSpec
from ._run_state_spec import RunStateSpec

MODES = {'periodic': 0, 'zero-flux': 1, 'fixed': 2}
LATTICE_BOUNDARIES = {'periodic': 'periodic', 'zero-flux': 'reflecting', 'fixed': 'fixed'}


@dataclass(frozen=True)
class _NoiseProcess:
    """A record of a named Wiener process that a noise output draws.

    It is shared by every output with the same key.

    Args:
        key:
            The identity of the process: a local label of a component, or the path of a `WienerProcessSpec`.

        index:
            The process within a `WienerProcessSpec`.

        dimension:
            The number of processes of a `WienerProcessSpec`, or 0 for a local label, which takes one process per
            member.

        factor:
            The factor `L` with `L L^T = C` of the covariance of a `WienerProcessSpec`, or `None` for the identity.

        links:
            For the link processes of a conserved noise, the process of every evaluation index among the `dimension`
            processes of the key, or -1 for a link that carries no process, otherwise `None`.
    """

    key: object
    index: int = 0
    dimension: int = 0
    factor: npt.NDArray[np.float64] | None = field(default=None, compare=False)
    links: npt.NDArray[np.intp] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class _Output:
    """A record of one value that a kernel of laws gives for one target.

    Every law of a kernel gives one value for its target, such as the rate of the size of a population, written as a
    symbolic expression or computed by a Python function. The spec also says how the value is combined when several laws
    set one state, whether it reads a named noise process, and whether its function draws random numbers. It holds the
    rate of a population, the noise of a price, and the update of a cell.

    Read more in the user guide, under Extend the library.

    Args:
        target:
            The target component, or `None` for the propensity of a jump.

        expression:
            The symbolic value, or `None` when a callable produces it.

        function:
            The callable or `Call` function that produces the values of every component of its target, or `None`.

        arguments:
            The argument expressions of a `Call` function, or `None` for a function of time and the named values.

        component:
            The component of the callable's result that this output holds.

        components:
            The number of components the callable returns per evaluation index.

        process:
            The named Wiener process of a noise output, or `None` for one independent process per evaluation index.

        combined:
            Whether an assigned output adds its change from the state to the value an earlier output set, so that
            several update laws of one state combine.

        reduce:
            `'sum'` or `'mean'` for an assigned output that sets every member it writes to the sum or the mean of the
            values of the rows that write it, or `None`.

        counts:
            The number of rows, over every kernel, that write the member of every row of an output that reduces, or
            `None`.

        relative:
            Whether a function reads the values keyed by the names of its component.

        random:
            Whether a function receives the random stream as the keyword argument `rng`.

        vectorized:
            Whether a `Call` function runs once per evaluation with arrays.

    See Also:
        LawKernel: The kernel that gives the values.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> kernel = LogisticGrowthSystem().prepare().simulate(until=0).problem.jacobian.kernels[0]
        >>> kernel.outputs[0].expression
        -r_growth*x_population*(x_population/K_growth - 1)
    """

    target: ReferenceSpec | None
    expression: Expression | None = None
    function: NumericFunction | Callable[..., object] | None = None
    arguments: tuple[Expression, ...] | None = None
    component: int = 0
    components: int = 1
    process: _NoiseProcess | None = None
    combined: bool = False
    reduce: str | None = None
    counts: Indices | None = field(default=None, compare=False, repr=False)
    relative: bool = False
    random: bool = False
    vectorized: bool = False


@dataclass(frozen=True)
class _JumpChannels:
    """Random events of one jump law, one channel per evaluation index."""

    offset: int
    targets: npt.NDArray[np.intp]
    increments: Array
    changes: tuple[Callable[[float, Array, Array], Array] | None, ...] = ()

    def get_increments_at(self, t: float, x: Array, p: Array, member: int) -> Array:
        """Return the increment of every target at a firing.

        Only a target that has a function gets an increment from it.
        """
        if not self.changes:
            return self.increments
        return np.array(
            [
                increment if change is None else change(t, x, p)[member]
                for increment, change in zip(self.increments, self.changes, strict=True)
            ],
            dtype=float,
        )

    @property
    def size(self) -> int:
        """Number of channels."""
        return int(self.targets.shape[1])


@dataclass(frozen=True)
class _WienerBlock:
    """A record of the correlated Wiener processes of one process spec.

    They are a contiguous block of the processes of a problem.

    Args:
        start:
            The index of the first process of the block.

        size:
            The number of processes of the block.

        factor:
            The factor `L` with `L L^T = C` of the covariance `C` of the block.
    """

    start: int
    size: int
    factor: npt.NDArray[np.float64]


class LawKernel(BaseKernel):
    """A kernel of the laws of one kind of a component, run for every member at once.

    It compiles the laws of a component, such as the rate laws of the growth of a population, into one function and runs
    it over every member of a group, every pair of a neighbourhood, or every edge of a network, as arrays. It gathers
    the values every symbol stands for, calls the Python functions that some laws hold, and gives the values of every
    law. It computes the growth of a population, the forces between particles, and the trades between firms.

    Read more in the user guide, under Extend the library.

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

        names:
            A mapping from the name a relative function reads to the dotted name of the variable.

        stream:
            The random stream a function with `random=True` receives, or `None` for a kernel without one.

    Attributes:
        values_:
            The number of evaluation indices, the pairs in pair mode, and one array of values per output, of the last
            run.

    See Also:
        KernelSpec: What the laws read while they compile.
        _Output: One value the kernel gives.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> kernel = LogisticGrowthSystem().prepare().simulate(until=0).problem.jacobian.kernels[0]
        >>> kernel.owner, kernel.names['target.x']
        ('growth', 'population.x')
    """

    values_: tuple[int, OptionalPairs, list[Array]]

    def __init__(
        self,
        owner: str,
        time: Expression,
        references: Mapping[Expression, ReferenceSpec],
        rows: int,
        layout_states: LayoutSpec,
        *,
        layout_parameters: LayoutSpec,
        pairs: PairsSearch | None = None,
        outputs: Sequence[_Output] = (),
        names: Mapping[str, str] | None = None,
        stream: RandomStream | None = None,
    ) -> None:
        self.owner = owner
        self.time = time
        self.references = references
        self.rows = rows
        self.layout_states = layout_states
        self.layout_parameters = layout_parameters
        self.pairs = pairs
        self.outputs = outputs
        self.names = names
        self.stream = stream

    @functools.cached_property
    def _plan(self) -> tuple[list[tuple[int, int]], list[Expression]]:
        """Return the slot of every output among the compiled expressions.

        The expressions are returned too, found once. Outputs that call the same function on the same arguments share
        one slot.
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
    def _function(self) -> Callable[..., object] | None:
        """NumPy function of the symbolic outputs and the arguments of calls.

        It is `None` without expressions.
        """
        expressions = self._plan[1]
        if not expressions:
            return None
        printer = ExpressionPrinter(time=self.time, compiled=True).prepare(self._arguments)
        return ExpressionPrinter._get_function_of(printer.run(expressions).code_)

    @functools.cached_property
    def _readers(self) -> dict[Expression, ReadFunction]:
        """Reader of every symbol resolved so far."""
        return {}

    @functools.cached_property
    def _applied(self) -> dict[int, Array]:
        """Values of the global operators applied in the current evaluation.

        They are cleared when an evaluation starts.
        """
        return {}

    @functools.cached_property
    def _source_values(self) -> dict[str, Array]:
        """Values of the node of a topology request in the current evaluation.

        They are replaced when an evaluation starts.
        """
        return {}

    @functools.cached_property
    def _argument_readers(self) -> list[ReadFunction]:
        """Readers of the arguments of the compiled function, in order."""
        return [self._get_reader(item) for item in self._arguments]

    def _prepare(self, context: object | None) -> None:
        """Check that the outputs read only declared symbols.

        Neighbour symbols are read only inside a reduction.

        Args:
            context:
                What the kernel serves, which it does not read.

        Raises:
            ValidationError:
                If an output reads an undeclared symbol, or a neighbour symbol outside a neighbourhood reduction.
        """
        super()._prepare(context)
        used = set(self._arguments)
        unknown = used - set(self.references)
        if unknown:
            names = ', '.join(sorted(map(str, unknown)))
            error_msg = (
                f'A law of {self.owner} uses the symbols {names}, which are not declared. Declare them with `state` '
                'or `parameter` and use them through the namespace `s` that `laws` receives.'
            )
            raise ValidationError(error_msg)
        others = sorted(str(item) for item in used if self.references[item].source is ReadKind.OTHER)
        if others:
            error_msg = (
                f'A law of {self.owner} reads the neighbour symbols {", ".join(others)} outside a reduction. Read '
                '`s.other` inside `neighbors`, which reduces over the neighbours of every member, or inside a '
                'reduction over the group.'
            )
            raise ValidationError(error_msg)

    @property
    def arguments(self) -> list[Expression]:
        """Symbols the outputs read.

        They come in the order the compiled function takes them after time.
        """
        return list(self._arguments)

    def gather(self, t: float, x: Array, p: Array) -> list[object]:
        """Return the values of the symbols the outputs read.

        There are values for every evaluation index of a member kernel.

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
        self._applied.clear()
        return [read(x, p, None, t) for read in self._argument_readers]

    def read(self, symbol: Expression, t: float, x: Array, p: Array) -> object:
        """Return the values of one symbol the kernel knows.

        There are values for every evaluation index of a member kernel.

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
        self._applied.clear()
        return self._get_reader(symbol)(x, p, None, t)

    @property
    def has_python_functions(self) -> bool:
        """Whether an output is computed by a function, either a callable law or a `Call`."""
        return any(output.function is not None for output in self.outputs)

    def _gather_around(
        self,
        reference: ReferenceSpec,
        x: Array,
        p: Array,
        pairs: tuple[Indices, Indices] | None,
        t: float,
    ) -> object:
        """Return the values of a symbol read at neighbouring sites.

        Values derived by a global operator are returned too.
        """
        if reference.source is ReadKind.NEIGHBOR:
            return reference.shifted.read(reference.get_column_of(x, p), p)
        read = reference.derived
        if id(read) not in self._applied:
            values = [self._gather(item, x, p, pairs, t) for item in read.symbols]
            operand = read.get_operand_values(t, values, reference.entry.rows)
            self._applied[id(read)] = read.run(RunStateSpec(t, operand, p)).values_
        return self._applied[id(read)]

    def _get_reader(self, symbol: Expression) -> ReadFunction:
        """Return the function that reads the values of a symbol.

        It reads them for every evaluation index, resolved once.
        """
        if symbol in self._readers:
            return self._readers[symbol]
        reference = self.references[symbol]
        source = reference.source
        read: ReadFunction
        if source is ReadKind.DERIVED or source is ReadKind.NEIGHBOR:
            if source is ReadKind.DERIVED:
                for item in reference.derived.symbols:
                    self._get_reader(item)

            def read(x: Array, p: Array, pairs: OptionalPairs, t: float) -> object:
                return self._gather_around(reference, x, p, pairs, t)

        elif source is ReadKind.SOURCE:
            read = self._get_source_reader(reference)
        elif source is ReadKind.SINGLE or (source is ReadKind.REDUCED and reference.entry.members is None):
            read = self._get_single_reader(reference.is_state, reference.index)
        else:
            read = self._get_member_reader(reference)
        self._readers[symbol] = read
        return read

    def _get_source_reader(self, reference: ReferenceSpec) -> ReadFunction:
        """Return the function that reads a component of the node of a topology request.

        It is the node the request starts from.
        """
        name, column = reference.entry.name, reference.column

        def read(x: Array, p: Array, pairs: OptionalPairs, t: float) -> Array:
            if name not in self._source_values:
                error_msg = (
                    f'A law of {self.owner} reads {name} of `namespace.source`, the node a topology request starts '
                    'from, where no such node exists. Read it in the weight, or in the condition or the probability '
                    'per step of a request without a target, outside reductions and rates.'
                )
                raise ValidationError(error_msg)
            return self._source_values[name][column]

        return read

    def _get_member_reader(self, reference: ReferenceSpec) -> ReadFunction:
        """Return the function that reads a member symbol.

        It is indexed by the pairs of a pair kernel.
        """
        is_state, members, source = reference.is_state, reference.members, reference.source
        search = self.pairs if isinstance(self.pairs, PairSearch) else None
        images = search is not None and reference.entry == search.second and is_state
        column = reference.column

        def read(x: Array, p: Array, pairs: OptionalPairs, t: float) -> Array:
            values = (x if is_state else p)[members]
            if source is ReadKind.MEMBER or pairs is None:
                return values
            first, second = pairs
            if source is ReadKind.FIRST:
                return values[first]
            if images and search is not None:
                return search.get_second_images(x, first, second)[:, column]
            return values[second]

        return read

    def _gather(
        self,
        symbol: Expression,
        x: Array,
        p: Array,
        pairs: OptionalPairs,
        t: float = 0.0,
    ) -> object:
        """Return the values of a symbol for every evaluation index."""
        return self._get_reader(symbol)(x, p, pairs, t)

    @property
    def reads_source(self) -> bool:
        """Whether an output reads the node a topology request starts from."""
        return any(self.references[item].source is ReadKind.SOURCE for item in self.arguments)

    @property
    def reads_reductions(self) -> bool:
        """Whether an output reads a reduction.

        It reads one directly or through the operand of a derived read.
        """
        return any(self._reads_reduction(item) for item in self.arguments)

    def _reads_reduction(self, item: Expression) -> bool:
        """Check whether a symbol reads a reduction.

        It reads one directly or through the operand of a derived read.
        """
        reference = self.references[item]
        if reference.source is ReadKind.REDUCED:
            return True
        if reference.source is ReadKind.DERIVED:
            return any(self._reads_reduction(symbol) for symbol in reference.derived.symbols)
        return False

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the number of evaluation indices and one array of values per output.

        In pair mode, the pairs are set too.
        """
        state = self._get_state_of(inputs)
        t, x, p = state.t, state.x, state.p
        self._applied.clear()
        self._source_values.clear()
        self._source_values.update(state.source or {})
        pairs = self.pairs.run(state).pairs_ if self.pairs is not None else None
        size = len(pairs[0]) if pairs is not None else self.rows
        computed: list[npt.ArrayLike] = []
        if self._function is not None:
            computed = ExpressionPrinter._get_values_of(
                self._function(t, *[read(x, p, pairs, t) for read in self._argument_readers]),
            )
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
        self.values_ = size, pairs, values

    def _get_named_values(self, x: Array, p: Array, *, relative: bool) -> dict[str, object]:
        """Return the value of every state and parameter.

        Values are keyed by dotted name or by the name in the component.
        """
        values: dict[str, object] = {**self.layout_states.get_values(x), **self.layout_parameters.get_values(p)}
        if not relative:
            return values
        return {name: values[path] for name, path in (self.names or {}).items() if path in values}

    def _get_keywords(self, output: _Output) -> dict[str, np.random.Generator]:
        """Return the keyword arguments of a function.

        The random stream is passed when the function asks for it.
        """
        if not output.random:
            return {}
        if self.stream is None:
            error_msg = f'A function of {self.owner} asks for the random stream, which this kind of law does not take.'
            raise ValidationError(error_msg)
        return {'rng': self.stream.generator}

    def _call(self, output: _Output, t: float, x: Array, p: Array, *, size: int, arguments: list[Array]) -> Array:
        """Return the result of a function as an array.

        It has one row per evaluation index and one column per component.
        """
        function: Callable[..., object] | None = output.function
        if function is None:
            error_msg = f'The output of {self.owner} computed by a function holds no function.'
            raise ValidationError(error_msg)
        keywords = self._get_keywords(output)
        if output.arguments is not None and not output.vectorized:
            rows = [
                np.atleast_1d(np.asarray(function(t, *row, **keywords), dtype=float))
                for row in (zip(*arguments, strict=True) if arguments else [()] * size)
            ]
            result = np.asarray(rows, dtype=float).reshape(size, -1) if rows else np.zeros((0, output.components))
            return np.broadcast_to(result, (size, output.components))
        if output.arguments is not None:
            result = np.asarray(function(t, *arguments, **keywords), dtype=float)
        else:
            values = self._get_named_values(x, p, relative=output.relative)
            result = np.asarray(function(t, values, **keywords), dtype=float)
        if result.size == size * output.components:
            return result.reshape(size, output.components)
        return np.broadcast_to(result.reshape(-1)[: output.components], (size, output.components))

    def _change_function(self, value: object) -> Callable[[float, Array, Array], Array]:
        """Return the function that gives the change of a jump for every evaluation index."""
        relative = isinstance(value, CallSpec) and value.relative
        function = value.function if isinstance(value, CallSpec) else value
        if not callable(function) or (isinstance(value, CallSpec) and value.arguments):
            error_msg = (
                f'A change of a jump of {self.owner} is {value!r}. Give a number, a function `f(t, values)`, or '
                '`CallSpec(f, relative=True)`.'
            )
            raise ValidationError(error_msg)
        rows = self.rows

        def change(t: float, x: Array, p: Array) -> Array:
            values = self._get_named_values(x, p, relative=relative)
            return np.broadcast_to(np.asarray(function(t, values), dtype=float), (rows,))

        return change

    @staticmethod
    def _get_single_reader(is_state: bool, index: int) -> ReadFunction:
        """Return the function that reads one component shared by every evaluation index."""
        if is_state:
            return lambda x, p, pairs, t: x[index]
        return lambda x, p, pairs, t: p[index]

    @staticmethod
    def _broadcast(value: npt.ArrayLike, size: int) -> Array:
        """Return the values of an output as one float per evaluation index."""
        if np.ndim(value) == 0:
            return np.full(size, value, dtype=float)
        return np.broadcast_to(np.asarray(value, dtype=float), (size,))

    @staticmethod
    def _lower_noise(kernels: list['LawKernel']) -> tuple[Indices, Indices, int, tuple[_WienerBlock, ...]]:
        """Return the state and the Wiener process of every amplitude of noise kernels.

        The processes they draw are returned too. Processes are numbered in order of first use. An output without a
        named process takes one new process per evaluation index, so noise without named processes draws one process per
        amplitude in evaluation order.

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
        rows: list[Indices] = []
        columns: list[Indices] = []
        local: dict[object, Indices] = {}
        starts: dict[object, int] = {}
        blocks: list[_WienerBlock] = []
        counter = [0, -1]
        for kernel in kernels:
            for item in kernel.outputs:
                if item.target is None:
                    error_msg = f'A noise output of {kernel.owner} has no target.'
                    raise ValidationError(error_msg)
                indices = LawKernel._get_member_indices(item.target, kernel.rows, kernel.pairs)
                rows.append(indices)
                columns.append(
                    LawKernel._get_process_columns(
                        item.process,
                        indices.size,
                        counter=counter,
                        local=local,
                        starts=starts,
                        blocks=blocks,
                    ),
                )
        empty = np.zeros(0, dtype=np.intp)
        return (
            np.concatenate([empty, *rows]).astype(np.intp),
            np.concatenate([empty, *columns]).astype(np.intp),
            counter[0],
            tuple(blocks),
        )

    @staticmethod
    def _get_process_columns(
        process: _NoiseProcess | None,
        size: int,
        *,
        counter: list[int],
        local: dict[object, Indices],
        starts: dict[object, int],
        blocks: list[_WienerBlock],
    ) -> Indices:
        """Return the process of every evaluation index of a noise output.

        New processes are numbered in `counter`. `counter` holds the number of processes so far and the column of the
        silent process, -1 until one links.
        """
        if process is None:
            counter[0] += size
            return np.arange(counter[0] - size, counter[0], dtype=np.intp)
        if process.key not in starts and (process.links is not None or process.dimension):
            starts[process.key] = counter[0]
            if process.factor is not None and process.links is None:
                blocks.append(_WienerBlock(counter[0], process.dimension, process.factor))
            counter[0] += process.dimension
        if process.links is not None:
            links = process.links
            if np.all(links >= 0):
                return (starts[process.key] + links).astype(np.intp)
            if counter[1] < 0:
                counter[1] = counter[0]
                blocks.append(_WienerBlock(counter[0], 1, np.zeros((1, 1))))
                counter[0] += 1
            return np.where(links >= 0, starts[process.key] + links, counter[1]).astype(np.intp)
        if process.dimension:
            return np.full(size, starts[process.key] + process.index, dtype=np.intp)
        if process.key not in local:
            counter[0] += size
            local[process.key] = np.arange(counter[0] - size, counter[0], dtype=np.intp)
        if local[process.key].size != size:
            error_msg = (
                f'The noise process {process.key!r} drives members of groups of different sizes. Name one process per '
                'group.'
            )
            raise ValidationError(error_msg)
        return local[process.key]

    @staticmethod
    def _jump_channels(
        kernels: list['LawKernel'],
        changes: Mapping[int, list[tuple[ReferenceSpec, object]]],
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
                    [
                        LawKernel._get_member_indices(reference, kernel.rows, kernel.pairs)
                        for reference, _ in law_changes
                    ],
                    dtype=np.intp,
                ).reshape(len(law_changes), kernel.rows)
                increments = np.array(
                    [increment if isinstance(increment, float) else np.nan for _, increment in law_changes],
                    dtype=float,
                )
                functions = tuple(
                    None if isinstance(increment, float) else kernel._change_function(increment)
                    for _, increment in law_changes
                )
                channels.append(
                    _JumpChannels(offset, targets, increments, functions if any(functions) else ()),
                )
                offset += kernel.rows
        return tuple(channels)

    @staticmethod
    def _get_member_indices(reference: ReferenceSpec, rows: int, pairs: PairsSearch | None = None) -> Indices:
        """Return the flat index of a target component for every evaluation index.

        The kernel is a member or edge kernel.

        Args:
            reference:
                The target component.

            rows:
                The number of evaluation indices of the kernel.

            pairs:
                The edges of an edge kernel, whose ends targets of the source and target roles index, or `None`.

        Returns:
            indices:
                One flat index per evaluation index.

        Raises:
            ValidationError:
                A target of an end of an edge lies on a network whose edges change.
        """
        if reference.source is ReadKind.MEMBER:
            return reference.flat_indices
        if reference.source in {ReadKind.FIRST, ReadKind.SECOND} and isinstance(pairs, EdgeSearch):
            if pairs.source is not None:
                error_msg = (
                    f'A law targets {reference.entry.name} at an end of the edges of a network whose edges change, '
                    'whose ends a fixed channel cannot follow. Target the edge member, or the node through its own '
                    'laws.'
                )
                raise ValidationError(error_msg)
            ends = pairs.sources if reference.source is ReadKind.FIRST else pairs.targets
            return (reference.entry.start + ends * reference.entry.columns + reference.column).astype(np.intp)
        if reference.source is ReadKind.NEIGHBOR:
            if reference.shifted.entry is not None:
                error_msg = (
                    f'A law targets {reference.entry.name} at the members of hyperedges that change during a run, '
                    'whose members a fixed channel cannot follow. Write a rate or an update in place of a jump.'
                )
                raise ValidationError(error_msg)
            table = np.asarray(reference.shifted.table, dtype=np.intp)
            inside = table < reference.entry.rows
            flat = reference.entry.start + table * reference.entry.columns + reference.column
            return np.where(inside, flat, -1).astype(np.intp)
        return np.full(rows, reference.entry.start + reference.column, dtype=np.intp)
