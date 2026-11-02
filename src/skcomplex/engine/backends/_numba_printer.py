"""Print the kernels and reductions of a problem as the source of fused Numba functions, with its layout slots."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import Expression

from ...core.spaces import ConeNeighborhood, NearestNeighborhood
from ...errors import ValidationError
from ...specs import EntrySpec, ReadSpec
from .._utils import symbols_of
from ..problems import BasePrinter, ExpressionPrinter, OutputSpec, ReductionScheduleSpec, ReferenceSpec
from ..problems._edge_row_search import EdgeRowSearch
from ..problems._edge_search import EdgeSearch
from ..problems._law_kernel import LawKernel
from ..problems._metric_search import MetricSearch
from ..problems._neighborhood_kernel import NeighborhoodKernel
from ..problems._pair_search import PairSearch
from ..problems._reduction_kernel import ReductionKernel
from ..problems._static_search import StaticSearch
from ..types import ReadKind
from ._base import _AUTO_MEMBERS
from ._numba_components import GIANT, LABEL, LARGEST, SIZE

INDENT = '    '
COMPONENT_KINDS = {
    'component-label': LABEL,
    'component-size': SIZE,
    'component-largest': LARGEST,
    'component-giant': GIANT,
}


class NumbaPrinter(BasePrinter[ReductionScheduleSpec, Sequence[tuple[str, str, list[LawKernel], bool]]]):
    """Printer of the kernels and reductions of a problem as the source of fused Numba functions.

    `prepare(reductions)` starts an empty source and prints the function that fuses the reductions, or none for
    `None`, and `run(parts)` prints the kernels of every law kind into the source lines of `code_`. A part is the label
    of the entry function, its mode, `'add'`, `'assign'`, or `'concatenate'`, the kernels, and whether they read the
    reductions. After a run, `functions_` holds the compiled functions the source calls by name, and `layout(entries)`
    returns the array of the starts and rows of the entries it reads.

    Args:
        parallel:
            Whether member and pair loops over groups of at least 1000 members, and the fused reductions, run in
            parallel threads.

    Attributes:
        code_:
            The printed source lines.

        slots_:
            The layout slot of every entry and data array by name.

        functions_:
            The compiled functions the source calls, by global name.

        data_:
            The integer data appended to the layout, by name.

        orders_:
            The orders drawn at every step that gather tables read through, by the name of their data.
    """

    code_: list[str]

    def __init__(self, parallel: bool = False) -> None:
        self.parallel = parallel

    def _prepare(self, context: ReductionScheduleSpec | None) -> None:
        """Start an empty source and print the function that fuses the reductions, unless there are none."""
        self.code_ = []
        self.slots_: dict[str, int] = {}
        self.functions_: dict[str, object] = {}
        self.data_: dict[str, np.ndarray] = {}
        self.orders_: dict[str, np.ndarray] = {}
        if context is not None and context.steps:
            self._reduce_function(context, self.parallel)

    def _run(self, inputs: Sequence[tuple[str, str, list[LawKernel], bool]] | None) -> None:
        """Print the kernels of every part, each part with its entry function, into the source lines.

        Args:
            inputs:
                The parts: the label of the entry function, its mode, the kernels, and whether they read reductions.

        Raises:
            ValidationError:
                No parts are passed.
        """
        if inputs is None:
            error_msg = 'The Numba printer runs on the parts of a problem. Pass them to `run`.'
            raise ValidationError(error_msg)
        for label, mode, kernels, reduced in inputs:
            names = []
            for index, kernel in enumerate(kernels):
                name = f'_{label}{index}_'
                if isinstance(kernel.pairs, EdgeSearch):
                    self._edge_kernel(kernel, name, mode)
                elif kernel.pairs is not None:
                    self._pair_kernel(kernel, name, self._is_parallel(kernel))
                else:
                    self._member_kernel(kernel, name, mode, self._is_parallel(kernel))
                names.append(name)
            size = ' + '.join(self._output_count(kernel) for kernel in kernels) or '0'
            self._entry_function(label, names, mode, size, reduced=reduced)

    def _is_parallel(self, kernel: LawKernel) -> bool:
        """Check whether a kernel runs in parallel threads: over groups of at least 1000 members."""
        if not self.parallel:
            return False
        rows = [
            reference.entry.rows
            for reference in kernel.references.values()
            if reference.source in {ReadKind.MEMBER, ReadKind.FIRST, ReadKind.SECOND}
        ]
        return max(rows, default=0) >= _AUTO_MEMBERS

    def _slot(self, entry: EntrySpec) -> int:
        """Return the layout slot of an entry, whose start is `L[2 * slot]` and rows `L[2 * slot + 1]`.

        Args:
            entry:
                The entry.

        Returns:
            slot:
                The slot.
        """
        return self.slots_.setdefault(entry.name, len(self.slots_))

    def _data_slot(self, name: str, values: np.ndarray) -> int:
        """Return the layout slot of integer data appended to the layout, whose offset is `L[2 * slot]`.

        Args:
            name:
                The name of the data, unique in the source.

            values:
                The integers.

        Returns:
            slot:
                The slot.
        """
        self.data_[name] = np.asarray(values, dtype=np.int64)
        return self.slots_.setdefault(name, len(self.slots_))

    def layout(self, entries: Mapping[str, EntrySpec]) -> np.ndarray:
        """Return the layout array of starts and rows of every slot, followed by the data of the data slots.

        Args:
            entries:
                Every entry by name.

        Returns:
            layout:
                The layout array.
        """
        layout = np.zeros(2 * len(self.slots_), dtype=np.int64)
        offset = layout.size
        for name, slot in self.slots_.items():
            if name in self.data_:
                layout[2 * slot], layout[2 * slot + 1] = offset, self.data_[name].size
                offset += self.data_[name].size
            else:
                layout[2 * slot], layout[2 * slot + 1] = entries[name].start, entries[name].rows
        return np.concatenate([layout, *(self.data_[name] for name in self.slots_ if name in self.data_)])

    def _function_name(self, function: object) -> str:
        """Return the global name of a compiled function called from the generated source.

        Args:
            function:
                The Numba-compiled function.

        Returns:
            name:
                The global name.
        """
        for name, known in self.functions_.items():
            if known is function:
                return name
        name = f'_function{len(self.functions_)}_'
        self.functions_[name] = function
        return name

    @staticmethod
    def _vector(reference: ReferenceSpec) -> str:
        """Return the name of the vector that holds a reference."""
        return 'x' if reference.is_state else 'p'

    @staticmethod
    def _is_shared(reference: ReferenceSpec) -> bool:
        """Check whether a reference holds one value for every evaluation index, a single element or a reduction."""
        return reference.source is ReadKind.SINGLE or (
            reference.source is ReadKind.REDUCED and reference.entry.members is None
        )

    @staticmethod
    def _hoist_layout(lines: list[str]) -> list[str]:
        """Return function lines with every layout lookup `_L<k>_` read once after the signature."""
        used = sorted({int(index) for line in lines for index in re.findall(r'_L(\d+)_', line)})
        reads = [f'{INDENT}_L{index}_ = L[{index}]' for index in used]
        return [*lines[:2], *reads, *lines[2:]]

    def _index(self, reference: ReferenceSpec, member: str | None) -> str:
        """Return the flat index of a reference for a member variable, or for a single element."""
        start = f'_L{2 * self._slot(reference.entry)}_'
        if member is None:
            return f'{start} + {reference.column}'
        return f'{start} + {member} * {reference.entry.columns} + {reference.column}'

    @staticmethod
    def _printed(printer: ExpressionPrinter, expressions: list[Expression]) -> tuple[list[tuple[str, str]], list[str]]:
        """Return the assignments of the repeated subexpressions and the codes of expressions printed by a printer.

        Args:
            printer:
                The prepared printer of code, not of a compiled function.

            expressions:
                The expressions.

        Returns:
            printed:
                The assignments and the code of every expression.

        Raises:
            TypeError:
                The printer keeps a compiled function instead of code.
        """
        printed = printer.run(expressions).code_
        if not isinstance(printed, tuple):
            error_msg = f'The printer keeps assignments and codes, not a {type(printed).__name__}.'
            raise TypeError(error_msg)
        assignments, codes = printed
        return assignments, codes

    @staticmethod
    def _target_of(output: OutputSpec) -> ReferenceSpec:
        """Return the target of an output a kernel writes.

        Args:
            output:
                The output.

        Returns:
            target:
                The target reference.

        Raises:
            ValidationError:
                The output has no target.
        """
        if output.target is None:
            error_msg = 'A kernel output that the Numba printer writes needs a target.'
            raise ValidationError(error_msg)
        return output.target

    @staticmethod
    def _pair_search_of(kernel: LawKernel) -> PairSearch:
        """Return the pair search of a pair kernel.

        Args:
            kernel:
                The pair kernel.

        Returns:
            search:
                The pair search.

        Raises:
            TypeError:
                The kernel holds no pair search.
        """
        if not isinstance(kernel.pairs, PairSearch):
            error_msg = f'A pair kernel holds a pair search, not a {type(kernel.pairs).__name__}.'
            raise TypeError(error_msg)
        return kernel.pairs

    @staticmethod
    def _by_end(
        outputs: Sequence[tuple[int, OutputSpec]],
    ) -> tuple[list[tuple[int, OutputSpec]], list[tuple[int, OutputSpec]]]:
        """Return the outputs that target the first members of pairs and those that target the second members.

        Args:
            outputs:
                The outputs, with their indices.

        Returns:
            ends:
                The outputs of the first members and of the second members.
        """
        firsts = [item for item in outputs if NumbaPrinter._target_of(item[1]).source is ReadKind.FIRST]
        seconds = [item for item in outputs if NumbaPrinter._target_of(item[1]).source is ReadKind.SECOND]
        return firsts, seconds

    def _emit_values(
        self,
        outputs: Sequence[OutputSpec],
        read: Callable[[Expression], str],
        depth: int,
    ) -> tuple[list[str], list[str]]:
        """Return the lines that read symbols and compute outputs, and the code of every output value."""
        pad = INDENT * depth
        expressions = [output.expression for output in outputs if output.expression is not None]
        arguments = [argument for output in outputs if output.expression is None for argument in output.arguments or ()]
        symbols = sorted(symbols_of(expressions + arguments), key=str)
        reads = [(str(item), read(item)) for item in symbols if str(item) != 't']
        lines = [f'{pad}{name} = {code}' for name, code in reads if name != code]
        assignments, codes = self._printed(ExpressionPrinter('scalar').check().prepare(), [*expressions, *arguments])
        lines += [f'{pad}{name} = {code}' for name, code in assignments]
        expression_codes, argument_codes = iter(codes[: len(expressions)]), iter(codes[len(expressions) :])
        values: list[str] = []
        calls: dict[tuple[int, tuple[str, ...]], str] = {}
        for output in outputs:
            if output.expression is not None:
                values.append(next(expression_codes))
                continue
            printed = [next(argument_codes) for _ in output.arguments or ()]
            key = (id(output.function), tuple(printed))
            if key not in calls:
                calls[key] = f'_call{len(calls)}_'
                call_arguments = ', '.join(['t', *printed])
                lines.append(f'{pad}{calls[key]} = {self._function_name(output.function)}({call_arguments})')
            values.append(calls[key] if output.components == 1 else f'{calls[key]}[{output.component}]')
        return lines, values

    def _single_reads(self, kernel: LawKernel, outputs: Sequence[OutputSpec]) -> list[str]:
        """Return the lines that read every single-element symbol of outputs once, before the member loops."""
        expressions = [output.expression for output in outputs if output.expression is not None]
        arguments = [argument for output in outputs for argument in output.arguments or ()]
        symbols = sorted(symbols_of(expressions + arguments), key=str)
        return [
            f'{INDENT}{item} = {NumbaPrinter._vector(reference)}[{self._index(reference, None)}]'
            for item in symbols
            if (reference := kernel.references.get(item)) is not None and NumbaPrinter._is_shared(reference)
        ]

    def _edge_rows(self, search: EdgeSearch) -> tuple[str, list[str]]:
        """Return the code of the number of edges and the lines that read the ends `_a_` and `_b_` of the edge `_m_`."""
        pad = INDENT * 2
        if search.source is not None and search.target is not None:
            first, second = self._slot(search.source), self._slot(search.target)
            return f'_L{2 * first + 1}_', [
                f'{pad}_a_ = int(p[_L{2 * first}_ + _m_])',
                f'{pad}_b_ = int(p[_L{2 * second}_ + _m_])',
            ]
        digest = hashlib.sha256(search.sources.tobytes() + search.targets.tobytes()).hexdigest()[:16]
        first = self._data_slot(f'__edge_sources_{digest}', search.sources)
        second = self._data_slot(f'__edge_targets_{digest}', search.targets)
        return f'_L{2 * first + 1}_', [f'{pad}_a_ = L[_L{2 * first}_ + _m_]', f'{pad}_b_ = L[_L{2 * second}_ + _m_]']

    @staticmethod
    def _is_gathered(reference: ReferenceSpec) -> bool:
        """Check whether a reference reads a group through a gather table of members, with no value outside it."""
        return (
            reference.source is ReadKind.NEIGHBOR
            and reference.shifted.sizes is None
            and reference.shifted.ghost is None
        )

    def _gathered_member(self, reference: ReferenceSpec) -> str:
        """Return the code of the member a gathered reference reads at the row `_m_`, from its table in the layout.

        A table whose members change during a run holds positions in a parameter entry, which holds the members, and a
        table of the positions in an order drawn at every step reads the member at its position in the order, whose data
        the layout holds after the tables.
        """
        read = reference.shifted
        digest = hashlib.sha256(np.asarray(read.table, dtype=np.int64).tobytes()).hexdigest()[:16]
        slot = self._data_slot(f'__gather_{digest}', read.table)
        row = f'L[_L{2 * slot}_ + _m_]'
        if read.entry is not None:
            row = f'int(p[_L{2 * self._slot(read.entry)}_ + {row}])'
        if read.order is None:
            return row
        name = f'__order_{id(read.order)}'
        self.orders_[name] = read.order
        order = self._data_slot(name, read.order)
        return f'L[_L{2 * order}_ + {row}]'

    def _output_count(self, kernel: LawKernel) -> str:
        """Return the code of the number of values a member kernel writes, its outputs times its rows."""
        if isinstance(kernel.pairs, EdgeSearch):
            rows, _ = self._edge_rows(kernel.pairs)
            return f'{len(kernel.outputs)} * L[{rows[2:-1]}]'
        if any(NumbaPrinter._is_gathered(reference) for reference in kernel.references.values()):
            return f'{len(kernel.outputs)} * {kernel.rows}'
        entry = next((item.entry for item in kernel.references.values() if item.source is ReadKind.MEMBER), None)
        if entry is None:
            return str(len(kernel.outputs))
        return f'{len(kernel.outputs)} * L[{2 * self._slot(entry) + 1}]'

    def _pair_block(
        self,
        kernel: LawKernel,
        selected: list[tuple[int, OutputSpec]],
        *,
        first: str,
        second: str,
        second_positions: str,
        depth: int,
    ) -> list[str]:
        """Return the lines that add the outputs of one pair orientation to their accumulators within the cutoff."""
        pairs = self._pair_search_of(kernel)
        pad = INDENT * depth
        lines = []
        for column in range(pairs.first.columns):
            lines.append(f'{pad}_d{column}_ = positions_a[{first}, {column}] - {second_positions}[{second}, {column}]')
            if pairs.period is not None:
                lines.append(f'{pad}_d{column}_ -= PERIOD * np.rint(_d{column}_ / PERIOD)')
        distance = ' + '.join(f'_d{column}_ * _d{column}_' for column in range(pairs.first.columns))
        lines.append(f'{pad}if {distance} <= _cutoff_ * _cutoff_:')

        def read(item: Expression) -> str:
            reference = kernel.references[item]
            if NumbaPrinter._is_shared(reference):
                return str(item)
            if reference.source is ReadKind.SECOND and reference.is_state and reference.entry == pairs.second:
                return f'positions_a[{first}, {reference.column}] - _d{reference.column}_'
            member = first if reference.source is ReadKind.FIRST else second
            return f'{NumbaPrinter._vector(reference)}[{self._index(reference, member)}]'

        body, values = self._emit_values([output for _, output in selected], read, depth + 1)
        lines += body
        lines += [
            f'{INDENT * (depth + 1)}_acc{index}_ += {value}' for (index, _), value in zip(selected, values, strict=True)
        ]
        return lines

    def _pair_loop(self, kernel: LawKernel, outputs: list[tuple[int, OutputSpec]], loop: str, same: bool) -> list[str]:
        """Return the lines of one loop over members and their neighbours that accumulates pair outputs.

        The loop over `_a_` runs over the first members and computes the outputs of first members with `first=_a_` and
        `second=_n_`. The outputs of second members are computed with `first=_n_` and `second` the loop member, either
        in the same loop for pairs inside one group or in a loop over `_b_` for two groups.
        """
        cells = 'a' if same or loop == '_b_' else 'b'
        own = 'a' if loop == '_a_' else 'b'
        second_positions = 'positions_a' if same else 'positions_b'
        pairs = self._pair_search_of(kernel)
        n_dim = pairs.first.columns
        lines = [f'{INDENT}for _i_ in prange(rows_{loop}):', f'{INDENT * 2}{loop} = order_{own}[_i_]']
        lines += [f'{INDENT * 2}_acc{index}_ = 0.0' for index, _ in outputs]
        for dimension in range(n_dim):
            value = f'positions_{own}[{loop}, {dimension}]'
            lines.append(f'{INDENT * 2}_c{dimension}_ = _cell_coordinate({value}, {dimension}, shape_{cells}, _box_)')
        depth = 2
        for dimension in range(n_dim):
            pad = INDENT * depth
            counts = f'shape_{cells}[0][{dimension}]'
            lines.append(f'{pad}for _o{dimension}_ in range(-1, 2):')
            if pairs.period is not None:
                lines += [
                    (
                        f'{pad}{INDENT}if (_o{dimension}_ == -1 and {counts} < 3) or '
                        f'(_o{dimension}_ == 1 and {counts} < 2):'
                    ),
                    f'{pad}{INDENT * 2}continue',
                    f'{pad}{INDENT}_g{dimension}_ = (_c{dimension}_ + _o{dimension}_) % {counts}',
                ]
            else:
                lines += [
                    f'{pad}{INDENT}_g{dimension}_ = _c{dimension}_ + _o{dimension}_',
                    f'{pad}{INDENT}if _g{dimension}_ < 0 or _g{dimension}_ >= {counts}:',
                    f'{pad}{INDENT * 2}continue',
                ]
            depth += 1
        pad = INDENT * depth
        flat = ' + '.join(f'_g{dimension}_ * strides_{cells}[{dimension}]' for dimension in range(n_dim))
        lines += [
            f'{pad}_cell_ = {flat}',
            f'{pad}for _k_ in range(starts_{cells}[_cell_], starts_{cells}[_cell_ + 1]):',
            f'{pad}{INDENT}_n_ = order_{cells}[_k_]',
        ]
        base = depth + 1
        if same:
            lines += [f'{INDENT * base}if _n_ == _a_:', f'{INDENT * (base + 1)}continue']
        firsts, seconds = self._by_end(outputs)
        for selected, first, second, condition in ((firsts, '_a_', '_n_', '>'), (seconds, '_n_', loop, '<')):
            if not selected:
                continue
            depth = base
            if same:
                lines.append(f'{INDENT * base}if _n_ {condition} _a_:')
                depth = base + 1
            lines += self._pair_block(
                kernel,
                selected,
                first=first,
                second=second,
                second_positions=second_positions,
                depth=depth,
            )
        lines += [
            f'{INDENT * 2}out[{self._index(self._target_of(output), loop)}] += _acc{index}_'
            for index, output in outputs
        ]
        return lines

    @staticmethod
    def _written(output: OutputSpec, index: str, value: str, mode: str) -> str:
        """Return the line that adds or assigns the value of an output, or adds the change of a combined output."""
        if mode == 'add':
            return f'{INDENT * 2}out[{index}] += {value}'
        if output.combined:
            return f'{INDENT * 2}out[{index}] += ({value}) - x[{index}]'
        return f'{INDENT * 2}out[{index}] = {value}'

    def _reduced(self, output: OutputSpec, index: str, value: str) -> str:
        """Return the line that adds the share of the row `_m_` of an output that reduces over rows to its member."""
        counts = np.asarray(output.counts, dtype=np.int64)
        digest = hashlib.sha256(counts.tobytes()).hexdigest()[:16]
        count = f'L[_L{2 * self._data_slot(f"__counts_{digest}", counts)}_ + _m_]'
        if output.reduce == 'sum':
            return f'{INDENT * 2}out[{index}] += ({value}) - x[{index}] / {count}'
        return f'{INDENT * 2}out[{index}] += (({value}) - x[{index}]) / {count}'

    def _member_written(self, output: OutputSpec, value: str, mode: str) -> str:
        """Return the line that writes the value of an output of a member kernel for the row `_m_`."""
        target = self._target_of(output)
        member = '_m_' if target.source is ReadKind.MEMBER else None
        member = self._gathered_member(target) if NumbaPrinter._is_gathered(target) else member
        if output.reduce is not None:
            return self._reduced(output, self._index(target, member), value)
        return NumbaPrinter._written(output, self._index(target, member), value, mode)

    def _member_kernel(self, kernel: LawKernel, name: str, mode: str, parallel: bool) -> None:
        """Add the source of a kernel evaluated once per member, for adding, assigning, or concatenating outputs."""
        member_entry = next(
            (reference.entry for reference in kernel.references.values() if reference.source is ReadKind.MEMBER),
            None,
        )
        gathered = any(NumbaPrinter._is_gathered(reference) for reference in kernel.references.values())
        rows = f'_L{2 * self._slot(member_entry) + 1}_' if member_entry is not None else '1'
        rows = str(kernel.rows) if gathered and member_entry is None else rows
        lines = [
            f'@njit(cache=CACHE, parallel={parallel and not gathered})',
            f'def {name}(t, x, p, L, out, offset):',
            f'{INDENT}rows = {rows}',
        ]

        lines += self._single_reads(kernel, kernel.outputs)

        def read_single(item: Expression) -> str:
            reference = kernel.references[item]
            if NumbaPrinter._is_shared(reference):
                return str(item)
            if NumbaPrinter._is_gathered(reference):
                return f'{NumbaPrinter._vector(reference)}[{self._index(reference, self._gathered_member(reference))}]'
            if reference.source is ReadKind.NEIGHBOR:
                read = reference.shifted
                if read.sizes is None or read.steps is None:
                    error_msg = f'The neighbour read of {reference.entry.name} is not on a lattice.'
                    raise ValidationError(error_msg)
                return (
                    f'_neighbor_value({NumbaPrinter._vector(reference)}, _m_, (_L{2 * self._slot(reference.entry)}_, '
                    f'{reference.entry.columns}, {reference.column}), ({tuple(read.sizes)}, {tuple(read.steps)}), '
                    f'({read.mode}, {float(read.ghost or 0.0)}))'
                )
            return f'{NumbaPrinter._vector(reference)}[{self._index(reference, '_m_')}]'

        sums = [
            index
            for index, output in enumerate(kernel.outputs)
            if output.target is not None and output.target.source is ReadKind.SINGLE and mode == 'add'
        ]
        lines += [f'{INDENT}_sum{index}_ = 0.0' for index in sums]
        lines.append(f'{INDENT}for _m_ in prange(rows):')
        body, values = self._emit_values(kernel.outputs, read_single, 2)
        lines += body
        for index, (output, value) in enumerate(zip(kernel.outputs, values, strict=True)):
            if mode == 'concatenate':
                lines.append(f'{INDENT * 2}out[offset + {index} * rows + _m_] = {value}')
            elif index in sums:
                lines.append(f'{INDENT * 2}_sum{index}_ += {value}')
            else:
                lines.append(self._member_written(output, value, mode))
        for index in sums:
            target = self._target_of(kernel.outputs[index])
            lines.append(f'{INDENT}out[{self._index(target, None)}] += _sum{index}_')
        lines.append(f'{INDENT}return offset + {len(kernel.outputs)} * rows')
        self.code_ += ['', '', *NumbaPrinter._hoist_layout(lines)]

    def _pair_kernel(self, kernel: LawKernel, name: str, parallel: bool) -> None:
        """Add the source of a pair kernel that sums pair outputs over neighbours found in the bounded cell grids."""
        pairs = self._pair_search_of(kernel)
        same = pairs.first == pairs.second
        columns = pairs.first.columns
        cutoff = ReferenceSpec(pairs.cutoff, is_state=False, column=0)
        lines = [
            f'@njit(cache=CACHE, parallel={parallel})',
            f'def {name}(t, x, p, L, out, offset):',
            f'{INDENT}_cutoff_ = p[{self._index(cutoff, None)}]',
            f'{INDENT}_box_ = (np.full({columns}, PERIOD), PERIODIC)',
        ]
        for label, entry in (('a', pairs.first), ('b', pairs.second)) if not same else (('a', pairs.first),):
            start, rows = f'_L{2 * self._slot(entry)}_', f'_L{2 * self._slot(entry) + 1}_'
            cells = ', '.join(f'{part}_{label}' for part in ('shape', 'strides', 'starts', 'order'))
            positions = f'x[{start} : {start} + rows__{label}_ * {columns}].reshape((rows__{label}_, {columns}))'
            lines += [
                f'{INDENT}rows__{label}_ = {rows}',
                f'{INDENT}positions_{label} = {positions}',
                f'{INDENT}{cells} = _build_grid(positions_{label}, _cutoff_, _box_)',
            ]
        lines += self._single_reads(kernel, kernel.outputs)
        outputs = list(enumerate(kernel.outputs))
        if same:
            lines += self._pair_loop(kernel, outputs, '_a_', same=True)
        else:
            firsts, seconds = self._by_end(outputs)
            if firsts:
                lines += self._pair_loop(kernel, firsts, '_a_', same=False)
            if seconds:
                lines += self._pair_loop(kernel, seconds, '_b_', same=False)
        lines.append(f'{INDENT}return offset')
        period = pairs.period if pairs.period is not None else 1.0
        text = '\n'.join(NumbaPrinter._hoist_layout(lines)).replace('PERIODIC', str(pairs.period is not None))
        text = text.replace('PERIOD', repr(float(period)))
        self.code_ += ['', '', text]

    def _edge_kernel(self, kernel: LawKernel, name: str, mode: str) -> None:
        """Add the source of a kernel evaluated once per edge, whose laws read the ends and the edge member."""
        search = kernel.pairs
        assert isinstance(search, EdgeSearch)
        rows, ends = self._edge_rows(search)
        lines = ['@njit(cache=CACHE)', f'def {name}(t, x, p, L, out, offset):', f'{INDENT}rows = {rows}']
        lines += self._single_reads(kernel, kernel.outputs)
        members = {ReadKind.MEMBER: '_m_', ReadKind.FIRST: '_a_', ReadKind.SECOND: '_b_'}

        def read(item: Expression) -> str:
            reference = kernel.references[item]
            if NumbaPrinter._is_shared(reference):
                return str(item)
            return f'{NumbaPrinter._vector(reference)}[{self._index(reference, members[reference.source])}]'

        sums = [
            index
            for index, output in enumerate(kernel.outputs)
            if output.target is not None and output.target.source is ReadKind.SINGLE and mode == 'add'
        ]
        lines += [f'{INDENT}_sum{index}_ = 0.0' for index in sums]
        lines += [f'{INDENT}for _m_ in range(rows):', *ends]
        body, values = self._emit_values(kernel.outputs, read, 2)
        lines += body
        for index, (output, value) in enumerate(zip(kernel.outputs, values, strict=True)):
            if mode == 'concatenate':
                lines.append(f'{INDENT * 2}out[offset + {index} * rows + _m_] = {value}')
            elif index in sums:
                lines.append(f'{INDENT * 2}_sum{index}_ += {value}')
            else:
                target = output.target
                assert target is not None
                lines.append(NumbaPrinter._written(output, self._index(target, members[target.source]), value, mode))
        for index in sums:
            target = self._target_of(kernel.outputs[index])
            lines.append(f'{INDENT}out[{self._index(target, None)}] += _sum{index}_')
        lines.append(f'{INDENT}return offset + {len(kernel.outputs)} * rows')
        self.code_ += ['', '', *NumbaPrinter._hoist_layout(lines)]

    def _entry_function(
        self,
        name: str,
        kernels: list[str],
        mode: str,
        size_expression: str,
        *,
        reduced: bool,
    ) -> None:
        """Add a top-level function that evaluates the reductions and runs kernels into a new output vector."""
        start = {'add': 'np.zeros(x.size)', 'assign': 'x.copy()', 'concatenate': f'np.empty({size_expression})'}[mode]
        lines = [
            '@njit(cache=CACHE)',
            f'def {name}(t, x, p, L):',
            *([f'{INDENT}p = _reduce(t, x, p, L)'] if reduced else []),
            f'{INDENT}out = {start}',
            f'{INDENT}offset = 0',
        ]
        lines += [f'{INDENT}offset = {kernel}(t, x, p, L, out, offset)' for kernel in kernels]
        lines.append(f'{INDENT}return out')
        self.code_ += ['', '', *lines]

    def _reduction_read(self, name: str, read: ReadSpec, index: int) -> str:
        """Return the line that reads one symbol of a reduction, for the member `_m_` or once for a shared value."""
        vector = 'x' if read.is_state else 'p'
        if read.entry is None:
            array = f'_set{index}_{name}_'
            self.functions_[array] = np.asarray(read.index, dtype=np.int64)
            return f'{name} = {vector}[{array}[_m_]]'
        start = f'_L{2 * self._slot(read.entry)}_'
        if read.per_member:
            return f'{name} = {vector}[{start} + _m_ * {read.stride} + {read.column}]'
        return f'{name} = {vector}[{start} + {read.column}]'

    def _reduction_rows(self, reduction: ReductionKernel) -> str:
        """Return the code of the number of members a reduction reduces over."""
        if reduction.members is not None:
            return f'_L{2 * self._slot(reduction.members) + 1}_'
        return str(reduction.size)

    def _reduction_loop(
        self,
        reduction: ReductionKernel,
        index: int,
        body: list[str],
        *,
        parallel: bool,
    ) -> list[str]:
        """Return a loop over the reduced members that reads their symbols and runs a body."""
        reads = [
            f'{INDENT * 2}{self._reduction_read(str(item), read, index)}'
            for item, read in reduction.reads.items()
            if read.per_member
        ]
        condition = reduction.condition if reduction.condition is not None else 1
        printer = ExpressionPrinter('scalar', prefix=f'_r{index}cse').check().prepare()
        assignments, codes = self._printed(printer, [reduction.argument, condition])
        loop = 'prange' if parallel else 'range'
        return [
            f'{INDENT}for _m_ in {loop}(_n{index}_):',
            *reads,
            *(f'{INDENT * 2}{name} = {code}' for name, code in assignments),
            f'{INDENT * 2}_a_ = {codes[0]}',
            f'{INDENT * 2}_c_ = {codes[1]}',
            *(f'{INDENT * 2}{line}' for line in body),
        ]

    def _shaped_weights_lines(self, reduction: ReductionKernel, weights: ReadSpec, index: int, count: str) -> list[str]:
        """Return the lines of a reduction weighted by a shaped parameter, a product with its row-major block."""
        entry = weights.entry if weights.entry is not None else reduction.entry
        readers, size = entry.shape
        vector = 'x' if weights.is_state else 'p'
        start, block = f'_L{2 * self._slot(reduction.entry)}_', f'_L{2 * self._slot(entry)}_'
        cnt = f'_cnt{index}_'
        lines = [f'{INDENT}_values{index}_ = np.empty(_n{index}_)']
        body = [f'_values{index}_[_m_] = _a_ if _c_ else 0.0', count]
        lines += self._reduction_loop(reduction, index, body, parallel=True)
        divisor = f' / {cnt} if {cnt} > 0.0 else 0.0' if reduction.kind == 'mean' else ''
        return [
            *lines,
            f'{INDENT}for _i_ in prange({readers}):',
            f'{INDENT * 2}_s_ = 0.0',
            f'{INDENT * 2}for _k_ in range({size}):',
            f'{INDENT * 3}_s_ += {vector}[{block} + _i_ * {size} + _k_] * _values{index}_[_k_]',
            f'{INDENT * 2}p[{start} + _i_] = _s_{divisor}',
        ]

    def _log_sum_exp_lines(self, reduction: ReductionKernel, index: int, start: str) -> list[str]:
        """Return the lines of a log-sum-exp reduction, a sum of exponentials shifted by the largest value."""
        acc, cnt, top = f'_acc{index}_', f'_cnt{index}_', f'_top{index}_'
        largest = [
            f'if _c_ and ({cnt} == 0.0 or _a_ > {top}):',
            f'{INDENT}{top} = _a_',
            f'{cnt} += 1.0 if _c_ else 0.0',
        ]
        shifted = [f'{acc} += np.exp(_a_ - {top}) if _c_ else 0.0']
        return [
            f'{INDENT}{top} = 0.0',
            *self._reduction_loop(reduction, index, largest, parallel=False),
            *self._reduction_loop(reduction, index, shifted, parallel=True),
            f'{INDENT}p[{start}] = {top} + np.log({acc}) if {cnt} > 0.0 else 0.0',
        ]

    def _reduction_lines(self, reduction: ReductionKernel, index: int) -> list[str]:
        """Return the lines that compute one reduction and store it in the extended parameter vector."""
        start = f'_L{2 * self._slot(reduction.entry)}_'
        acc, cnt = f'_acc{index}_', f'_cnt{index}_'
        lines = [f'{INDENT}_n{index}_ = {self._reduction_rows(reduction)}']
        lines += [
            f'{INDENT}{self._reduction_read(str(item), read, index)}'
            for item, read in reduction.reads.items()
            if not read.per_member
        ]
        lines += [f'{INDENT}{acc} = 0.0', f'{INDENT}{cnt} = 0.0']
        sums = [f'{acc} += _a_ if _c_ else 0.0', f'{cnt} += 1.0 if _c_ else 0.0']
        mean = f'({acc} / {cnt} if {cnt} > 0.0 else 0.0)'
        if isinstance(reduction.weights, ReadSpec):
            return lines + self._shaped_weights_lines(reduction, reduction.weights, index, sums[1])
        if reduction.weights is not None:
            weights = scipy.sparse.csr_array(reduction.weights)
            for part in ('indptr', 'indices', 'data'):
                self.functions_[f'_w{index}_{part}_'] = np.asarray(getattr(weights, part))
            lines.append(f'{INDENT}_values{index}_ = np.empty(_n{index}_)')
            body = [f'_values{index}_[_m_] = _a_ if _c_ else 0.0', sums[1]]
            lines += self._reduction_loop(reduction, index, body, parallel=True)
            divisor = f' / {cnt} if {cnt} > 0.0 else 0.0' if reduction.kind == 'mean' else ''
            lines += [
                f'{INDENT}for _i_ in prange({weights.shape[0]}):',
                f'{INDENT * 2}_s_ = 0.0',
                f'{INDENT * 2}for _k_ in range(_w{index}_indptr_[_i_], _w{index}_indptr_[_i_ + 1]):',
                f'{INDENT * 3}_s_ += _w{index}_data_[_k_] * _values{index}_[_w{index}_indices_[_k_]]',
                f'{INDENT * 2}p[{start} + _i_] = _s_{divisor}',
            ]
            return lines
        if reduction.kind in {'max', 'min'}:
            better = '>' if reduction.kind == 'max' else '<'
            body = [
                f'if _c_ and ({cnt} == 0.0 or _a_ {better} {acc}):',
                f'{INDENT}{acc} = _a_',
                f'{cnt} += 1.0 if _c_ else 0.0',
            ]
            lines += self._reduction_loop(reduction, index, body, parallel=False)
            return [*lines, f'{INDENT}p[{start}] = {acc}']
        if reduction.kind == 'logsumexp':
            return lines + self._log_sum_exp_lines(reduction, index, start)
        lines += self._reduction_loop(reduction, index, sums, parallel=True)
        if reduction.kind == 'var':
            lines.append(f'{INDENT}_mean{index}_ = {mean}')
            lines.append(f'{INDENT}_sq{index}_ = 0.0')
            body = [f'_sq{index}_ += (_a_ - _mean{index}_) ** 2 if _c_ else 0.0']
            lines += self._reduction_loop(reduction, index, body, parallel=True)
            return [*lines, f'{INDENT}p[{start}] = _sq{index}_ / {cnt} if {cnt} > 0.0 else 0.0']
        value = {'sum': acc, 'count': cnt, 'mean': mean}[reduction.kind]
        return [*lines, f'{INDENT}p[{start}] = {value}']

    def _search_lines(self, reduction: NeighborhoodKernel, index: int) -> list[str]:
        """Return the lines that find the neighbours of every member, as rows `_ip<k>_` and neighbours `_ix<k>_`."""
        search = reduction.search
        if isinstance(search, StaticSearch):
            rows = self._data_slot(f'__neighbours_{index}_indptr', search.indptr)
            neighbours = self._data_slot(f'__neighbours_{index}_indices', search.indices)
            lines = [
                f'{INDENT}_ip{index}_ = L[_L{2 * rows}_ : _L{2 * rows}_ + _L{2 * rows + 1}_]',
                f'{INDENT}_ix{index}_ = L[_L{2 * neighbours}_ : _L{2 * neighbours}_ + _L{2 * neighbours + 1}_]',
            ]
            if search.edges is not None:
                edges = self._data_slot(f'__neighbours_{index}_edges', search.edges)
                lines.append(f'{INDENT}_ie{index}_ = L[_L{2 * edges}_ : _L{2 * edges}_ + _L{2 * edges + 1}_]')
            return lines
        if isinstance(search, EdgeRowSearch):
            starts = ', '.join(f'_L{2 * self._slot(entry)}_' for entry in (search.source, search.target, search.active))
            call = (
                f'_edge_neighbours(p, ({starts}), _L{2 * self._slot(search.source) + 1}_, {search.size}, '
                f'({search.directed}, {search.center}))'
            )
            return [f'{INDENT}_ip{index}_, _ix{index}_, _ie{index}_ = {call}']
        assert isinstance(search, MetricSearch)
        entry = search.positions
        start, columns = f'_L{2 * self._slot(entry)}_', entry.columns
        periods = search.periods if search.periods is not None else (1.0,) * columns
        periodic = search.periods is not None
        lines = [
            (
                f'{INDENT}_pos{index}_ = x[{start} : {start} + {reduction.size} * {columns}]'
                f'.reshape(({reduction.size}, {columns}))'
            ),
            f'{INDENT}_per{index}_ = np.array([{", ".join(repr(float(item)) for item in periods)}])',
        ]
        neighbourhood = search.neighbourhood
        center = bool(neighbourhood.center)
        if isinstance(neighbourhood, NearestNeighborhood):
            call = f'_nearest_neighbours(_pos{index}_, {neighbourhood.k}, _per{index}_, {periodic}, {center})'
            return [*lines, f'{INDENT}_ip{index}_, _ix{index}_ = {call}']
        if not isinstance(neighbourhood, ConeNeighborhood):
            call = f'_radius_neighbours(_pos{index}_, {neighbourhood.radius!r}, _per{index}_, {periodic}, {center})'
            return [*lines, f'{INDENT}_ip{index}_, _ix{index}_ = {call}']
        heading = search.direction
        assert heading is not None
        first = f'_L{2 * self._slot(heading.entry)}_'
        if heading.angle:
            lines += [
                f'{INDENT}_hd{index}_ = np.empty(({reduction.size}, 2))',
                f'{INDENT}for _i_ in range({reduction.size}):',
                f'{INDENT * 2}_hd{index}_[_i_, 0] = math.cos(x[{first} + _i_])',
                f'{INDENT * 2}_hd{index}_[_i_, 1] = math.sin(x[{first} + _i_])',
            ]
        else:
            lines.append(
                f'{INDENT}_hd{index}_ = x[{first} : {first} + {reduction.size} * {columns}].reshape(({reduction.size}, '
                f'{columns}))',
            )
        cosine = repr(float(np.cos(neighbourhood.angle)))
        call = (
            f'_cone_neighbours(_pos{index}_, _hd{index}_, ({neighbourhood.radius!r}, {cosine}), (_per{index}_, '
            f'{periodic}), {center})'
        )
        return [*lines, f'{INDENT}_ip{index}_, _ix{index}_ = {call}']

    def _pair_read_code(self, reduction: NeighborhoodKernel, item: Expression, index: int) -> str:
        """Return the code of one symbol of a neighbourhood reduction for the reader `_i_` and the neighbour `_j_`."""
        pair = reduction.reads[item]
        read = pair.read
        vector = 'x' if read.is_state else 'p'
        assert read.entry is not None
        start = f'_L{2 * self._slot(read.entry)}_'
        if pair.side == 'shared':
            return f'{vector}[{start} + {read.column}]'
        member = {'reader': '_i_', 'edge': f'_ie{index}_[_e_]', 'pair': '_e_'}.get(pair.side, '_j_')
        if pair.table is not None:
            slot = self._data_slot(f'__neighbours_{index}_{item}', pair.table)
            member = f'L[_L{2 * slot}_ + _j_]'
        value = f'{vector}[{start} + {member} * {read.stride} + {read.column}]'
        if not pair.image:
            return value
        search = reduction.search
        assert isinstance(search, MetricSearch)
        assert search.periods is not None
        period = repr(float(search.periods[read.column]))
        own = f'{vector}[{start} + _i_ * {read.stride} + {read.column}]'
        del index
        return f'{own} - (({own} - {value}) - {period} * np.rint(({own} - {value}) / {period}))'

    def _component_lines(
        self,
        reduction: NeighborhoodKernel,
        index: int,
        assignments: Sequence[tuple[str, str]],
        active: str,
    ) -> list[str]:
        """Return the lines that collect the active pairs of every member and write the values of their components."""
        sides = {item: reduction.reads[item].side for item in reduction.symbols}
        readers = [item for item in reduction.symbols if sides[item] == 'reader']
        others = [item for item in reduction.symbols if sides[item] in {'other', 'edge', 'pair'}]
        firsts, seconds, count = f'_cf{index}_', f'_cs{index}_', f'_cn{index}_'
        target = f'_L{2 * self._slot(reduction.entry)}_'
        size = reduction.size
        call = f'_component_values({size}, {firsts}, {seconds}, {count}, {COMPONENT_KINDS[reduction.kind]})'
        return [
            f'{INDENT}{firsts} = np.empty(_ip{index}_[{size}], dtype=np.int64)',
            f'{INDENT}{seconds} = np.empty(_ip{index}_[{size}], dtype=np.int64)',
            f'{INDENT}{count} = 0',
            f'{INDENT}for _i_ in range({size}):',
            *(f'{INDENT * 2}{item} = {self._pair_read_code(reduction, item, index)}' for item in readers),
            f'{INDENT * 2}for _e_ in range(_ip{index}_[_i_], _ip{index}_[_i_ + 1]):',
            f'{INDENT * 3}_j_ = _ix{index}_[_e_]',
            *(f'{INDENT * 3}{item} = {self._pair_read_code(reduction, item, index)}' for item in others),
            *(f'{INDENT * 3}{name} = {code}' for name, code in assignments),
            f'{INDENT * 3}if {active} != 0.0:',
            f'{INDENT * 4}{firsts}[{count}] = _i_',
            f'{INDENT * 4}{seconds}[{count}] = _j_',
            f'{INDENT * 4}{count} += 1',
            f'{INDENT}p[{target} : {target} + {size}] = {call}',
        ]

    def _neighborhood_lines(self, reduction: NeighborhoodKernel, index: int) -> list[str]:
        """Return the lines that compute a neighbourhood reduction of every member into the extended parameters."""
        lines = self._search_lines(reduction, index)
        sides = {item: reduction.reads[item].side for item in reduction.symbols}
        shared = [item for item in reduction.symbols if sides[item] == 'shared']
        readers = [item for item in reduction.symbols if sides[item] == 'reader']
        others = [item for item in reduction.symbols if sides[item] in {'other', 'edge', 'pair'}]
        lines += [f'{INDENT}{item} = {self._pair_read_code(reduction, item, index)}' for item in shared]
        printer = ExpressionPrinter('scalar', prefix=f'_n{index}cse').check().prepare()
        assignments, codes = self._printed(printer, [reduction.argument, reduction.weight])
        kind = reduction.kind
        if kind in COMPONENT_KINDS:
            return lines + self._component_lines(reduction, index, assignments, codes[0])
        acc, weights, found = f'_acc{index}_', f'_ws{index}_', f'_found{index}_'
        if kind == 'argmax':
            accumulate = [
                f'if not {found} or _w_ > {weights}:',
                f'{INDENT}{acc} = _a_',
                f'{INDENT}{weights} = _w_',
                f'{INDENT}{found} = True',
            ]
            value = f'{acc} if {found} else 0.0'
        elif kind in {'max', 'min'}:
            better = '>' if kind == 'max' else '<'
            accumulate = [
                '_v_ = _w_ * _a_',
                f'if not {found} or _v_ {better} {acc}:',
                f'{INDENT}{acc} = _v_',
                f'{INDENT}{found} = True',
            ]
            value = f'{acc} if {found} else 0.0'
        elif kind == 'count':
            accumulate = [f'{acc} += _w_ * (1.0 if _a_ != 0.0 else 0.0)']
            value = acc
        else:
            accumulate = [f'{acc} += _w_ * _a_', f'{weights} += _w_']
            count = f'(_ip{index}_[_i_ + 1] - _ip{index}_[_i_])'
            value = {
                'sum': acc,
                'mean': f'{acc} / {count} if {count} > 0 else 0.0',
                'normalized-mean': f'{acc} / {weights} if {weights} != 0.0 else 0.0',
            }[kind]
        target = f'_L{2 * self._slot(reduction.entry)}_'
        lines += [
            f'{INDENT}for _i_ in prange({reduction.size}):',
            *(f'{INDENT * 2}{item} = {self._pair_read_code(reduction, item, index)}' for item in readers),
            f'{INDENT * 2}{acc} = 0.0',
            f'{INDENT * 2}{weights} = 0.0',
            f'{INDENT * 2}{found} = False',
            f'{INDENT * 2}for _e_ in range(_ip{index}_[_i_], _ip{index}_[_i_ + 1]):',
            f'{INDENT * 3}_j_ = _ix{index}_[_e_]',
            *(f'{INDENT * 3}{item} = {self._pair_read_code(reduction, item, index)}' for item in others),
            *(f'{INDENT * 3}{name} = {code}' for name, code in assignments),
            f'{INDENT * 3}_a_ = {codes[0]}',
            f'{INDENT * 3}_w_ = {codes[1]}',
            *(f'{INDENT * 3}{line}' for line in accumulate),
            f'{INDENT * 2}p[{target} + _i_] = {value}',
        ]
        return lines

    def _reduce_function(self, reductions: ReductionScheduleSpec, parallel: bool) -> None:
        """Add the function that returns the parameter vector extended with every reduction, in dependency order."""
        extents = [
            f'_L{2 * self._slot(item.entry)}_ + _L{2 * self._slot(item.entry) + 1}_ * {item.entry.columns}'
            for item in reductions.steps
        ]
        size = extents[0] if len(extents) == 1 else f'max({", ".join(extents)})'
        lines = [
            f'@njit(cache=CACHE, parallel={parallel})',
            'def _reduce(t, x, p0, L):',
            f'{INDENT}p = np.empty({size})',
            f'{INDENT}p[: p0.size] = p0',
        ]
        for index, reduction in enumerate(reductions.steps):
            if isinstance(reduction, NeighborhoodKernel):
                lines += self._neighborhood_lines(reduction, index)
            else:
                lines += self._reduction_lines(reduction, index)
        lines.append(f'{INDENT}return p')
        self.code_ += ['', '', *NumbaPrinter._hoist_layout(lines)]
