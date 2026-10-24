"""Hold the generated source of Numba kernels, with its layout slots and the code that reads symbols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ..base import Expression, symbols_of
from ..compile._kernels import Kernel, Output, Reference, Source
from ..compile._layout import Entry
from ..compile._printing import SCALAR_DIALECT, print_expressions

INDENT = '    '


@dataclass
class _Source:
    """Generated source with the layout slots and the compiled functions it refers to."""

    lines: list[str] = field(default_factory=list)
    slots: dict[str, int] = field(default_factory=dict)
    functions: dict[str, object] = field(default_factory=dict)
    data: dict[str, np.ndarray] = field(default_factory=dict)

    def slot(self, entry: Entry) -> int:
        """Return the layout slot of an entry, whose start is `L[2 * slot]` and rows `L[2 * slot + 1]`.

        Args:
            entry:
                The entry.

        Returns:
            slot:
                The slot.
        """
        return self.slots.setdefault(entry.name, len(self.slots))

    def data_slot(self, name: str, values: np.ndarray) -> int:
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
        self.data[name] = np.asarray(values, dtype=np.int64)
        return self.slots.setdefault(name, len(self.slots))

    def layout(self, entries: Mapping[str, Entry]) -> np.ndarray:
        """Return the layout array of starts and rows of every slot, followed by the data of the data slots.

        Args:
            entries:
                Every entry by name.

        Returns:
            layout:
                The layout array.
        """
        layout = np.zeros(2 * len(self.slots), dtype=np.int64)
        offset = layout.size
        for name, slot in self.slots.items():
            if name in self.data:
                layout[2 * slot], layout[2 * slot + 1] = offset, self.data[name].size
                offset += self.data[name].size
            else:
                layout[2 * slot], layout[2 * slot + 1] = entries[name].start, entries[name].rows
        return np.concatenate([layout, *(self.data[name] for name in self.slots if name in self.data)])

    def function_name(self, function: object) -> str:
        """Return the global name of a compiled function called from the generated source.

        Args:
            function:
                The Numba-compiled function.

        Returns:
            name:
                The global name.
        """
        for name, known in self.functions.items():
            if known is function:
                return name
        name = f'_function{len(self.functions)}_'
        self.functions[name] = function
        return name


def _vector(reference: Reference) -> str:
    """Return the name of the vector that holds a reference."""
    return 'x' if reference.is_state else 'p'


def _hoist_layout(lines: list[str]) -> list[str]:
    """Return function lines with every layout lookup `_L<k>_` read once after the signature."""
    used = sorted({int(index) for line in lines for index in re.findall(r'_L(\d+)_', line)})
    reads = [f'{INDENT}_L{index}_ = L[{index}]' for index in used]
    return [*lines[:2], *reads, *lines[2:]]


def _index(source: _Source, reference: Reference, member: str | None) -> str:
    """Return the flat index of a reference for a member variable, or for a single element."""
    start = f'_L{2 * source.slot(reference.entry)}_'
    if member is None:
        return f'{start} + {reference.column}'
    return f'{start} + {member} * {reference.entry.columns} + {reference.column}'


def _emit_values(
    source: _Source,
    outputs: list[Output],
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
    assignments, codes = print_expressions([*expressions, *arguments], SCALAR_DIALECT)
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
            lines.append(f'{pad}{calls[key]} = {source.function_name(output.function)}({call_arguments})')
        values.append(calls[key] if output.components == 1 else f'{calls[key]}[{output.component}]')
    return lines, values


def _is_shared(reference: Reference) -> bool:
    """Check whether a reference holds one value for every evaluation index, a single element or a reduction."""
    return reference.source is Source.SINGLE or (reference.source is Source.REDUCED and reference.entry.members is None)


def _single_reads(source: _Source, kernel: Kernel, outputs: list[Output]) -> list[str]:
    """Return the lines that read every single-element symbol of outputs once, before the member loops."""
    expressions = [output.expression for output in outputs if output.expression is not None]
    arguments = [argument for output in outputs for argument in output.arguments or ()]
    symbols = sorted(symbols_of(expressions + arguments), key=str)
    return [
        f'{INDENT}{item} = {_vector(reference)}[{_index(source, reference, None)}]'
        for item in symbols
        if (reference := kernel.references.get(item)) is not None and _is_shared(reference)
    ]
