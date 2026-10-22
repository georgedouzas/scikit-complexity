"""Generate, compile, and cache Numba kernels and stepper loops from the kernels of a problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
import importlib.util
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
import scipy.sparse
from numba.core.dispatcher import Dispatcher

from ..base import (
    NUMERIC_DOMAINS,
    ComponentKind,
    DomainKind,
    Expression,
    Form,
    LawKind,
    OrderKind,
    SpaceKind,
    Supports,
    TimeKind,
    ValidationError,
    symbols_of,
)
from ..compile._backends import (
    AUTO_MEMBERS,
    Backend,
    Lowering,
    adding_function,
    assigning_function,
    concatenating_function,
    register_backend,
)
from ..compile._kernels import Kernel, Output, Reference, Source, jump_channels, noise_rows
from ..compile._layout import Entry
from ..compile._printing import SCALAR_DIALECT, print_expressions
from ..compile._problem import Accelerated, Draws, Vector
from ..compile._reductions import CompiledReduction, Read, ReductionPass
from ..laws import Boundary, Jump, Noise, Rate, Update

DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'skcomplex' / 'kernels'
FORM_KINDS: dict[Form, list[type]] = {
    Form.ODE: [Rate],
    Form.FRACTIONAL: [Rate],
    Form.SDE: [Rate, Noise],
    Form.JUMP: [Jump],
    Form.DISCRETE: [Update],
}
INDENT = '    '
NO_BOUNDARY = """
HAS_BOUNDARY = False


@njit(cache=CACHE)
def boundary(t, x, p, L):
    return x
"""
EVENT_FORMS = (Form.JUMP, Form.DISCRETE)
HEADER = """import math

import numpy as np
from numba import njit, prange

from skcomplex.core.backends._numba_cells import build_cells
from skcomplex.core.backends._numba_stencils import neighbor_value
"""
LOOPS = {
    'rk4': """
@njit(cache=CACHE)
def rk4(t, x, p, L, until, h):
    if until <= t:
        return x.copy()
    n_steps = max(1, int(math.ceil((until - t) / h - 1e-12)))
    step = (until - t) / n_steps
    y = x.copy()
    for index in range(n_steps):
        time = t + index * step
        k1 = drift(time, y, p, L)
        k2 = drift(time + step / 2, y + step / 2 * k1, p, L)
        k3 = drift(time + step / 2, y + step / 2 * k2, p, L)
        k4 = drift(time + step, y + step * k3, p, L)
        y = y + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if HAS_BOUNDARY:
            y = boundary(time + step, y, p, L)
    return y
""",
    'velocity_verlet': """
@njit(cache=CACHE)
def velocity_verlet(t, x, p, L, until, h, positions, velocities):
    if until <= t:
        return x.copy()
    n_steps = max(1, int(math.ceil((until - t) / h - 1e-12)))
    step = (until - t) / n_steps
    y = x.copy()
    rates = drift(t, y, p, L)
    for index in range(n_steps):
        time = t + (index + 1) * step
        for component in range(velocities.size):
            y[velocities[component]] += step / 2 * rates[velocities[component]]
        for component in range(positions.size):
            y[positions[component]] += step * y[velocities[component]]
        if HAS_BOUNDARY:
            y = boundary(time, y, p, L)
        rates = drift(time, y, p, L)
        for component in range(velocities.size):
            y[velocities[component]] += step / 2 * rates[velocities[component]]
    return y
""",
    'euler_maruyama': """
@njit(cache=CACHE)
def euler_maruyama(t, x, p, L, until, h, rng, rows):
    if until <= t:
        return x.copy()
    n_steps = max(1, int(math.ceil((until - t) / h - 1e-12)))
    step = (until - t) / n_steps
    scale = math.sqrt(step)
    y = x.copy()
    for index in range(n_steps):
        time = t + index * step
        amplitudes = diffusion(time, y, p, L)
        increments = rng.normal(0.0, scale, amplitudes.size)
        y = y + drift(time, y, p, L) * step
        for process in range(amplitudes.size):
            y[rows[process]] += amplitudes[process] * increments[process]
        if HAS_BOUNDARY:
            y = boundary(time + step, y, p, L)
    return y
""",
    'gillespie': """
@njit(cache=CACHE)
def gillespie(t, x, p, L, until, rng, targets, increments):
    y = x.copy()
    if until <= t:
        return y
    while True:
        rates = propensities(t, y, p, L)
        total = 0.0
        for channel in range(rates.size):
            if not (rates[channel] >= 0.0 and rates[channel] < np.inf):
                raise ValueError('Propensities must be finite and non-negative.')
            total += rates[channel]
        if total == 0.0:
            return y
        t += rng.exponential(1.0 / total)
        if t > until:
            return y
        channel = min(np.searchsorted(np.cumsum(rates), rng.uniform(0.0, total), side='right'), rates.size - 1)
        for change in range(targets.shape[1]):
            if targets[channel, change] >= 0:
                y[targets[channel, change]] += increments[channel, change]
        if HAS_BOUNDARY:
            y = boundary(t, y, p, L)
""",
    'iterate': """
@njit(cache=CACHE)
def iterate(t, x, p, L, until):
    y = x.copy()
    for step in range(int(round(t)), int(math.floor(until + 1e-9))):
        y = update(float(step), y, p, L)
        if HAS_BOUNDARY:
            y = boundary(float(step + 1), y, p, L)
    return y
""",
    'iterate_random': """
@njit(cache=CACHE)
def iterate_random(t, x, p, L, until, rng, offset, normal):
    y = x.copy()
    q = np.zeros(offset + normal.size)
    q[:offset] = p[:offset]
    for step in range(int(round(t)), int(math.floor(until + 1e-9))):
        for index in range(normal.size):
            if normal[index]:
                q[offset + index] = rng.standard_normal()
            else:
                q[offset + index] = rng.random()
        y = update(float(step), y, q, L)
        if HAS_BOUNDARY:
            y = boundary(float(step + 1), y, q, L)
    return y
""",
}


def _is_compilable(kernel: Kernel) -> bool:
    """Check whether every output of a kernel can be compiled, so it has no Python callable."""
    lattice = all(
        (reference.source is not Source.NEIGHBOR or reference.neighbor.sizes is not None)
        and reference.source is not Source.DERIVED
        for reference in kernel.references.values()
    )
    return lattice and all(
        output.function is None or (output.arguments is not None and isinstance(output.function, Dispatcher))
        for output in kernel.outputs
    )


@dataclass
class _Source:
    """Generated source with the layout slots and the compiled functions it refers to."""

    lines: list[str] = field(default_factory=list)
    slots: dict[str, int] = field(default_factory=dict)
    functions: dict[str, object] = field(default_factory=dict)

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

    def layout(self, entries: Mapping[str, Entry]) -> np.ndarray:
        """Return the layout array of starts and rows of every slot.

        Args:
            entries:
                Every entry by name.

        Returns:
            layout:
                The layout array.
        """
        layout = np.zeros(2 * len(self.slots), dtype=np.int64)
        for name, slot in self.slots.items():
            layout[2 * slot], layout[2 * slot + 1] = entries[name].start, entries[name].rows
        return layout

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


def _member_kernel(source: _Source, kernel: Kernel, name: str, mode: str, parallel: bool) -> None:
    """Add the source of a kernel evaluated once per member, for adding, assigning, or concatenating outputs."""
    member_entry = next(
        (reference.entry for reference in kernel.references.values() if reference.source is Source.MEMBER),
        None,
    )
    rows = f'_L{2 * source.slot(member_entry) + 1}_' if member_entry is not None else '1'
    lines = [
        f'@njit(cache=CACHE, parallel={parallel})',
        f'def {name}(t, x, p, L, out, offset):',
        f'{INDENT}rows = {rows}',
    ]

    lines += _single_reads(source, kernel, kernel.outputs)

    def read_single(item: Expression) -> str:
        reference = kernel.references[item]
        if _is_shared(reference):
            return str(item)
        if reference.source is Source.NEIGHBOR:
            read = reference.neighbor
            return (
                f'neighbor_value({_vector(reference)}, _L{2 * source.slot(reference.entry)}_, '
                f'{reference.entry.columns}, {reference.column}, _m_, {tuple(read.sizes)}, {tuple(read.steps)}, '
                f'{read.mode}, {float(read.ghost or 0.0)})'
            )
        return f'{_vector(reference)}[{_index(source, reference, '_m_')}]'

    sums = [
        index
        for index, output in enumerate(kernel.outputs)
        if output.target is not None and output.target.source is Source.SINGLE and mode == 'add'
    ]
    lines += [f'{INDENT}_sum{index}_ = 0.0' for index in sums]
    lines.append(f'{INDENT}for _m_ in prange(rows):')
    body, values = _emit_values(source, kernel.outputs, read_single, 2)
    lines += body
    for index, (output, value) in enumerate(zip(kernel.outputs, values, strict=True)):
        if mode == 'concatenate':
            lines.append(f'{INDENT * 2}out[offset + {index} * rows + _m_] = {value}')
        elif index in sums:
            lines.append(f'{INDENT * 2}_sum{index}_ += {value}')
        else:
            target = output.target
            member = '_m_' if target.source is Source.MEMBER else None  # type: ignore[union-attr]
            operator = '+=' if mode == 'add' else '='
            lines.append(f'{INDENT * 2}out[{_index(source, target, member)}] {operator} {value}')  # type: ignore[arg-type]
    for index in sums:
        lines.append(f'{INDENT}out[{_index(source, kernel.outputs[index].target, None)}] += _sum{index}_')  # type: ignore[arg-type]
    lines.append(f'{INDENT}return offset + {len(kernel.outputs)} * rows')
    source.lines += ['', '', *_hoist_layout(lines)]


def _pair_block(
    source: _Source,
    kernel: Kernel,
    selected: list[tuple[int, Output]],
    *,
    first: str,
    second: str,
    second_positions: str,
    depth: int,
) -> list[str]:
    """Return the lines that add the outputs of one pair orientation to their accumulators when within the cutoff."""
    pairs = kernel.pairs
    assert pairs is not None
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
        if _is_shared(reference):
            return str(item)
        if reference.source is Source.SECOND and reference.is_state and reference.entry == pairs.second:
            return f'positions_a[{first}, {reference.column}] - _d{reference.column}_'
        member = first if reference.source is Source.FIRST else second
        return f'{_vector(reference)}[{_index(source, reference, member)}]'

    body, values = _emit_values(source, [output for _, output in selected], read, depth + 1)
    lines += body
    lines += [
        f'{INDENT * (depth + 1)}_acc{index}_ += {value}' for (index, _), value in zip(selected, values, strict=True)
    ]
    return lines


def _pair_loop(source: _Source, kernel: Kernel, outputs: list[tuple[int, Output]], loop: str, same: bool) -> list[str]:
    """Return the lines of one loop over members and their neighbours that accumulates pair outputs.

    The loop over `_a_` runs over the first members and computes the outputs of first members with `first=_a_` and
    `second=_n_`. The outputs of second members are computed with `first=_n_` and `second` the loop member, either in
    the same loop for pairs inside one group or in a loop over `_b_` for two groups.
    """
    cells = 'a' if same or loop == '_b_' else 'b'
    own = 'a' if loop == '_a_' else 'b'
    second_positions = 'positions_a' if same else 'positions_b'
    pairs = kernel.pairs
    assert pairs is not None
    n_dim = pairs.first.columns
    lines = [f'{INDENT}for _i_ in prange(rows_{loop}):', f'{INDENT * 2}{loop} = order_{own}[_i_]']
    lines += [f'{INDENT * 2}_acc{index}_ = 0.0' for index, _ in outputs]
    for dimension in range(n_dim):
        value = f'positions_{own}[{loop}, {dimension}]'
        side, count, origin = (
            f'sides_{cells}[{dimension}]',
            f'counts_{cells}[{dimension}]',
            f'origin_{cells}[{dimension}]',
        )
        coordinate = (
            f'min(int(({value} % PERIOD) // {side}), {count} - 1)'
            if pairs.period is not None
            else f'int(np.floor(({value} - {origin}) / {side}))'
        )
        lines.append(f'{INDENT * 2}_c{dimension}_ = {coordinate}')
    depth = 2
    for dimension in range(n_dim):
        pad = INDENT * depth
        counts = f'counts_{cells}[{dimension}]'
        lines.append(f'{pad}for _o{dimension}_ in range(-1, 2):')
        if pairs.period is not None:
            lines += [
                f'{pad}{INDENT}if {counts} == 1 and _o{dimension}_ != 0:',
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
    firsts = [(index, output) for index, output in outputs if output.target.source is Source.FIRST]  # type: ignore[union-attr]
    seconds = [(index, output) for index, output in outputs if output.target.source is Source.SECOND]  # type: ignore[union-attr]
    for selected, first, second, condition in ((firsts, '_a_', '_n_', '>'), (seconds, '_n_', loop, '<')):
        if not selected:
            continue
        depth = base
        if same:
            lines.append(f'{INDENT * base}if _n_ {condition} _a_:')
            depth = base + 1
        lines += _pair_block(
            source,
            kernel,
            selected,
            first=first,
            second=second,
            second_positions=second_positions,
            depth=depth,
        )
    lines += [f'{INDENT * 2}out[{_index(source, output.target, loop)}] += _acc{index}_' for index, output in outputs]  # type: ignore[arg-type]
    return lines


def _pair_kernel(source: _Source, kernel: Kernel, name: str, parallel: bool) -> None:
    """Add the source of a pair kernel that sums pair outputs over neighbours found with cell lists."""
    pairs = kernel.pairs
    assert pairs is not None
    same = pairs.first == pairs.second
    columns = pairs.first.columns
    cutoff = Reference(pairs.cutoff, is_state=False, column=0)
    lines = [
        f'@njit(cache=CACHE, parallel={parallel})',
        f'def {name}(t, x, p, L, out, offset):',
        f'{INDENT}_cutoff_ = p[{_index(source, cutoff, None)}]',
    ]
    for label, entry in (('a', pairs.first), ('b', pairs.second)) if not same else (('a', pairs.first),):
        start, rows = f'_L{2 * source.slot(entry)}_', f'_L{2 * source.slot(entry) + 1}_'
        cells = ', '.join(f'{part}_{label}' for part in ('counts', 'sides', 'origin', 'strides', 'starts', 'order'))
        positions = f'x[{start} : {start} + rows__{label}_ * {columns}].reshape((rows__{label}_, {columns}))'
        lines += [
            f'{INDENT}rows__{label}_ = {rows}',
            f'{INDENT}positions_{label} = {positions}',
            f'{INDENT}{cells} = build_cells(positions_{label}, _cutoff_, PERIOD, PERIODIC)',
        ]
    lines += _single_reads(source, kernel, kernel.outputs)
    outputs = list(enumerate(kernel.outputs))
    if same:
        lines += _pair_loop(source, kernel, outputs, '_a_', same=True)
    else:
        firsts = [(index, output) for index, output in outputs if output.target.source is Source.FIRST]  # type: ignore[union-attr]
        seconds = [(index, output) for index, output in outputs if output.target.source is Source.SECOND]  # type: ignore[union-attr]
        if firsts:
            lines += _pair_loop(source, kernel, firsts, '_a_', same=False)
        if seconds:
            lines += _pair_loop(source, kernel, seconds, '_b_', same=False)
    lines.append(f'{INDENT}return offset')
    period = pairs.period if pairs.period is not None else 1.0
    text = '\n'.join(_hoist_layout(lines)).replace('PERIODIC', str(pairs.period is not None))
    text = text.replace('PERIOD', repr(float(period)))
    source.lines += ['', '', text]


def _reduction_read(source: _Source, name: str, read: Read, index: int) -> str:
    """Return the line that reads one symbol of a reduction, for the member `_m_` or once for a shared value."""
    vector = 'x' if read.is_state else 'p'
    if read.entry is None:
        array = f'_set{index}_{name}_'
        source.functions[array] = np.asarray(read.index, dtype=np.int64)
        return f'{name} = {vector}[{array}[_m_]]'
    start = f'_L{2 * source.slot(read.entry)}_'
    if read.per_member:
        return f'{name} = {vector}[{start} + _m_ * {read.stride} + {read.column}]'
    return f'{name} = {vector}[{start} + {read.column}]'


def _reduction_rows(source: _Source, reduction: CompiledReduction) -> str:
    """Return the code of the number of members a reduction reduces over."""
    if reduction.members is not None:
        return f'_L{2 * source.slot(reduction.members) + 1}_'
    return str(reduction.size)


def _reduction_loop(
    source: _Source,
    reduction: CompiledReduction,
    index: int,
    body: list[str],
    *,
    parallel: bool,
) -> list[str]:
    """Return a loop over the reduced members that reads their symbols and runs a body."""
    reads = [
        f'{INDENT * 2}{_reduction_read(source, str(item), read, index)}'
        for item, read in reduction.reads.items()
        if read.per_member
    ]
    assignments, codes = print_expressions(
        [reduction.argument, reduction.condition if reduction.condition is not None else 1],
        SCALAR_DIALECT,
        prefix=f'_r{index}cse',
    )
    loop = 'prange' if parallel else 'range'
    return [
        f'{INDENT}for _m_ in {loop}(_n{index}_):',
        *reads,
        *(f'{INDENT * 2}{name} = {code}' for name, code in assignments),
        f'{INDENT * 2}_a_ = {codes[0]}',
        f'{INDENT * 2}_c_ = {codes[1]}',
        *(f'{INDENT * 2}{line}' for line in body),
    ]


def _reduction_lines(source: _Source, reduction: CompiledReduction, index: int) -> list[str]:
    """Return the lines that compute one reduction and store it in the extended parameter vector."""
    start = f'_L{2 * source.slot(reduction.entry)}_'
    acc, cnt = f'_acc{index}_', f'_cnt{index}_'
    lines = [f'{INDENT}_n{index}_ = {_reduction_rows(source, reduction)}']
    lines += [
        f'{INDENT}{_reduction_read(source, str(item), read, index)}'
        for item, read in reduction.reads.items()
        if not read.per_member
    ]
    lines += [f'{INDENT}{acc} = 0.0', f'{INDENT}{cnt} = 0.0']
    sums = [f'{acc} += _a_ if _c_ else 0.0', f'{cnt} += 1.0 if _c_ else 0.0']
    mean = f'({acc} / {cnt} if {cnt} > 0.0 else 0.0)'
    if reduction.weights is not None:
        weights = scipy.sparse.csr_array(reduction.weights)
        for part in ('indptr', 'indices', 'data'):
            source.functions[f'_w{index}_{part}_'] = np.asarray(getattr(weights, part))
        lines.append(f'{INDENT}_values{index}_ = np.empty(_n{index}_)')
        body = [f'_values{index}_[_m_] = _a_ if _c_ else 0.0', sums[1]]
        lines += _reduction_loop(source, reduction, index, body, parallel=True)
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
        lines += _reduction_loop(source, reduction, index, body, parallel=False)
        return [*lines, f'{INDENT}p[{start}] = {acc}']
    lines += _reduction_loop(source, reduction, index, sums, parallel=True)
    if reduction.kind == 'var':
        lines.append(f'{INDENT}_mean{index}_ = {mean}')
        lines.append(f'{INDENT}_sq{index}_ = 0.0')
        body = [f'_sq{index}_ += (_a_ - _mean{index}_) ** 2 if _c_ else 0.0']
        lines += _reduction_loop(source, reduction, index, body, parallel=True)
        return [*lines, f'{INDENT}p[{start}] = _sq{index}_ / {cnt} if {cnt} > 0.0 else 0.0']
    value = {'sum': acc, 'count': cnt, 'mean': mean}[reduction.kind]
    return [*lines, f'{INDENT}p[{start}] = {value}']


def _reduce_function(source: _Source, reductions: ReductionPass, parallel: bool) -> None:
    """Add the function that returns the parameter vector extended with every reduction, in dependency order."""
    extents = [
        f'_L{2 * source.slot(item.entry)}_ + _L{2 * source.slot(item.entry) + 1}_ * {item.entry.columns}'
        for item in reductions.reductions
    ]
    size = extents[0] if len(extents) == 1 else f'max({", ".join(extents)})'
    lines = [
        f'@njit(cache=CACHE, parallel={parallel})',
        'def _reduce(t, x, p0, L):',
        f'{INDENT}p = np.empty({size})',
        f'{INDENT}p[: p0.size] = p0',
    ]
    for index, reduction in enumerate(reductions.reductions):
        lines += _reduction_lines(source, reduction, index)
    lines.append(f'{INDENT}return p')
    source.lines += ['', '', *_hoist_layout(lines)]


def _reductions_compile(reductions: ReductionPass) -> bool:
    """Check whether every reduction prints in Numba's dialect."""
    try:
        for reduction in reductions.reductions:
            print_expressions(
                [reduction.argument, reduction.condition if reduction.condition is not None else 1],
                SCALAR_DIALECT,
            )
    except ValidationError:
        return False
    return True


def _output_count(source: _Source, kernel: Kernel) -> str:
    """Return the code of the number of values a member kernel writes, its outputs times its rows."""
    entry = next((item.entry for item in kernel.references.values() if item.source is Source.MEMBER), None)
    if entry is None:
        return str(len(kernel.outputs))
    return f'{len(kernel.outputs)} * L[{2 * source.slot(entry) + 1}]'


def _entry_function(
    source: _Source,
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
    source.lines += ['', '', *lines]


def _load(text: str, functions: Mapping[str, object], cache_dir: Path) -> ModuleType:
    """Return the module of generated source, from a cached file when it has no injected functions."""
    digest = hashlib.sha256(text.encode()).hexdigest()[:24]
    name = f'skcomplex_kernels_{digest}'
    if functions:
        module = ModuleType(name)
        module.__dict__.update(functions)
        code = compile(text.replace('cache=CACHE', 'cache=False'), name, 'exec')
        exec(code, module.__dict__)  # noqa: S102  # nosec B102: runs the kernel source generated above
        return module
    if name in sys.modules:
        return sys.modules[name]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{name}.py'
    if not path.exists():
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(text.replace('cache=CACHE', 'cache=True'))
        temporary.replace(path)
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)  # type: ignore[arg-type]
    sys.modules[name] = module
    specification.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class NumbaBackend(Backend):
    """Backend that generates fused Numba kernels and compiled stepper loops, cached on disk.

    Kinds of laws that contain a plain Python callable fall back to the NumPy functions, and the problem then has no
    compiled stepper loops.

    Args:
        cache_dir:
            The directory of the generated kernel sources and Numba's cache, or `None` for `~/.cache/skcomplex`.

        parallel:
            Whether member and pair loops over groups of at least 1000 members run in parallel threads. Kernels that
            the compiled loops of jump processes and discrete maps call once per event always run serially.

    Examples:
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowth, Population
        >>> system = System(
        ...     components=[Population.group('herds', 2, n=[1.0, 2.0])],
        ...     interactions=[ExponentialGrowth('growth', target='herds', r=3.0)],
        ... )
        >>> problem = compile_system(system, backend='numba')
        >>> problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist()
        [3.0, 6.0]
    """

    supports: ClassVar[Supports] = Supports(
        time=set(TimeKind),
        space=set(SpaceKind),
        laws=set(LawKind),
        domains=NUMERIC_DOMAINS | {DomainKind.CATEGORICAL},
        components=set(ComponentKind),
        orders=set(OrderKind),
    )

    name: ClassVar[str] = 'numba'

    def __init__(self, cache_dir: str | Path | None = None, parallel: bool = True) -> None:
        self.cache_dir = cache_dir
        self.parallel = parallel

    def _is_parallel(self, kernel: Kernel, form: Form) -> bool:
        """Check whether a kernel runs in parallel threads: outside event loops, over groups of many members."""
        if not self.parallel or form in EVENT_FORMS:
            return False
        rows = [
            reference.entry.rows
            for reference in kernel.references.values()
            if reference.source in {Source.MEMBER, Source.FIRST, Source.SECOND}
        ]
        return max(rows, default=0) >= AUTO_MEMBERS

    def _kind(
        self,
        source: _Source,
        kernels: list[Kernel],
        label: str,
        mode: str,
        *,
        form: Form,
        reduced: bool,
    ) -> tuple[str, str]:
        """Add the kernels of one law kind and return the top-level function name and its output size."""
        names = []
        for index, kernel in enumerate(kernels):
            name = f'_{label}{index}_'
            parallel = self._is_parallel(kernel, form)
            if kernel.pairs is not None:
                _pair_kernel(source, kernel, name, parallel)
            else:
                _member_kernel(source, kernel, name, mode, parallel)
            names.append(name)
        size = ' + '.join(_output_count(source, kernel) for kernel in kernels) or '0'
        _entry_function(source, label, names, mode, size, reduced=reduced)
        return label, size

    def lower(
        self,
        form: Form,
        kernels: Mapping[type, list[Kernel]],
        changes: Mapping[int, list[tuple[Reference, float]]],
        size: int,
        *,
        reductions: ReductionPass | None = None,
        draws: Draws | None = None,
    ) -> Lowering:
        """Return the numeric functions of a form compiled with Numba.

        Args:
            form:
                The form of the problem.

            kernels:
                The kernels of every law kind, in evaluation order.

            changes:
                The changes of every jump output, keyed by the output's identity.

            size:
                The size of the state vector.

            reductions:
                The reductions the kernels read, fused into one compiled function that every entry function runs
                first, or `None`.

            draws:
                The random values a stochastic map draws at every step into the parameter vector extended after its
                parameters, which the compiled loop takes from the generator, or `None` for a deterministic problem.

        Returns:
            lowering:
                The numeric functions.
        """
        used = {
            Boundary: kernels.get(Boundary, []),
            Rate: kernels.get(Rate, []) if form in (Form.ODE, Form.FRACTIONAL, Form.SDE) else [],
            Noise: kernels.get(Noise, []) if form is Form.SDE else [],
            Jump: kernels.get(Jump, []) if form is Form.JUMP else [],
            Update: kernels.get(Update, []) if form is Form.DISCRETE else [],
        }
        reductions = reductions or ReductionPass()
        reduced = bool(reductions.reductions)
        fused = _reductions_compile(reductions)
        compiled = {kind: fused and all(_is_compilable(kernel) for kernel in items) for kind, items in used.items()}
        source = _Source()
        if reduced and fused:
            _reduce_function(source, reductions, self.parallel and form not in EVENT_FORMS)
        plan: dict[type, tuple[str, str]] = {
            Rate: ('drift', 'add'),
            Noise: ('diffusion', 'concatenate'),
            Jump: ('propensities', 'concatenate'),
            Update: ('update', 'assign'),
            Boundary: ('boundary', 'assign'),
        }
        needed = list(FORM_KINDS[form])
        if kernels.get(Boundary):
            needed = [*needed, Boundary]
        for kind in needed:
            if compiled[kind]:
                self._kind(source, used[kind], *plan[kind], form=form, reduced=reduced)
        accelerated_loop = {
            Form.ODE: 'rk4',
            Form.SDE: 'euler_maruyama',
            Form.JUMP: 'gillespie',
            Form.DISCRETE: 'iterate' if draws is None else 'iterate_random',
        }.get(form, '')
        has_loop = bool(accelerated_loop) and all(compiled[kind] for kind in needed)
        bounded = 'HAS_BOUNDARY = True\n' if Boundary in needed else NO_BOUNDARY
        loops = (
            [bounded, LOOPS[accelerated_loop], LOOPS['velocity_verlet'] if form is Form.ODE else ''] if has_loop else []
        )
        text = '\n'.join([HEADER, *source.lines, *loops]) + '\n'
        cache_dir = Path(self.cache_dir) if self.cache_dir is not None else DEFAULT_CACHE_DIR
        module = _load(text, source.functions, cache_dir)
        entries = {
            reference.entry.name: reference.entry
            for items in used.values()
            for kernel in items
            for reference in kernel.references.values()
        }
        for item in reductions.reductions:
            entries[item.entry.name] = item.entry
            entries.update({read.entry.name: read.entry for read in item.reads.values() if read.entry is not None})
            if item.members is not None:
                entries[item.members.name] = item.members
        layout = source.layout(entries)
        loop = accelerated_loop if has_loop else None
        return self._lowering(
            form,
            needed,
            used=used,
            compiled=compiled,
            changes=changes,
            size=size,
            module=module,
            layout=layout,
            loop=loop,
            reductions=reductions,
            draws=draws,
        )

    def _lowering(
        self,
        form: Form,
        needed: list[type],
        *,
        used: Mapping[type, list[Kernel]],
        compiled: Mapping[type, bool],
        changes: Mapping[int, list[tuple[Reference, float]]],
        size: int,
        module: ModuleType,
        layout: np.ndarray,
        loop: str | None,
        reductions: ReductionPass,
        draws: Draws | None,
    ) -> Lowering:
        """Return the lowering that binds the compiled functions to the layout, with NumPy fallbacks."""

        def bind(name: str) -> Callable[[float, Vector, Vector], Vector]:
            function = getattr(module, name)
            return lambda t, x, p: function(t, x, p, layout)

        numpy = {
            Rate: lambda: adding_function(used[Rate], size),
            Noise: lambda: concatenating_function(used[Noise]),
            Jump: lambda: concatenating_function(used[Jump]),
            Update: lambda: assigning_function(used[Update]),
            Boundary: lambda: assigning_function(used[Boundary]),
        }
        names = {Rate: 'drift', Noise: 'diffusion', Jump: 'propensities', Update: 'update', Boundary: 'boundary'}
        functions: dict[type, Any] = {
            kind: bind(names[kind]) if compiled[kind] else reductions.wrap(numpy[kind]()) for kind in needed
        }
        lowering = Lowering(boundary=functions.get(Boundary))
        if form in (Form.ODE, Form.FRACTIONAL, Form.SDE):
            lowering.drift = functions[Rate]
        if form is Form.SDE:
            lowering.diffusion = functions.get(Noise, concatenating_function([]))
            lowering.noise_rows = noise_rows(used[Noise])
        if form is Form.JUMP:
            lowering.propensities = functions.get(Jump, concatenating_function([]))
            lowering.jumps = jump_channels(used[Jump], changes)
        if form is Form.DISCRETE:
            lowering.update = functions[Update]
        if loop is not None:
            lowering.accelerated = self._accelerated(loop, module, layout, lowering, draws)
        return lowering

    def _accelerated(
        self,
        loop: str,
        module: ModuleType,
        layout: np.ndarray,
        lowering: Lowering,
        draws: Draws | None,
    ) -> Accelerated:
        """Return the compiled stepper loop of a form bound to the layout."""
        function = getattr(module, loop)
        if loop == 'rk4':
            verlet = module.velocity_verlet
            return Accelerated(
                rk4=lambda t, x, p, until, h: function(t, x, p, layout, until, h),
                velocity_verlet=lambda t, x, p, until, h, positions, velocities: verlet(
                    t,
                    x,
                    p,
                    layout,
                    until,
                    h,
                    positions,
                    velocities,
                ),
            )
        if loop == 'euler_maruyama':
            rows = lowering.noise_rows.astype(np.int64)
            return Accelerated(
                euler_maruyama=lambda t, x, p, until, h, rng: function(t, x, p, layout, until, h, rng, rows),
            )
        if loop == 'gillespie':
            targets, increments = _channel_arrays(lowering)
            return Accelerated(
                gillespie=lambda t, x, p, until, rng: function(t, x, p, layout, until, rng, targets, increments),
            )
        if loop == 'iterate_random' and draws is not None:
            offset, flags = draws.offset, draws.normal
            return Accelerated(iterate=lambda t, x, p, until, rng: function(t, x, p, layout, until, rng, offset, flags))
        return Accelerated(iterate=lambda t, x, p, until: function(t, x, p, layout, until))


def _channel_arrays(lowering: Lowering) -> tuple[np.ndarray, np.ndarray]:
    """Return the state targets and increments of every jump channel, padded with -1 targets."""
    n_channels = sum(channels.size for channels in lowering.jumps)
    width = max((len(channels.increments) for channels in lowering.jumps), default=0)
    targets = np.full((n_channels, width), -1, dtype=np.int64)
    increments = np.zeros((n_channels, width))
    for channels in lowering.jumps:
        rows = slice(channels.offset, channels.offset + channels.size)
        count = len(channels.increments)
        targets[rows, :count] = channels.targets.T
        increments[rows, :count] = channels.increments
    return targets, increments


register_backend(NumbaBackend)
