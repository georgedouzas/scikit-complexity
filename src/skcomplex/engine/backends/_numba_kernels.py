"""Generate the Numba source of kernels evaluated once per member, per pair within a cutoff, or per edge."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib

from numba.core.dispatcher import Dispatcher

from ...core.base import Expression
from ..compile._kernels import EdgeSearch, Kernel, Output, PairSearch, Reference, Source
from ._numba_source import INDENT, _emit_values, _hoist_layout, _index, _is_shared, _single_reads, _Source, _vector


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
    assert isinstance(pairs, PairSearch)  # nosec B101: pair kernels hold a pair search
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
    assert isinstance(pairs, PairSearch)  # nosec B101: pair kernels hold a pair search
    n_dim = pairs.first.columns
    lines = [f'{INDENT}for _i_ in prange(rows_{loop}):', f'{INDENT * 2}{loop} = order_{own}[_i_]']
    lines += [f'{INDENT * 2}_acc{index}_ = 0.0' for index, _ in outputs]
    for dimension in range(n_dim):
        value = f'positions_{own}[{loop}, {dimension}]'
        lines.append(f'{INDENT * 2}_c{dimension}_ = cell_coordinate({value}, {dimension}, shape_{cells}, _box_)')
    depth = 2
    for dimension in range(n_dim):
        pad = INDENT * depth
        counts = f'shape_{cells}[0][{dimension}]'
        lines.append(f'{pad}for _o{dimension}_ in range(-1, 2):')
        if pairs.period is not None:
            lines += [
                f'{pad}{INDENT}if (_o{dimension}_ == -1 and {counts} < 3) or (_o{dimension}_ == 1 and {counts} < 2):',
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
    """Add the source of a pair kernel that sums pair outputs over neighbours found in the bounded cell grids."""
    pairs = kernel.pairs
    assert isinstance(pairs, PairSearch)  # nosec B101: pair kernels hold a pair search
    same = pairs.first == pairs.second
    columns = pairs.first.columns
    cutoff = Reference(pairs.cutoff, is_state=False, column=0)
    lines = [
        f'@njit(cache=CACHE, parallel={parallel})',
        f'def {name}(t, x, p, L, out, offset):',
        f'{INDENT}_cutoff_ = p[{_index(source, cutoff, None)}]',
        f'{INDENT}_box_ = (np.full({columns}, PERIOD), PERIODIC)',
    ]
    for label, entry in (('a', pairs.first), ('b', pairs.second)) if not same else (('a', pairs.first),):
        start, rows = f'_L{2 * source.slot(entry)}_', f'_L{2 * source.slot(entry) + 1}_'
        cells = ', '.join(f'{part}_{label}' for part in ('shape', 'strides', 'starts', 'order'))
        positions = f'x[{start} : {start} + rows__{label}_ * {columns}].reshape((rows__{label}_, {columns}))'
        lines += [
            f'{INDENT}rows__{label}_ = {rows}',
            f'{INDENT}positions_{label} = {positions}',
            f'{INDENT}{cells} = build_grid(positions_{label}, _cutoff_, _box_)',
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


def _edge_rows(source: _Source, search: EdgeSearch) -> tuple[str, list[str]]:
    """Return the code of the number of edges and the lines that read the ends `_a_` and `_b_` of the edge `_m_`."""
    pad = INDENT * 2
    if search.source is not None and search.target is not None:
        first, second = source.slot(search.source), source.slot(search.target)
        return f'_L{2 * first + 1}_', [
            f'{pad}_a_ = int(p[_L{2 * first}_ + _m_])',
            f'{pad}_b_ = int(p[_L{2 * second}_ + _m_])',
        ]
    digest = hashlib.sha256(search.sources.tobytes() + search.targets.tobytes()).hexdigest()[:16]
    first = source.data_slot(f'__edge_sources_{digest}', search.sources)
    second = source.data_slot(f'__edge_targets_{digest}', search.targets)
    return f'_L{2 * first + 1}_', [f'{pad}_a_ = L[_L{2 * first}_ + _m_]', f'{pad}_b_ = L[_L{2 * second}_ + _m_]']


def _edge_kernel(source: _Source, kernel: Kernel, name: str, mode: str) -> None:
    """Add the source of a kernel evaluated once per edge, whose laws read the ends and the edge member."""
    search = kernel.pairs
    assert isinstance(search, EdgeSearch)
    rows, ends = _edge_rows(source, search)
    lines = ['@njit(cache=CACHE)', f'def {name}(t, x, p, L, out, offset):', f'{INDENT}rows = {rows}']
    lines += _single_reads(source, kernel, kernel.outputs)
    members = {Source.MEMBER: '_m_', Source.FIRST: '_a_', Source.SECOND: '_b_'}

    def read(item: Expression) -> str:
        reference = kernel.references[item]
        if _is_shared(reference):
            return str(item)
        return f'{_vector(reference)}[{_index(source, reference, members[reference.source])}]'

    sums = [
        index
        for index, output in enumerate(kernel.outputs)
        if output.target is not None and output.target.source is Source.SINGLE and mode == 'add'
    ]
    lines += [f'{INDENT}_sum{index}_ = 0.0' for index in sums]
    lines += [f'{INDENT}for _m_ in range(rows):', *ends]
    body, values = _emit_values(source, kernel.outputs, read, 2)
    lines += body
    for index, (output, value) in enumerate(zip(kernel.outputs, values, strict=True)):
        if mode == 'concatenate':
            lines.append(f'{INDENT * 2}out[offset + {index} * rows + _m_] = {value}')
        elif index in sums:
            lines.append(f'{INDENT * 2}_sum{index}_ += {value}')
        else:
            target = output.target
            assert target is not None
            operator = '+=' if mode == 'add' else '='
            lines.append(f'{INDENT * 2}out[{_index(source, target, members[target.source])}] {operator} {value}')
    for index in sums:
        lines.append(f'{INDENT}out[{_index(source, kernel.outputs[index].target, None)}] += _sum{index}_')  # type: ignore[arg-type]
    lines.append(f'{INDENT}return offset + {len(kernel.outputs)} * rows')
    source.lines += ['', '', *_hoist_layout(lines)]


def _output_count(source: _Source, kernel: Kernel) -> str:
    """Return the code of the number of values a member kernel writes, its outputs times its rows."""
    if isinstance(kernel.pairs, EdgeSearch):
        rows, _ = _edge_rows(source, kernel.pairs)
        return f'{len(kernel.outputs)} * L[{rows[2:-1]}]'
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
