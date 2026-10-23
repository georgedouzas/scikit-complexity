"""Generate the Numba source of the function that extends the parameter vector with every reduction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import scipy.sparse

from ..base import Expression, ValidationError
from ..compile._neighbourhoods import CompiledNeighbourhood, EdgeRows, MetricSearch, StaticSearch
from ..compile._printing import SCALAR_DIALECT, print_expressions
from ..compile._reductions import CompiledReduction, Read, ReductionPass
from ..space import Cone, Nearest
from ._numba_source import INDENT, _hoist_layout, _Source


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


def _search_lines(source: _Source, reduction: CompiledNeighbourhood, index: int) -> list[str]:
    """Return the lines that find the neighbours of every member, as rows `_ip<k>_` and neighbours `_ix<k>_`."""
    search = reduction.search
    if isinstance(search, StaticSearch):
        rows = source.data_slot(f'__neighbours_{index}_indptr', search.indptr)
        neighbours = source.data_slot(f'__neighbours_{index}_indices', search.indices)
        lines = [
            f'{INDENT}_ip{index}_ = L[_L{2 * rows}_ : _L{2 * rows}_ + _L{2 * rows + 1}_]',
            f'{INDENT}_ix{index}_ = L[_L{2 * neighbours}_ : _L{2 * neighbours}_ + _L{2 * neighbours + 1}_]',
        ]
        if search.edges is not None:
            edges = source.data_slot(f'__neighbours_{index}_edges', search.edges)
            lines.append(f'{INDENT}_ie{index}_ = L[_L{2 * edges}_ : _L{2 * edges}_ + _L{2 * edges + 1}_]')
        return lines
    if isinstance(search, EdgeRows):
        starts = ', '.join(f'_L{2 * source.slot(entry)}_' for entry in (search.source, search.target, search.active))
        call = (
            f'edge_neighbours(p, ({starts}), _L{2 * source.slot(search.source) + 1}_, {search.size}, '
            f'{search.directed}, {search.center})'
        )
        return [f'{INDENT}_ip{index}_, _ix{index}_, _ie{index}_ = {call}']
    assert isinstance(search, MetricSearch)
    entry = search.positions
    start, columns = f'_L{2 * source.slot(entry)}_', entry.columns
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
    if isinstance(neighbourhood, Nearest):
        call = f'nearest_neighbours(_pos{index}_, {neighbourhood.k}, _per{index}_, {periodic}, {center})'
        return [*lines, f'{INDENT}_ip{index}_, _ix{index}_ = {call}']
    if not isinstance(neighbourhood, Cone):
        call = f'radius_neighbours(_pos{index}_, {neighbourhood.radius!r}, _per{index}_, {periodic}, {center})'
        return [*lines, f'{INDENT}_ip{index}_, _ix{index}_ = {call}']
    heading = search.heading
    assert heading is not None
    first = f'_L{2 * source.slot(heading.entry)}_'
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
        f'cone_neighbours(_pos{index}_, _hd{index}_, {neighbourhood.radius!r}, {cosine}, (_per{index}_, '
        f'{periodic}), {center})'
    )
    return [*lines, f'{INDENT}_ip{index}_, _ix{index}_ = {call}']


def _pair_read_code(source: _Source, reduction: CompiledNeighbourhood, item: Expression, index: int) -> str:
    """Return the code of one symbol of a neighbourhood reduction for the reader `_i_` and the neighbour `_j_`."""
    pair = reduction.reads[item]
    read = pair.read
    vector = 'x' if read.is_state else 'p'
    assert read.entry is not None
    start = f'_L{2 * source.slot(read.entry)}_'
    if pair.side == 'shared':
        return f'{vector}[{start} + {read.column}]'
    member = {'reader': '_i_', 'edge': f'_ie{index}_[_e_]'}.get(pair.side, '_j_')
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


def _neighbourhood_lines(source: _Source, reduction: CompiledNeighbourhood, index: int) -> list[str]:
    """Return the lines that compute a neighbourhood reduction of every member into the extended parameter vector."""
    lines = _search_lines(source, reduction, index)
    sides = {item: reduction.reads[item].side for item in reduction.symbols}
    shared = [item for item in reduction.symbols if sides[item] == 'shared']
    readers = [item for item in reduction.symbols if sides[item] == 'reader']
    others = [item for item in reduction.symbols if sides[item] in {'other', 'edge'}]
    lines += [f'{INDENT}{item} = {_pair_read_code(source, reduction, item, index)}' for item in shared]
    assignments, codes = print_expressions(
        [reduction.argument, reduction.weight],
        SCALAR_DIALECT,
        prefix=f'_n{index}cse',
    )
    acc, weights, found = f'_acc{index}_', f'_ws{index}_', f'_found{index}_'
    kind = reduction.kind
    if kind in {'max', 'min'}:
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
    target = f'_L{2 * source.slot(reduction.entry)}_'
    lines += [
        f'{INDENT}for _i_ in prange({reduction.size}):',
        *(f'{INDENT * 2}{item} = {_pair_read_code(source, reduction, item, index)}' for item in readers),
        f'{INDENT * 2}{acc} = 0.0',
        f'{INDENT * 2}{weights} = 0.0',
        f'{INDENT * 2}{found} = False',
        f'{INDENT * 2}for _e_ in range(_ip{index}_[_i_], _ip{index}_[_i_ + 1]):',
        f'{INDENT * 3}_j_ = _ix{index}_[_e_]',
        *(f'{INDENT * 3}{item} = {_pair_read_code(source, reduction, item, index)}' for item in others),
        *(f'{INDENT * 3}{name} = {code}' for name, code in assignments),
        f'{INDENT * 3}_a_ = {codes[0]}',
        f'{INDENT * 3}_w_ = {codes[1]}',
        *(f'{INDENT * 3}{line}' for line in accumulate),
        f'{INDENT * 2}p[{target} + _i_] = {value}',
    ]
    return lines


def _reduce_function(source: _Source, reductions: ReductionPass, parallel: bool) -> None:
    """Add the function that returns the parameter vector extended with every reduction, in dependency order."""
    extents = [
        f'_L{2 * source.slot(item.entry)}_ + _L{2 * source.slot(item.entry) + 1}_ * {item.entry.columns}'
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
        if isinstance(reduction, CompiledNeighbourhood):
            lines += _neighbourhood_lines(source, reduction, index)
        else:
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
        for neighbourhood in reductions.neighbourhoods:
            print_expressions([neighbourhood.argument, neighbourhood.weight], SCALAR_DIALECT)
    except ValidationError:
        return False
    return True
