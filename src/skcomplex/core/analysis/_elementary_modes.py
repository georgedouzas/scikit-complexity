"""Enumerate the elementary modes and extreme pathways of a steady-state cone.

The cone is the steady-state cone of a linear program.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd

import numpy as np
import numpy.typing as npt
from cysignals.alarm import AlarmInterrupt, alarm, cancel_alarm
from sage.all__sagemath_polyhedra import Polyhedron
from sage.rings.rational_field import QQ

from ..components.systems import System
from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems import ProblemSpec
from ..engine.problems._linear_program_kernel import LinearProgramKernel
from ..errors import AnalysisError, EnumerationLimitError
from ..specs import AnalysisTags, RunStateSpec
from ..specs._base_spec import BaseSpec
from ..types import Array, DataKind
from ._base import _composite
from .types import Source

KINDS = ('modes', 'pathways')
MAXIMUM_DENOMINATOR = 10**6
ROUNDING_LIMIT = 1e-9

TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class _ConeSpec(BaseSpec):
    """Steady-state cone.

    It holds a matrix over the variables kept, their directions, and which of them are exchanges.
    """

    matrix: Array
    labels: tuple[str, ...]
    irreversible: frozenset[int]
    reversed: frozenset[int]
    exchanges: frozenset[int]
    rows: tuple[str, ...]


@dataclass(frozen=True)
class ElementaryModesResult:
    """Generators of the steady-state cone of a linear program.

    They are the elementary modes or the extreme pathways.

    Args:
        labels:
            The name of every column of `modes`.

        modes:
            One row per generator, scaled to `normalize` or to the smallest integer vector, as fractions.

        lines:
            The generators that run in both directions, the lines of a cone that is not pointed, one row per line.
    """

    labels: tuple[str, ...]
    modes: npt.NDArray[np.object_]
    lines: npt.NDArray[np.object_]


def elementary_modes(
    source: Source,
    *,
    kind: str = 'modes',
    normalize: str | None = None,
    exchanges: Sequence[str] | None = None,
    timeout: float = 60.0,
    labels: Sequence[str] | None = None,
    irreversible: Sequence[int] | None = None,
    sort: str | None = None,
) -> ElementaryModesResult:
    """Return the exact generators of the steady-state cone of a linear program.

    The cone is `{v : S v = 0, v_irr >= 0}`. A system gives the program of its linear program law at its initial values,
    whose bounds give the directions: a negative lower bound makes a variable reversible, a nonpositive upper bound
    reverses its column, and a variable fixed at zero is removed. A matrix gives the cone directly, with the indices of
    its irreversible columns.

    - `'modes'`: the steady-state vectors of minimal support in which every irreversible variable runs in its declared
      direction. Every reversible column splits in two, the extreme rays of the split cone come from the Parma
      Polyhedra Library, the two-cycles of split pairs drop, and the rest maps back (Gagneur and Klamt 2004).
    - `'pathways'`: the generators of the cone with every reversible internal column split and each exchange kept as one
      column, of either sign when it is reversible (Schilling et al. 2000), and without any generator that is a
      non-negative combination of the others (Raman 2021, Section 9.8). The generators that run in both directions leave
      `modes` for `lines`.

    Args:
        source:
            The system, or the matrix.

        kind:
            The generators to enumerate, `'modes'` or `'pathways'`.

        normalize:
            The name of a variable to scale every row to one where it is not zero, or `None` for the smallest integer
            vector.

        exchanges:
            The names of the exchange variables, or `None` for the variables in one row.

        timeout:
            The time limit of the enumeration in seconds.

        labels:
            The names of the columns of a matrix, or `None` for `v0`, `v1`, and so on.

        irreversible:
            The indices of the irreversible columns of a matrix, or `None` for every column.

        sort:
            The name of a variable to order the rows by, largest first, such as the growth flux of modes normalised to
            the uptake, which orders them by yield, or `None` for a fixed order.

    Returns:
        result:
            The names of the columns, the generators, and the lines.

    Raises:
        AnalysisError:
            The kind is unknown, the right side of a balance is not zero, a coefficient is not close to a fraction, or
            no system or matrix holds a program.

        EnumerationLimitError:
            The enumeration does not finish within the time limit.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import elementary_modes
        >>> matrix = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        >>> elementary_modes(matrix, normalize='v0').modes.tolist()
        [[Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)]]
    """
    source = _composite(source)
    if kind not in KINDS:
        error_msg = f'Unknown kind {kind!r}. Use one of {", ".join(KINDS)}.'
        raise AnalysisError(error_msg)
    if not timeout > 0:
        error_msg = f'The time limit must be positive, but it is {timeout!r}.'
        raise AnalysisError(error_msg)
    if isinstance(source, System):
        cone = _system_cone(source, exchanges)
    elif isinstance(source, np.ndarray):
        cone = _matrix_cone(np.asarray(source, dtype=float), exchanges, labels, irreversible)
    else:
        error_msg = f'elementary_modes takes a system or a matrix, not {type(source).__name__}.'
        raise AnalysisError(error_msg)
    fractions = _exact_matrix(cone)
    vectors = _mapped_rays(cone, fractions, _split_columns(cone, kind), kind, timeout)
    rays = _unique(cone, vectors, normalize, sort)
    if kind == 'pathways':
        rays = _independent(cone, rays, timeout)
    kept = {tuple(row) for row in rays}
    opposite = [tuple(-item for item in row) in kept for row in rays]
    size = len(cone.labels)
    lines = np.asarray(
        [row for row, both in zip(rays, opposite, strict=True) if both and next(i for i in row if i != 0) > 0],
        dtype=object,
    ).reshape(-1, size)
    modes = rays if kind == 'modes' else rays[~np.asarray(opposite, dtype=bool)]
    return ElementaryModesResult(cone.labels, modes, lines)


def _kernel_of(problem: ProblemSpec) -> LinearProgramKernel:
    """Return the prepared kernel of the only program of a problem."""
    if len(problem.programs) != 1:
        error_msg = f'The analysis reads one linear program, but the system holds {len(problem.programs)}.'
        raise AnalysisError(error_msg)
    state = RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)
    return problem.programs[0].prepare(state)


def _system_cone(system: System, exchanges: Sequence[str] | None) -> _ConeSpec:
    """Return the cone of the program of a system at its initial values."""
    problem = PassCompiler().prepare(system).run().problem_
    if not isinstance(problem, ProblemSpec):
        error_msg = f'The analysis reads a linear program, but the problem is a {type(problem).__name__}.'
        raise AnalysisError(error_msg)
    kernel = _kernel_of(problem)
    program = kernel.program
    _check_steady(program.constraints, kernel.row_lower_[0], kernel.row_upper_[0])
    lower, upper = kernel.lower_[0], kernel.upper_[0]
    kept = [column for column in range(len(program.names)) if lower[column] != 0 or upper[column] != 0]
    matrix = np.asarray(kernel.matrix_.toarray(), dtype=float)
    flipped = {column for column in kept if upper[column] <= 0 < -lower[column]}
    irreversible = {column for column in kept if lower[column] >= 0 or column in flipped}
    positions = {column: position for position, column in enumerate(kept)}
    cone = matrix[:, kept]
    return _ConeSpec(
        cone,
        tuple(program.names[column] for column in kept),
        frozenset(positions[column] for column in irreversible),
        frozenset(positions[column] for column in flipped),
        _exchanges_of(cone, tuple(program.names[column] for column in kept), exchanges),
        tuple(program.constraints),
    )


def _check_steady(rows: Sequence[str], lower: Array, upper: Array) -> None:
    """Raise an error for a row that is no balance at zero."""
    for name, below, above in zip(rows, lower, upper, strict=True):
        if below != 0 or above != 0:
            error_msg = (
                f'The balance {name} has the right side [{below}, {above}], but modes are defined at steady state, '
                'where every balance is zero.'
            )
            raise AnalysisError(error_msg)


def _exchanges_of(matrix: Array, labels: Sequence[str], exchanges: Sequence[str] | None) -> frozenset[int]:
    """Return the columns of the exchanges.

    They are those named, or the variables in one row.
    """
    if exchanges is not None:
        named = set(exchanges)
        return frozenset(position for position, name in enumerate(labels) if name in named)
    return frozenset(np.flatnonzero(np.count_nonzero(matrix, axis=0) == 1).tolist())


def _matrix_cone(
    matrix: Array,
    exchanges: Sequence[str] | None,
    labels: Sequence[str] | None,
    irreversible: Sequence[int] | None,
) -> _ConeSpec:
    """Return the cone of a matrix with its irreversible columns."""
    columns = matrix.shape[1]
    names = tuple(labels) if labels is not None else tuple(f'v{column}' for column in range(columns))
    chosen = range(columns) if irreversible is None else irreversible
    return _ConeSpec(
        matrix,
        names,
        frozenset(int(item) for item in chosen),
        frozenset(),
        _exchanges_of(matrix, names, exchanges),
        tuple(f'row {row}' for row in range(matrix.shape[0])),
    )


def _exact_matrix(cone: _ConeSpec) -> list[list[Fraction]]:
    """Return the matrix as fractions of small denominators.

    The reversed columns are negated.
    """
    rows = []
    for row, values in enumerate(cone.matrix):
        items = []
        for column, value in enumerate(values):
            fraction = Fraction(float(value)).limit_denominator(MAXIMUM_DENOMINATOR)
            if abs(float(fraction) - float(value)) > ROUNDING_LIMIT:
                error_msg = (
                    f'The coefficient {value} of {cone.labels[column]} in {cone.rows[row]} moves by more than '
                    f'{ROUNDING_LIMIT} as a fraction. Round it, or scale the constraint.'
                )
                raise AnalysisError(error_msg)
            items.append(-fraction if column in cone.reversed else fraction)
        rows.append(items)
    return rows


def _is_free(cone: _ConeSpec, column: int, kind: str) -> bool:
    """Check whether a column stays one column of either sign.

    Such a column is a reversible exchange of the pathways.
    """
    return kind == 'pathways' and column in cone.exchanges and column not in cone.irreversible


def _split_columns(cone: _ConeSpec, kind: str) -> list[tuple[int, int]]:
    """Return every column of the split cone.

    Each comes with the column it comes from and its sign.
    """
    split: list[tuple[int, int]] = []
    for column in range(len(cone.labels)):
        split.append((column, 1))
        keeps = column in cone.irreversible or (kind == 'pathways' and column in cone.exchanges)
        if not keeps:
            split.append((column, -1))
    return split


def _rays_of(
    cone: _ConeSpec,
    fractions: list[list[Fraction]],
    columns: Sequence[tuple[int, int]],
    kind: str,
    timeout: float,
) -> list[list[Fraction]]:
    """Return the extreme rays of the split cone, under the time limit."""
    equations = [[QQ(0), *(QQ(row[column]) * sign for column, sign in columns)] for row in fractions if any(row)]
    size = len(columns)
    inequalities = [
        [QQ(0), *(QQ(int(position == other)) for other in range(size))]
        for position, (column, _) in enumerate(columns)
        if not _is_free(cone, column, kind)
    ]
    try:
        alarm(timeout)
        polyhedron = Polyhedron(eqns=equations, ieqs=inequalities, backend='ppl')
        lines = [[-item for item in line] for line in polyhedron.lines_list()]
        generators = [*polyhedron.rays_list(), *polyhedron.lines_list(), *lines]
        rays = [[Fraction(int(item.numerator()), int(item.denominator())) for item in ray] for ray in generators]
        cancel_alarm()
    except AlarmInterrupt as error:
        raise _limit_error(cone, timeout) from error
    return rays


def _limit_error(cone: _ConeSpec, timeout: float) -> EnumerationLimitError:
    """Return the error of an enumeration that did not finish within the time limit."""
    error_msg = (
        f'The enumeration over {len(cone.labels)} variables did not finish within {timeout} s. Close exchanges, or '
        'enumerate a subnetwork.'
    )
    return EnumerationLimitError(error_msg)


def _independent(cone: _ConeSpec, rows: np.ndarray, timeout: float) -> np.ndarray:
    """Return the rows that no non-negative combination of the other rows gives.

    The search runs under the time limit.
    """
    kept = [tuple(row) for row in rows]
    try:
        alarm(timeout)
        for row in list(kept):
            others = [[QQ(item) for item in other] for other in kept if other != row]
            if others and Polyhedron(rays=others, backend='ppl').contains([QQ(item) for item in row]):
                kept.remove(row)
        cancel_alarm()
    except AlarmInterrupt as error:
        raise _limit_error(cone, timeout) from error
    return np.asarray(kept, dtype=object).reshape(-1, rows.shape[1])


def _mapped_rays(
    cone: _ConeSpec,
    fractions: list[list[Fraction]],
    columns: Sequence[tuple[int, int]],
    kind: str,
    timeout: float,
) -> list[list[Fraction]]:
    """Return the generators mapped back to the columns of the cone.

    The two-cycles of split pairs are dropped.
    """
    size = len(cone.labels)
    found = []
    for ray in _rays_of(cone, fractions, columns, kind, timeout):
        vector = [Fraction(0)] * size
        for value, (column, sign) in zip(ray, columns, strict=True):
            vector[column] += sign * value
        if any(vector):
            found.append(vector)
    return found


def _scaled(cone: _ConeSpec, vector: Sequence[Fraction], normalize: str | None) -> list[Fraction]:
    """Return a generator with `normalize` at one, or as the smallest integer vector.

    The generator is in declared directions.
    """
    directed = [-item if column in cone.reversed else item for column, item in enumerate(vector)]
    if normalize is not None and normalize in cone.labels:
        pivot = directed[cone.labels.index(normalize)]
        if pivot != 0:
            return [item / abs(pivot) for item in directed]
    denominators = np.lcm.reduce([item.denominator for item in directed])
    whole = [int(item * int(denominators)) for item in directed]
    divisor = 0
    for item in whole:
        divisor = gcd(divisor, item)
    return [Fraction(item, divisor or 1) for item in whole]


def _unique(
    cone: _ConeSpec,
    vectors: Sequence[Sequence[Fraction]],
    normalize: str | None,
    sort: str | None,
) -> np.ndarray:
    """Return the distinct generators as a matrix of fractions, in a fixed order."""
    size = len(cone.labels)
    kept = sorted({tuple(_scaled(cone, vector, normalize)) for vector in vectors}, reverse=True)
    if sort is not None and sort in cone.labels:
        position = cone.labels.index(sort)
        kept.sort(key=lambda row: row[position], reverse=True)
    return np.asarray(kept, dtype=object).reshape(-1, size)
