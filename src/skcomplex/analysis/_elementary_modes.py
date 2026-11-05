"""Enumerate the elementary modes and extreme pathways of the steady-state cone of a linear program."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd

import numpy as np
from cysignals.alarm import AlarmInterrupt, alarm, cancel_alarm
from sage.all__sagemath_polyhedra import Polyhedron
from sage.rings.rational_field import QQ

from ..core.system import System
from ..engine.compile import compile_system
from ..engine.problems import LinearProgramKernel, ProblemSpec
from ..errors import AnalysisError, EnumerationLimitError
from ..specs import RunStateSpec
from ..specs._base_spec import BaseSpec
from ..types import Array, DataKind
from ._base import BaseAnalysis
from .types import Source

KINDS = ('modes', 'pathways')
MAXIMUM_DENOMINATOR = 10**6
ROUNDING_LIMIT = 1e-9


@dataclass(frozen=True)
class _ConeSpec(BaseSpec):
    """Steady-state cone: a matrix over the variables kept, their directions, and which of them are exchanges."""

    matrix: Array
    labels: tuple[str, ...]
    irreversible: frozenset[int]
    reversed: frozenset[int]
    exchanges: frozenset[int]
    rows: tuple[str, ...]


class ElementaryModeAnalysis(BaseAnalysis):
    """Exact enumeration of the generators of the steady-state cone `{v : S v = 0, v_irr >= 0}` of a linear program.

    A system gives the program of its linear program law at its initial values, whose bounds give the directions: a
    negative lower bound makes a variable reversible, a nonpositive upper bound reverses its column, and a variable
    fixed at zero is removed. A matrix gives the cone directly, with the indices of its irreversible columns.

    - `'modes'`: the steady-state vectors of minimal support in which every irreversible variable runs in its declared
      direction. Every reversible column splits in two, the extreme rays of the split cone come from the Parma
      Polyhedra Library, the two-cycles of split pairs drop, and the rest maps back (Gagneur and Klamt 2004).
    - `'pathways'`: the generators of the cone with every reversible internal column split and each exchange kept as one
      column, of either sign when it is reversible (Schilling et al. 2000), and without any generator that is a
      non-negative combination of the others (Raman 2021, Section 9.8). The generators that run in both directions leave
      `modes_` for `lines_`.

    Args:
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

    Attributes:
        modes_:
            One row per generator, scaled to `normalize` or to the smallest integer vector, as fractions.

        labels_:
            The name of every column of `modes_`.

        lines_:
            The generators that run in both directions, the lines of a cone that is not pointed, one row per line.

    Raises:
        AnalysisError:
            The kind is unknown, the right side of a balance is not zero, a coefficient is not close to a fraction, or
            no system or matrix holds a program, raised by `prepare`.

        EnumerationLimitError:
            The enumeration does not finish within the time limit, raised by `run`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import ElementaryModeAnalysis
        >>> matrix = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        >>> analysis = ElementaryModeAnalysis(normalize='v0').prepare(matrix).run()
        >>> analysis.modes_.tolist()
        [[Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)]]
    """

    def __init__(
        self,
        kind: str = 'modes',
        *,
        normalize: str | None = None,
        exchanges: Sequence[str] | None = None,
        timeout: float = 60.0,
        labels: Sequence[str] | None = None,
        irreversible: Sequence[int] | None = None,
        sort: str | None = None,
    ) -> None:
        self.kind = kind
        self.normalize = normalize
        self.exchanges = exchanges
        self.timeout = timeout
        self.labels = labels
        self.irreversible = irreversible
        self.sort = sort

    def _get_data_kind(self) -> DataKind:
        """Return the kind of data the analysis reads, the program of a system."""
        return DataKind.SERIES

    def _check(self) -> None:
        """Raise an error for an unknown kind or a time limit that is not positive."""
        if self.kind not in KINDS:
            error_msg = f'Unknown kind {self.kind!r}. Use one of {", ".join(KINDS)}.'
            raise AnalysisError(error_msg)
        if not self.timeout > 0:
            error_msg = f'The time limit must be positive, but it is {self.timeout!r}.'
            raise AnalysisError(error_msg)

    @staticmethod
    def _kernel_of(problem: ProblemSpec) -> LinearProgramKernel:
        """Return the prepared kernel of the only program of a problem."""
        if len(problem.programs) != 1:
            error_msg = f'The analysis reads one linear program, but the system holds {len(problem.programs)}.'
            raise AnalysisError(error_msg)
        state = RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)
        return problem.programs[0].prepare(state)

    def _system_cone(self, system: System) -> _ConeSpec:
        """Return the cone of the program of a system at its initial values."""
        problem = compile_system(system)
        if not isinstance(problem, ProblemSpec):
            error_msg = f'The analysis reads a linear program, but the problem is a {type(problem).__name__}.'
            raise AnalysisError(error_msg)
        kernel = self._kernel_of(problem)
        program = kernel.program
        self._check_steady(program.constraints, kernel.row_lower_[0], kernel.row_upper_[0])
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
            self._exchanges_of(cone, tuple(program.names[column] for column in kept)),
            tuple(program.constraints),
        )

    @staticmethod
    def _check_steady(rows: Sequence[str], lower: Array, upper: Array) -> None:
        """Raise an error for a row that is no balance at zero."""
        for name, below, above in zip(rows, lower, upper, strict=True):
            if below != 0 or above != 0:
                error_msg = (
                    f'The balance {name} has the right side [{below}, {above}], but modes are defined at steady state, '
                    'where every balance is zero.'
                )
                raise AnalysisError(error_msg)

    def _exchanges_of(self, matrix: Array, labels: Sequence[str]) -> frozenset[int]:
        """Return the columns of the exchanges: those named, or the variables in one row."""
        if self.exchanges is not None:
            named = set(self.exchanges)
            return frozenset(position for position, name in enumerate(labels) if name in named)
        return frozenset(np.flatnonzero(np.count_nonzero(matrix, axis=0) == 1).tolist())

    def _matrix_cone(self, matrix: Array) -> _ConeSpec:
        """Return the cone of a matrix with its irreversible columns."""
        columns = matrix.shape[1]
        labels = tuple(self.labels) if self.labels is not None else tuple(f'v{column}' for column in range(columns))
        chosen = range(columns) if self.irreversible is None else self.irreversible
        return _ConeSpec(
            matrix,
            labels,
            frozenset(int(item) for item in chosen),
            frozenset(),
            self._exchanges_of(matrix, labels),
            tuple(f'row {row}' for row in range(matrix.shape[0])),
        )

    def _prepare(self, context: Source) -> None:
        """Read the cone of a system or a matrix and turn its coefficients into fractions.

        Args:
            context:
                The system, or the matrix.

        Raises:
            AnalysisError:
                No system or matrix is passed, a right side is not zero, or a coefficient is not close to a fraction.
        """
        if isinstance(context, System):
            self._cone = self._system_cone(context)
        elif isinstance(context, np.ndarray):
            self._cone = self._matrix_cone(np.asarray(context, dtype=float))
        else:
            error_msg = f'The analysis takes a system or a matrix, not {type(context).__name__}.'
            raise AnalysisError(error_msg)
        self.labels_ = self._cone.labels
        self._fractions = self._exact_matrix(self._cone)

    @staticmethod
    def _exact_matrix(cone: _ConeSpec) -> list[list[Fraction]]:
        """Return the matrix as fractions of small denominators, with reversed columns negated."""
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

    def _is_free(self, column: int) -> bool:
        """Check whether a column stays one column of either sign, a reversible exchange of the pathways."""
        cone = self._cone
        return self.kind == 'pathways' and column in cone.exchanges and column not in cone.irreversible

    def _split_columns(self) -> list[tuple[int, int]]:
        """Return every column of the split cone: the column it comes from and its sign."""
        cone = self._cone
        split: list[tuple[int, int]] = []
        for column in range(len(cone.labels)):
            split.append((column, 1))
            keeps = column in cone.irreversible or (self.kind == 'pathways' and column in cone.exchanges)
            if not keeps:
                split.append((column, -1))
        return split

    def _rays_of(self, columns: Sequence[tuple[int, int]]) -> list[list[Fraction]]:
        """Return the extreme rays of the split cone, under the time limit."""
        equations = [
            [QQ(0), *(QQ(row[column]) * sign for column, sign in columns)] for row in self._fractions if any(row)
        ]
        size = len(columns)
        inequalities = [
            [QQ(0), *(QQ(int(position == other)) for other in range(size))]
            for position, (column, _) in enumerate(columns)
            if not self._is_free(column)
        ]
        try:
            alarm(self.timeout)
            cone = Polyhedron(eqns=equations, ieqs=inequalities, backend='ppl')
            lines = [[-item for item in line] for line in cone.lines_list()]
            generators = [*cone.rays_list(), *cone.lines_list(), *lines]
            rays = [[Fraction(int(item.numerator()), int(item.denominator())) for item in ray] for ray in generators]
            cancel_alarm()
        except AlarmInterrupt as error:
            raise self._limit_error() from error
        return rays

    def _limit_error(self) -> EnumerationLimitError:
        """Return the error of an enumeration that did not finish within the time limit."""
        error_msg = (
            f'The enumeration over {len(self._cone.labels)} variables did not finish within {self.timeout} s. '
            'Close exchanges, or enumerate a subnetwork.'
        )
        return EnumerationLimitError(error_msg)

    def _independent(self, rows: np.ndarray) -> np.ndarray:
        """Return the rows that no non-negative combination of the other rows gives, under the time limit."""
        kept = [tuple(row) for row in rows]
        try:
            alarm(self.timeout)
            for row in list(kept):
                others = [[QQ(item) for item in other] for other in kept if other != row]
                if others and Polyhedron(rays=others, backend='ppl').contains([QQ(item) for item in row]):
                    kept.remove(row)
            cancel_alarm()
        except AlarmInterrupt as error:
            raise self._limit_error() from error
        return np.asarray(kept, dtype=object).reshape(-1, rows.shape[1])

    def _mapped_rays(self, columns: Sequence[tuple[int, int]]) -> list[list[Fraction]]:
        """Return the generators mapped back to the columns of the cone, without the two-cycles of split pairs."""
        size = len(self._cone.labels)
        found = []
        for ray in self._rays_of(columns):
            vector = [Fraction(0)] * size
            for value, (column, sign) in zip(ray, columns, strict=True):
                vector[column] += sign * value
            if any(vector):
                found.append(vector)
        return found

    def _scaled(self, vector: Sequence[Fraction]) -> list[Fraction]:
        """Return a generator with `normalize` at one, or as the smallest integer vector, in declared directions."""
        cone = self._cone
        directed = [-item if column in cone.reversed else item for column, item in enumerate(vector)]
        if self.normalize is not None and self.normalize in cone.labels:
            pivot = directed[cone.labels.index(self.normalize)]
            if pivot != 0:
                return [item / abs(pivot) for item in directed]
        denominators = np.lcm.reduce([item.denominator for item in directed])
        whole = [int(item * int(denominators)) for item in directed]
        divisor = 0
        for item in whole:
            divisor = gcd(divisor, item)
        return [Fraction(item, divisor or 1) for item in whole]

    def _unique(self, vectors: Sequence[Sequence[Fraction]]) -> np.ndarray:
        """Return the distinct generators as a matrix of fractions, in a fixed order."""
        size = len(self._cone.labels)
        kept = sorted({tuple(self._scaled(vector)) for vector in vectors}, reverse=True)
        if self.sort is not None and self.sort in self._cone.labels:
            position = self._cone.labels.index(self.sort)
            kept.sort(key=lambda row: row[position], reverse=True)
        return np.asarray(kept, dtype=object).reshape(-1, size)

    def _run(self, inputs: object = None) -> None:
        """Set `modes_` and `lines_`.

        Args:
            inputs:
                Not read.

        Raises:
            EnumerationLimitError:
                The enumeration does not finish within the time limit.
        """
        rays = self._unique(self._mapped_rays(self._split_columns()))
        if self.kind == 'pathways':
            rays = self._independent(rays)
        kept = {tuple(row) for row in rays}
        opposite = [tuple(-item for item in row) in kept for row in rays]
        size = len(self._cone.labels)
        self.lines_ = np.asarray(
            [row for row, both in zip(rays, opposite, strict=True) if both and next(i for i in row if i != 0) > 0],
            dtype=object,
        ).reshape(-1, size)
        self.modes_ = rays if self.kind == 'modes' else rays[~np.asarray(opposite, dtype=bool)]
