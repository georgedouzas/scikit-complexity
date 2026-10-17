"""Evaluate the laws of one component over its members or its pairs with array operations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import sympy
from scipy.spatial import cKDTree

from ..core import ValidationError
from ..model import NumericFunction
from ._layout import Entry, Layout
from ._problem import Vector

type Indices = npt.NDArray[np.intp]


class Source(Enum):
    """How a symbol is read for each evaluation index of a kernel."""

    MEMBER = 'member'
    SINGLE = 'single'
    FIRST = 'first'
    SECOND = 'second'


@dataclass(frozen=True)
class Reference:
    """Component of a state or parameter that a symbol stands for.

    Args:
        entry:
            The layout entry of the variable.

        is_state:
            Whether the variable is a state, as opposed to a parameter.

        column:
            The component of the variable of one member.

        source:
            How the component is read for each evaluation index.
    """

    entry: Entry
    is_state: bool
    column: int
    source: Source = Source.SINGLE

    def column_of(self, x: Vector, p: Vector) -> Vector:
        """Return the component of every member as a view into the state or parameter vector.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            column:
                A view with one value per member.
        """
        return self.entry.view(x if self.is_state else p)[:, self.column]

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
        return first_positions - (difference - self.period * np.round(difference / self.period))


@dataclass
class Output:
    """Value a kernel produces for one target component.

    Args:
        target:
            The target component, or `None` for the propensity of a jump.

        expression:
            The symbolic value, or `None` when a callable produces it.

        function:
            The callable that produces the values of every component of its target, or `None`.

        component:
            The component of the callable's result that this output holds.

        components:
            The number of components the callable returns per evaluation index.
    """

    target: Reference | None
    expression: sympy.Expr | None = None
    function: NumericFunction | None = None
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
    time: sympy.Symbol
    references: Mapping[sympy.Symbol, Reference]
    rows: int
    layout_states: Layout
    layout_parameters: Layout
    pairs: PairSearch | None = None
    outputs: list[Output] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compile the symbolic outputs into one NumPy function."""
        self._symbolic = [output for output in self.outputs if output.expression is not None]
        expressions = [output.expression for output in self.outputs if output.expression is not None]
        unknown = set().union(*(expression.free_symbols for expression in expressions)) - {self.time}
        unknown -= set(self.references)
        if unknown:
            names = ', '.join(sorted(map(str, unknown)))
            error_msg = (
                f'A law of {self.owner} uses the symbols {names}, which are not declared. Declare them with `state` '
                'or `parameter` and use them through the namespace `s` that `laws` receives.'
            )
            raise ValidationError(error_msg)
        used = set().union(*(expression.free_symbols for expression in expressions)) - {self.time}
        self._arguments = sorted(used, key=str)
        self._function: Callable[..., Any] | None = (
            sympy.lambdify([self.time, *self._arguments], expressions, modules='numpy', cse=True)
            if expressions
            else None
        )

    def _gather(
        self,
        symbol: sympy.Symbol,
        x: Vector,
        p: Vector,
        pairs: tuple[Indices, Indices] | None,
    ) -> Any:  # noqa: ANN401
        """Return the values of a symbol for every evaluation index."""
        reference = self.references[symbol]
        column = reference.column_of(x, p)
        if reference.source is Source.SINGLE:
            return column[0]
        if reference.source is Source.MEMBER or pairs is None:
            return column
        first, second = pairs
        if reference.source is Source.FIRST:
            return column[first]
        if self.pairs is not None and reference.entry == self.pairs.second and reference.is_state:
            return self.pairs.second_images(x, first, second)[:, reference.column]
        return column[second]

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
        symbolic: list[Any] = []
        if self._function is not None:
            symbolic = list(self._function(t, *(self._gather(symbol, x, p, pairs) for symbol in self._arguments)))
        values: list[Vector] = []
        callable_results: dict[int, Vector] = {}
        position = 0
        for output in self.outputs:
            if output.expression is not None:
                values.append(np.broadcast_to(np.asarray(symbolic[position], dtype=float), (size,)))
                position += 1
                continue
            key = id(output.function)
            if key not in callable_results:
                callable_results[key] = self._call(output, t, x, p, size)
            values.append(callable_results[key][:, output.component])
        return size, pairs, values

    def _call(self, output: Output, t: float, x: Vector, p: Vector, size: int) -> Vector:
        """Return the result of a callable as one row per evaluation index and one column per component."""
        values = {**self.layout_states.values(x), **self.layout_parameters.values(p)}
        result = np.asarray(output.function(t, values), dtype=float)  # type: ignore[misc]
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
    column = target.entry.view(output)[:, target.column]
    if target.source is Source.MEMBER:
        column += values
    elif target.source is Source.SINGLE:
        column[0] += values.sum()
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
    column = target.entry.view(output)[:, target.column]
    column[:] = values if target.source is Source.MEMBER else values[0]
