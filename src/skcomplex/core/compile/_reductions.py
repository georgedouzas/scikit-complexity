"""Evaluate the reductions that laws read, in dependency order, before the kernels of every evaluation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse

from ..base import Expression
from ..model import reduce_values
from ._layout import Entry
from ._printing import numpy_function

type Vector = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.intp]


@dataclass(frozen=True)
class Read:
    """Values a symbol of a reduction stands for, one per reduced member or one shared value.

    Args:
        is_state:
            Whether the values come from the state vector, as opposed to the extended parameter vector.

        index:
            The flat index of a shared value, or the flat index of the value of every reduced member.

        entry:
            The layout entry the values belong to, or `None` for an element set.

        column:
            The component of the entry's member that the symbol stands for.

        stride:
            The distance between the values of two consecutive members in the vector.
    """

    is_state: bool
    index: int | Indices
    entry: Entry | None = None
    column: int = 0
    stride: int = 1

    @property
    def per_member(self) -> bool:
        """Whether the symbol has one value per reduced member."""
        return np.ndim(self.index) > 0


@dataclass
class CompiledReduction:
    """Reduction that laws read, compiled into a function of the values of its symbols.

    Args:
        symbol:
            The placeholder symbol that kernels read instead of the reduction.

        kind:
            `'sum'`, `'mean'`, `'max'`, `'min'`, `'var'`, or `'count'`.

        argument:
            The expression of one reduced member, with nested reductions replaced by their placeholders.

        condition:
            The condition that selects the members, or `None` for all members.

        weights:
            The constant weight matrix as a CSR array of shape `(readers, size)`, or `None`.

        entry:
            The hidden entry that holds the value, after the parameters in the extended parameter vector.

        size:
            The number of reduced members.

        group:
            The path of the reduced group, the labels of an element set, or `None` for no group.

        reads:
            A mapping from every symbol of the argument and the condition to its values.

        time:
            The time symbol.

        members:
            The entry of the reduced group that gives its size to compiled loops, or `None` for an element set.

        set_indices:
            The flat state indices of the elements of an element set, one per element, or `None`.
    """

    symbol: Expression
    kind: str
    argument: Expression
    condition: Expression | None
    weights: scipy.sparse.csr_array | None
    entry: Entry
    size: int
    group: str | None
    reads: dict[Expression, Read]
    time: Expression
    members: Entry | None = None
    set_indices: Indices | None = None
    _function: Callable[..., list[Any]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile the argument and the condition into one NumPy function."""
        self._symbols = sorted(self.reads, key=str)
        expressions = [self.argument, self.condition if self.condition is not None else 1]
        self._function = numpy_function(self.time, self._symbols, expressions)

    @property
    def symbols(self) -> list[Expression]:
        """Symbols the argument and the condition read, in the order the compiled function takes them after time."""
        return list(self._symbols)

    def gather(self, x: Vector, q: Vector) -> list[Any]:
        """Return the values of the symbols the reduction reads.

        Args:
            x:
                The state vector.

            q:
                The extended parameter vector.

        Returns:
            values:
                One value or one array of member values per symbol of `symbols`.
        """
        return [(x if read.is_state else q)[read.index] for read in (self.reads[item] for item in self._symbols)]

    def values(self, t: float, x: Vector, q: Vector) -> tuple[Vector, npt.NDArray[np.bool_]]:
        """Return the argument and the condition of every reduced member.

        Args:
            t:
                The time.

            x:
                The state vector.

            q:
                The extended parameter vector.

        Returns:
            values:
                The argument and the condition, one value per reduced member.
        """
        argument, condition = self._function(t, *self.gather(x, q))  # type: ignore[misc]
        return (
            np.broadcast_to(np.asarray(argument, dtype=float), (self.size,)),
            np.broadcast_to(np.asarray(condition, dtype=bool), (self.size,)),
        )

    def evaluate(self, t: float, x: Vector, q: Vector) -> None:
        """Write the value of the reduction into the extended parameter vector.

        Args:
            t:
                The time.

            x:
                The state vector.

            q:
                The extended parameter vector, which holds the parameters and the reductions before this one.
        """
        argument, condition = self.values(t, x, q)
        if self.weights is None:
            q[self.entry.start] = reduce_values(argument[condition], self.kind)
            return
        weighted = self.weights @ np.where(condition, argument, 0.0)
        if self.kind == 'mean':
            selected = int(np.count_nonzero(condition))
            weighted = weighted / selected if selected else np.zeros_like(weighted)
        q[self.entry.indices] = weighted


@dataclass
class ReductionPass:
    """Reductions of a problem, evaluated in dependency order into the extended parameter vector.

    Args:
        reductions:
            The compiled reductions, every one after the reductions it reads.

        parameters:
            The size of the parameter vector, where the reductions start.
    """

    reductions: tuple[CompiledReduction, ...] = ()
    parameters: int = 0

    @property
    def extent(self) -> int:
        """Size of the extended parameter vector."""
        return max((item.entry.start + item.entry.size for item in self.reductions), default=self.parameters)

    def extend(self, t: float, x: Vector, p: Vector) -> Vector:
        """Return the parameter vector followed by the value of every reduction at a state.

        Args:
            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            extended:
                The extended parameter vector.
        """
        q = np.empty(self.extent)
        q[: self.parameters] = p[: self.parameters]
        for reduction in self.reductions:
            reduction.evaluate(t, x, q)
        return q

    def wrap(self, function: Callable[[float, Vector, Vector], Any] | None) -> Callable[..., Any] | None:
        """Return a function of the state and the parameters that reads the reductions at the state it receives.

        Args:
            function:
                A function `f(t, x, q)` of the extended parameter vector, or `None`.

        Returns:
            wrapped:
                The function `f(t, x, p)`, or the function itself without reductions.
        """
        if function is None or not self.reductions:
            return function

        def reduced(t: float, x: Vector, p: Vector) -> Any:  # noqa: ANN401
            return function(t, x, self.extend(t, x, p))

        return reduced
