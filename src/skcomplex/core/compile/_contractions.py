"""Hold encounter laws lowered to sparse contractions of densities on activity grids, and evaluate their rates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numba
import numpy as np
import numpy.typing as npt

from ..base import Expression
from ._printing import numpy_function

type Vector = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.intp]


@numba.njit(cache=True)
def _fused_gain(entries: Indices, weights: Vector, f: Vector, g: Vector, size: int) -> Vector:  # pragma: no cover
    """Return the gain of every outcome activity in one fused loop over the nonzero entries, one column each."""
    gain = np.zeros(size)
    for entry in range(weights.size):
        gain[entries[2, entry]] += weights[entry] * f[entries[0, entry]] * g[entries[1, entry]]
    return gain


@dataclass
class TableValues:
    """Values of selected entries of a table of numbers or parameter expressions at a parameter vector.

    Args:
        table:
            The table, a float array or an object array of Sage expressions.

        positions:
            The indices of the selected entries, one array per dimension.

        parameters:
            The flat index in the parameter vector of every parameter symbol the selected entries read.

        time:
            The time symbol.
    """

    table: npt.NDArray[Any]
    positions: tuple[Indices, ...]
    parameters: Mapping[Expression, int]
    time: Expression
    _constant: Vector | None = field(default=None, init=False, repr=False)
    _function: Callable[..., list[Any]] | None = field(default=None, init=False, repr=False)
    _indices: Indices = field(default_factory=lambda: np.zeros(0, dtype=np.intp), init=False, repr=False)

    def __post_init__(self) -> None:
        """Keep a constant table as an array, and compile a table of expressions into one NumPy function."""
        selected = self.table[self.positions] if self.table.ndim else np.full(self.positions[0].size, self.table[()])
        if selected.dtype != object:
            self._constant = np.asarray(selected, dtype=float)
            return
        symbols = sorted(self.parameters, key=str)
        self._indices = np.asarray([self.parameters[item] for item in symbols], dtype=np.intp)
        self._function = numpy_function(self.time, symbols, list(selected))

    def values(self, p: Vector) -> Vector:
        """Return the values of the selected entries.

        Args:
            p:
                The parameter vector.

        Returns:
            values:
                One value per selected entry.
        """
        if self._constant is not None:
            return self._constant
        function = cast('Callable[..., list[Any]]', self._function)
        return np.asarray(function(0.0, *(float(value) for value in p[self._indices])), dtype=float)


@dataclass
class CompiledEncounter:
    """Encounter law lowered to sparse contractions of the candidate, field, and test densities on activity grids.

    Args:
        owner:
            The path of the component that returns the law.

        kind:
            `'transition'` for a conservative encounter, `'proliferation'`, or `'destruction'`.

        candidate:
            The flat state indices of the candidate density, one per activity.

        field:
            The flat state indices of the field density.

        test:
            The flat state indices of the test density, which gains the particles.

        rate:
            The values of the encounter rate `eta[h, k]` of every candidate and field activity, flattened by row.

        entries:
            The candidate, field, and outcome activities of the nonzero outcome probabilities, empty for a destruction.

        probabilities:
            The values of the nonzero outcome probabilities, or `None` for a destruction.

        fused:
            Whether the gain runs in one fused loop compiled with Numba, rather than in NumPy.
    """

    owner: str
    kind: str
    candidate: Indices
    field: Indices
    test: Indices
    rate: TableValues
    entries: tuple[Indices, Indices, Indices]
    probabilities: TableValues | None
    fused: bool = False

    def __repr__(self) -> str:
        """Return the owner, the kind, and the sizes of the activity grids."""
        sizes = f'{self.candidate.size}, {self.field.size}, {self.test.size}'
        return f'CompiledEncounter({self.owner!r}, {self.kind!r}, activities=({sizes}))'

    @property
    def conserves(self) -> bool:
        """Whether the encounter moves particles and so conserves the total density."""
        return self.kind == 'transition'

    def rate_matrix(self, p: Vector) -> Vector:
        """Return the encounter rate of every candidate activity with every field activity.

        Args:
            p:
                The parameter vector.

        Returns:
            rates:
                The matrix `eta[h, k]`.
        """
        return self.rate.values(p).reshape(self.candidate.size, self.field.size)

    def weights(self, p: Vector) -> Vector:
        """Return the rate times the outcome probability of every nonzero entry, `eta[h, k] A[h, k, i]`.

        Args:
            p:
                The parameter vector.

        Returns:
            weights:
                One weight per entry, empty for a destruction.
        """
        if self.probabilities is None:
            return np.zeros(0)
        candidates, fields, _ = self.entries
        return self.rate_matrix(p)[candidates, fields] * self.probabilities.values(p)

    def gain(self, x: Vector, p: Vector) -> Vector:
        """Return the rate at which every test activity gains particles.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            gain:
                One rate per test activity.
        """
        candidates, fields, outcomes = self.entries
        f, g, weights = x[self.candidate], x[self.field], self.weights(p)
        if self.fused:
            return _fused_gain(np.stack(self.entries), weights, f, g, self.test.size)
        return np.bincount(outcomes, weights * f[candidates] * g[fields], minlength=self.test.size)

    def loss(self, x: Vector, p: Vector) -> Vector:
        """Return the rate at which every candidate activity loses particles, `f_h sum_k eta[h, k] g_k`.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            loss:
                One rate per candidate activity.
        """
        return x[self.candidate] * (self.rate_matrix(p) @ x[self.field])

    def rates(self, x: Vector, p: Vector) -> Vector:
        """Return the contribution of the encounter to the rate of every state.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            rates:
                One rate per state, zero outside the candidate and test densities.
        """
        rates = np.zeros(x.size)
        if self.kind != 'proliferation':
            rates[self.candidate] -= self.loss(x, p)
        if self.kind != 'destruction':
            rates[self.test] += self.gain(x, p)
        return rates
