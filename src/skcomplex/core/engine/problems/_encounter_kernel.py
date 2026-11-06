"""Encounter law lowered to sparse contractions of the candidate, field, and test densities on activity grids."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable

import numba
import numpy as np

from ...specs import RunStateSpec, TableValuesSpec
from ...types import Array, Indices
from ._base import BaseKernel
from ._expression_printer import ExpressionPrinter


@numba.njit(cache=True)
def _fused_gain(entries: Indices, weights: Array, f: Array, g: Array, size: int) -> Array:  # pragma: no cover
    """Return the gain of every outcome activity in one fused loop over the nonzero entries, one column each."""
    gain = np.zeros(size)
    for entry in range(weights.size):
        gain[entries[2, entry]] += weights[entry] * f[entries[0, entry]] * g[entries[1, entry]]
    return gain


class EncounterKernel(BaseKernel):
    """Encounter law lowered to sparse contractions of the candidate, field, and test densities on activity grids.

    Args:
        owner:
            The path of the component that returns the law.

        kind:
            `'transition'` for a conservative encounter, or `'bilinear'` for a gain or a loss at a signed rate.

        candidate:
            The flat state indices of the candidate density, one per activity.

        field:
            The flat state indices of the field density.

        test:
            The flat state indices of the test density, which gains the particles.

        rate:
            The values of the encounter rate `eta[h, k]` of every candidate and field activity, flattened by row.

        entries:
            The candidate, field, and outcome activities of the nonzero outcome probabilities.

        probabilities:
            The values of the nonzero outcome probabilities.

        fused:
            Whether the gain runs in one fused loop compiled with Numba, rather than in NumPy.

        positions:
            The number of positions of a phase space, each with its own encounters over the classes, or 1.

    Attributes:
        values_:
            The contribution of the encounter to the rate of every state at the last run.
    """

    values_: Array

    def __init__(
        self,
        owner: str,
        kind: str,
        candidate: Indices,
        field: Indices,
        test: Indices,
        *,
        rate: TableValuesSpec,
        entries: tuple[Indices, Indices, Indices],
        probabilities: TableValuesSpec,
        fused: bool = False,
        positions: int = 1,
    ) -> None:
        self.owner = owner
        self.kind = kind
        self.candidate = candidate
        self.field = field
        self.test = test
        self.rate = rate
        self.entries = entries
        self.probabilities = probabilities
        self.fused = fused
        self.positions = positions

    @functools.cached_property
    def _functions(self) -> dict[int, Callable[..., object]]:
        """NumPy functions of the selected entries of the tables of expressions, compiled once, by table identity."""
        return {}

    def _values(self, table: TableValuesSpec, p: Array) -> Array:
        """Return the values of the selected entries of a table at a parameter vector."""
        values = table.table
        selected = values[table.positions] if values.ndim else np.full(table.positions[0].size, values[()])
        if selected.dtype != object:
            return np.asarray(selected, dtype=float)
        symbols = sorted(table.parameters, key=str)
        if id(table) not in self._functions:
            printer = ExpressionPrinter(time=table.time, compiled=True).prepare(symbols)
            self._functions[id(table)] = ExpressionPrinter._function_of(printer.run(list(selected)).code_)
        indices = np.asarray([table.parameters[item] for item in symbols], dtype=np.intp)
        computed = self._functions[id(table)](0.0, *(float(value) for value in p[indices]))
        return np.asarray(ExpressionPrinter._values_of(computed), dtype=float)

    def _field_shape(self) -> tuple[int, int]:
        """Return the number of candidate classes and of field classes, one class for a redistribution."""
        return self.candidate.size // self.positions, max(self.field.size // self.positions, 1)

    def _densities(self, x: Array) -> tuple[Array, Array]:
        """Return the candidate and the field density with one row per position, a field of ones without a field."""
        candidates, fields = self._field_shape()
        f = np.reshape(x[self.candidate], (self.positions, candidates))
        return f, (
            np.reshape(x[self.field], (self.positions, fields)) if self.field.size else np.ones((self.positions, 1))
        )

    def rate_matrix(self, p: Array) -> Array:
        """Return the encounter rate of every candidate activity with every field activity.

        Args:
            p:
                The parameter vector.

        Returns:
            rates:
                The matrix `eta[h, k]`.
        """
        return self._values(self.rate, p).reshape(*self._field_shape())

    def weights(self, p: Array) -> Array:
        """Return the rate times the outcome probability of every nonzero entry, `eta[h, k] A[h, k, i]`.

        Args:
            p:
                The parameter vector.

        Returns:
            weights:
                One weight per entry.
        """
        candidates, fields, _ = self.entries
        return self.rate_matrix(p)[candidates, fields] * self._values(self.probabilities, p)

    def _gain(self, x: Array, p: Array) -> Array:
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
        weights = self.weights(p)
        if self.fused and self.positions == 1 and self.field.size:
            return _fused_gain(np.stack(self.entries), weights, x[self.candidate], x[self.field], self.test.size)
        f, g = self._densities(x)
        classes = self.test.size // self.positions
        places = np.arange(self.positions)[:, None] * classes + outcomes
        gained = weights * f[:, candidates] * g[:, fields]
        return np.bincount(places.ravel(), gained.ravel(), minlength=self.test.size)

    def _loss(self, x: Array, p: Array) -> Array:
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
        f, g = self._densities(x)
        return (f * (g @ self.rate_matrix(p).T)).ravel()

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the contribution of the encounter to the rate of every state, zero outside its densities."""
        state = self._state_of(inputs)
        x, p = state.x, state.p
        rates = np.zeros(x.size)
        if self.kind == 'transition':
            rates[self.candidate] -= self._loss(x, p)
        rates[self.test] += self._gain(x, p)
        self.values_ = rates
