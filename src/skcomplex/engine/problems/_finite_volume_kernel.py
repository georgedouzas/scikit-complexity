"""Compile conservation laws of finite-volume fields into flux differences with limited reconstruction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse

from ...specs import RunStateSpec
from ...types import Array
from .._utils import symbol
from ._base import BaseKernel
from ._expression_printer import ExpressionPrinter
from ._reference_spec import ReferenceSpec

TIME = symbol('t')
_STENCIL = 2


class FiniteVolumeKernel(BaseKernel):
    """Conservation laws of the states of one finite-volume field.

    Args:
        owner:
            The path of the field.

        starts:
            The index of the first component of every conserved state.

        dimensions:
            The lattice shape.

        spacings:
            The size of the cells along every axis.

        boundaries:
            The boundary condition of every state and its fixed value.

        fluxes:
            One function per dimension of the states and the parameters that returns the flux of every state.

        jacobians:
            One function per dimension that returns the flux Jacobian, row by row.

        parameters:
            The parameter references the fluxes read.

        limiter:
            The slope limiter.

        riemann:
            `'rusanov'` or `'godunov'`.

        critical:
            For Godunov fluxes, one function per dimension of the parameters that returns the states where the flux
            has a minimum or maximum.
    """

    def __init__(
        self,
        owner: str,
        starts: tuple[int, ...],
        dimensions: tuple[int, ...],
        spacings: tuple[float, ...],
        boundaries: tuple[tuple[str, float], ...],
        *,
        fluxes: tuple[Callable[..., object], ...],
        jacobians: tuple[Callable[..., object], ...],
        parameters: tuple[ReferenceSpec, ...],
        limiter: str | None,
        riemann: str,
        critical: tuple[Callable[..., object], ...] = (),
    ) -> None:
        self.owner = owner
        self.starts = starts
        self.dimensions = dimensions
        self.spacings = spacings
        self.boundaries = boundaries
        self.fluxes = fluxes
        self.jacobians = jacobians
        self.parameters = parameters
        self.limiter = limiter
        self.riemann = riemann
        self.critical = critical

    @property
    def size(self) -> int:
        """Number of cells."""
        return int(np.prod(self.dimensions))

    def _states(self, x: Array) -> list[Array]:
        """Return the conserved states in lattice shape."""
        return [np.reshape(x[start : start + self.size], self.dimensions) for start in self.starts]

    def _parameters(self, p: Array) -> list[float]:
        """Return the values of the parameters the fluxes read."""
        return [float(reference.column_of(np.zeros(0), p)[0]) for reference in self.parameters]

    def _speed(self, axis: int, states: Sequence[Array], parameters: list[float]) -> Array:
        """Return the largest wave speed of every cell or interface, the spectral radius of the flux Jacobian."""
        entries = ExpressionPrinter._values_of(self.jacobians[axis](0.0, *states, *parameters))
        count = len(states)
        shape = np.broadcast_shapes(*(np.shape(item) for item in states))
        matrix = np.stack([np.broadcast_to(np.asarray(item, dtype=float), shape) for item in entries], axis=-1)
        if count == 1:
            return np.abs(matrix[..., 0])
        eigenvalues = np.linalg.eigvals(matrix.reshape(*shape, count, count))
        return np.abs(eigenvalues).max(axis=-1)

    def _flux(self, axis: int, states: Sequence[Array], parameters: list[float]) -> list[Array]:
        """Return the physical flux of every state along an axis."""
        shape = np.broadcast_shapes(*(np.shape(item) for item in states))
        values = ExpressionPrinter._values_of(self.fluxes[axis](0.0, *states, *parameters))
        return [np.broadcast_to(np.asarray(item, dtype=float), shape) for item in values]

    def _godunov(self, axis: int, left: Array, right: Array, parameters: list[float]) -> Array:
        """Return the exact Godunov flux of one state: the minimum flux between rising states, else the maximum."""
        low, high = np.minimum(left, right), np.maximum(left, right)
        candidates = [self._flux(axis, [left], parameters)[0], self._flux(axis, [right], parameters)[0]]
        for root in ExpressionPrinter._values_of(self.critical[axis](0.0, *parameters)):
            value = float(np.real(root))
            inside = (low <= value) & (value <= high)
            flux = self._flux(axis, [np.full(left.shape, value)], parameters)[0]
            candidates.append(np.where(inside, flux, np.where(left <= right, np.inf, -np.inf)))
        stacked = np.stack(candidates)
        return np.where(left <= right, stacked.min(axis=0), stacked.max(axis=0))

    def _pad(self, states: list[Array], axis: int) -> list[Array]:
        """Return the states with two ghost cells on both sides of an axis."""
        width = [(0, 0)] * len(self.dimensions)
        width[axis] = (2, 2)
        padded = []
        for values, (kind, value) in zip(states, self.boundaries, strict=True):
            if kind == 'periodic':
                padded.append(np.pad(values, width, mode='wrap'))
            elif kind == 'zero-flux':
                padded.append(np.pad(values, width, mode='edge'))
            else:
                padded.append(np.pad(values, width, mode='constant', constant_values=value))
        return padded

    def _run(self, inputs: RunStateSpec | None) -> Array:
        """Return the rates of the conserved states, minus the divergence of the numerical fluxes, as a full vector."""
        state = self._state_of(inputs)
        x, p = state.x, state.p
        states = self._states(x)
        parameters = self._parameters(p)
        totals = [np.zeros(self.dimensions) for _ in states]
        for axis in range(len(self.dimensions)):
            padded = self._pad(states, axis)
            n = self.dimensions[axis]

            def take(values: Array, first: int, last: int, axis: int = axis) -> Array:
                return np.take(values, range(first, last), axis=axis)

            left, right = [], []
            for values in padded:
                backward = take(values, 1, n + 3) - take(values, 0, n + 2)
                forward = take(values, 2, n + 4) - take(values, 1, n + 3)
                slopes = self._limited(backward, forward, self.limiter)
                cells = take(values, 1, n + 3)
                left.append(take(cells + slopes / 2, 0, n + 1))
                right.append(take(cells - slopes / 2, 1, n + 2))
            if self.riemann == 'godunov' and len(states) == 1:
                interfaces = [self._godunov(axis, left[0], right[0], parameters)]
            else:
                speed = np.maximum(self._speed(axis, left, parameters), self._speed(axis, right, parameters))
                left_flux, right_flux = self._flux(axis, left, parameters), self._flux(axis, right, parameters)
                interfaces = [
                    (first + second) / 2 - speed / 2 * (upper - lower)
                    for first, second, lower, upper in zip(left_flux, right_flux, left, right, strict=True)
                ]
            for index, flux in enumerate(interfaces):
                totals[index] = totals[index] - (take(flux, 1, n + 1) - take(flux, 0, n)) / self.spacings[axis]
        rates = np.zeros_like(x)
        for start, total in zip(self.starts, totals, strict=True):
            rates[start : start + self.size] = total.ravel()
        return rates

    def max_speed(self, x: Array, p: Array) -> float:
        """Return the largest wave speed over the cells and the dimensions.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            speed:
                The largest wave speed.
        """
        states = self._states(x)
        parameters = self._parameters(p)
        return max(float(self._speed(axis, states, parameters).max()) for axis in range(len(self.dimensions)))

    def pattern(self, size: int) -> scipy.sparse.csr_array:
        """Return the band of the reconstruction stencil, which couples all the conserved states.

        Args:
            size:
                The size of the state vector.

        Returns:
            pattern:
                A sparse matrix of the size of the state vector, nonzero where the stencil couples two states.
        """
        grid = np.arange(self.size).reshape(self.dimensions)
        kinds = {kind for kind, _ in self.boundaries}
        rows, columns = [], []
        for axis, length in enumerate(self.dimensions):
            for offset in range(-_STENCIL, _STENCIL + 1):
                shifted = np.arange(length) + offset
                neighbours = []
                if 'periodic' in kinds:
                    neighbours.append(np.mod(shifted, length))
                if kinds - {'periodic'}:
                    neighbours.append(np.clip(shifted, 0, length - 1))
                for items in neighbours:
                    rows.append(grid.ravel())
                    columns.append(np.take(grid, items, axis=axis).ravel())
        cells, others = np.concatenate(rows), np.concatenate(columns)
        pairs = [(first + cells, second + others) for first in self.starts for second in self.starts]
        first = np.concatenate([item for item, _ in pairs])
        second = np.concatenate([item for _, item in pairs])
        return scipy.sparse.csr_array((np.ones(first.size), (first, second)), shape=(size, size))

    @staticmethod
    def _limited(backward: Array, forward: Array, limiter: str | None) -> Array:
        """Return the limited slope of every cell from its backward and forward differences."""
        if limiter is None:
            return np.zeros_like(forward)
        same = backward * forward > 0
        if limiter == 'minmod':
            return np.where(same, np.sign(forward) * np.minimum(np.abs(backward), np.abs(forward)), 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(same, 2 * backward * forward / (backward + forward), 0.0)
