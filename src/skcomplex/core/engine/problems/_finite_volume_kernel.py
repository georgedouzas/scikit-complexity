"""Compile conservation laws of finite-volume fields into flux differences with limited reconstruction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import Expression

from ...errors import SolverError, ValidationError
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
            The references of the other values the fluxes read, parameters or states that no flux conserves.

        limiter:
            The slope limiter.

        riemann:
            `'rusanov'` or `'godunov'`.

        reads:
            One function of the time and the vectors per value of `parameters` held by no vector, such as a moment
            over class axes or a field solved from it, and `None` for a value of a vector. An empty tuple reads the
            vectors.

        edges:
            The edges of every axis of a phase space, `'periodic'` or `'reflecting'` for a class axis, whose flux
            through the two ends of an interval is zero, and `None` for a position axis, which keeps the boundary of
            its states. An empty tuple gives every axis the boundary of its states.

        drifts:
            One function per dimension of the states and the parameters that returns the coefficient `a` of the flux
            `a u` of every state, or none when a flux is not such a linear drift.

        critical:
            For Godunov fluxes, one function per dimension of the parameters that returns the states where the flux
            has a minimum or maximum.

        expressions:
            The flux of every state, one column per dimension, in the symbols of the conserved states followed by the
            symbols of the parameters.

        speeds:
            One function per dimension of the states and the parameters that returns the closed-form eigenvalues of
            the flux Jacobian, or none when the Jacobian has no closed form.

    Attributes:
        values_:
            The rates of the conserved states of the last run, as a full vector.
    """

    values_: Array

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
        reads: tuple[Callable[[float, Array, Array], object] | None, ...] = (),
        edges: tuple[str | None, ...] = (),
        drifts: tuple[Callable[..., object], ...] = (),
        critical: tuple[Callable[..., object], ...] = (),
        expressions: tuple[tuple[Expression, ...], ...] = (),
        speeds: tuple[Callable[..., object], ...] = (),
    ) -> None:
        self.owner = owner
        self.starts = starts
        self.dimensions = dimensions
        self.spacings = spacings
        self.boundaries = boundaries
        self.fluxes = fluxes
        self.jacobians = jacobians
        self.parameters = parameters
        self.reads = reads
        self.edges = edges
        self.drifts = drifts
        self.limiter = limiter
        self.riemann = riemann
        self.critical = critical
        self.expressions = expressions
        self.speeds = speeds

    @property
    def size(self) -> int:
        """Number of cells."""
        return int(np.prod(self.dimensions))

    @property
    def closed_form(self) -> bool:
        """Whether the wave speeds come from closed-form eigenvalues instead of numerical ones."""
        return bool(self.speeds)

    def _window(self, axis: int, first: int, last: int) -> tuple[slice, ...]:
        """Return the index that selects the positions from first to last, excluded, along an axis."""
        index = [slice(None)] * len(self.dimensions)
        index[axis] = slice(first, last)
        return tuple(index)

    def _states(self, x: Array) -> list[Array]:
        """Return the conserved states in lattice shape."""
        return [np.reshape(x[start : start + self.size], self.dimensions) for start in self.starts]

    def _parameters(self, x: Array, p: Array, t: float = 0.0) -> list[float | Array]:
        """Return the other values the fluxes read, a number when uniform and the values of the cells otherwise."""
        values: list[float | Array] = []
        reads = self.reads or (None,) * len(self.parameters)
        for reference, read in zip(self.parameters, reads, strict=True):
            column = np.asarray(reference.column_of(x, p) if read is None else read(t, x, p), dtype=float)
            if column.size == self.size and not np.all(column == column[0]):
                values.append(np.reshape(column, self.dimensions))
            else:
                values.append(float(column[0]))
        return values

    def _sides(self, parameters: list[float | Array], axis: int) -> tuple[list[float | Array], list[float | Array]]:
        """Return the parameters of the cells left and right of every interface along an axis, numbers as they are."""
        n = self.dimensions[axis]
        width = [(0, 0)] * len(self.dimensions)
        width[axis] = (2, 2)
        periodic = self._kind(axis, self.boundaries[0][0]) == 'periodic'
        before, after = self._window(axis, 1, n + 2), self._window(axis, 2, n + 3)
        left: list[float | Array] = []
        right: list[float | Array] = []
        for value in parameters:
            if isinstance(value, float):
                left.append(value)
                right.append(value)
                continue
            padded = np.pad(value, width, mode='wrap') if periodic else np.pad(value, width, mode='edge')
            left.append(padded[before])
            right.append(padded[after])
        return left, right

    def _closed_speed(self, axis: int, states: Sequence[Array], parameters: Sequence[float | Array]) -> Array:
        """Return the largest absolute closed-form eigenvalue of the flux Jacobian of every cell or interface.

        Args:
            axis:
                The axis of the fluxes.

            states:
                The conserved states.

            parameters:
                The values of the parameters the fluxes read.

        Returns:
            speed:
                The largest wave speed of every cell or interface.

        Raises:
            SolverError:
                An eigenvalue is not real, so the states leave the hyperbolic region of the fluxes.
        """
        shape = np.broadcast_shapes(*(np.shape(item) for item in states))
        with np.errstate(invalid='ignore'):
            values = ExpressionPrinter._values_of(self.speeds[axis](0.0, *states, *parameters))
        speed = np.abs(np.broadcast_to(np.asarray(values[0], dtype=float), shape))
        for value in values[1:]:
            speed = np.maximum(speed, np.abs(np.asarray(value, dtype=float)))
        if np.isnan(speed).any():
            cell = tuple(int(index) for index in np.argwhere(np.isnan(speed))[0])
            error_msg = (
                f'The wave speeds of {self.owner} are not real at cell {cell}, so its states left the hyperbolic '
                'region of the fluxes. Reduce the step or check the initial values.'
            )
            raise SolverError(error_msg)
        return speed

    def _speed(self, axis: int, states: Sequence[Array], parameters: Sequence[float | Array]) -> Array:
        """Return the largest wave speed of every cell or interface, the spectral radius of the flux Jacobian."""
        if self.speeds:
            return self._closed_speed(axis, states, parameters)
        entries = ExpressionPrinter._values_of(self.jacobians[axis](0.0, *states, *parameters))
        count = len(states)
        shape = np.broadcast_shapes(*(np.shape(item) for item in states))
        matrix = np.stack([np.broadcast_to(np.asarray(item, dtype=float), shape) for item in entries], axis=-1)
        if count == 1:
            return np.abs(matrix[..., 0])
        eigenvalues = np.linalg.eigvals(matrix.reshape(*shape, count, count))
        return np.abs(eigenvalues).max(axis=-1)

    def _flux(self, axis: int, states: Sequence[Array], parameters: Sequence[float | Array]) -> list[Array]:
        """Return the physical flux of every state along an axis."""
        shape = np.broadcast_shapes(*(np.shape(item) for item in states))
        values = ExpressionPrinter._values_of(self.fluxes[axis](0.0, *states, *parameters))
        return [np.broadcast_to(np.asarray(item, dtype=float), shape) for item in values]

    def _godunov(self, axis: int, left: Array, right: Array, parameters: list[float | Array]) -> Array:
        """Return the exact Godunov flux of one state: the minimum flux between rising states, else the maximum.

        Args:
            axis:
                The axis of the interfaces.

            left:
                The state on the left of every interface.

            right:
                The state on the right of every interface.

            parameters:
                The values of the parameters the flux reads.

        Returns:
            flux:
                The flux through every interface.

        Raises:
            ValidationError:
                A parameter of the flux varies across the cells.
        """
        if not all(isinstance(value, float) for value in parameters):
            error_msg = f'The Godunov flux of {self.owner} takes uniform parameters. Use the Rusanov flux.'
            raise ValidationError(error_msg)
        low, high = np.minimum(left, right), np.maximum(left, right)
        candidates = [self._flux(axis, [left], parameters)[0], self._flux(axis, [right], parameters)[0]]
        for root in ExpressionPrinter._values_of(self.critical[axis](0.0, *parameters)):
            value = float(np.real(root))
            inside = (low <= value) & (value <= high)
            flux = self._flux(axis, [np.full(left.shape, value)], parameters)[0]
            candidates.append(np.where(inside, flux, np.where(left <= right, np.inf, -np.inf)))
        stacked = np.stack(candidates)
        return np.where(left <= right, stacked.min(axis=0), stacked.max(axis=0))

    def _kind(self, axis: int, given: str) -> str:
        """Return the boundary of a state along an axis, which on a class axis of a phase space is that of the axis."""
        edge = self.edges[axis] if self.edges else None
        if edge is None:
            return given
        return 'periodic' if edge == 'periodic' else 'zero-flux'

    def _pad(self, states: list[Array], axis: int) -> list[Array]:
        """Return the states with two ghost cells on both sides of an axis."""
        width = [(0, 0)] * len(self.dimensions)
        width[axis] = (2, 2)
        padded = []
        for values, (given, value) in zip(states, self.boundaries, strict=True):
            kind = self._kind(axis, given)
            if kind == 'periodic':
                padded.append(np.pad(values, width, mode='wrap'))
            elif kind == 'zero-flux':
                padded.append(np.pad(values, width, mode='edge'))
            else:
                padded.append(np.pad(values, width, mode='constant', constant_values=value))
        return padded

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the rates of the conserved states, minus the divergence of the numerical fluxes, as a full vector."""
        state = self._state_of(inputs)
        x, p = state.x, state.p
        states = self._states(x)
        parameters = self._parameters(x, p, state.t)
        totals = [np.zeros(self.dimensions) for _ in states]
        for axis in range(len(self.dimensions)):
            padded = self._pad(states, axis)
            n = self.dimensions[axis]
            behind, centre, ahead = (self._window(axis, offset, n + 2 + offset) for offset in range(3))
            left_cells, right_cells = (self._window(axis, offset, n + 1 + offset) for offset in range(2))
            left_faces, right_faces = (self._window(axis, offset, n + offset) for offset in range(2))
            left, right = [], []
            for values in padded:
                cells = values[centre]
                slopes = self._limited(cells - values[behind], values[ahead] - cells, self.limiter)
                left.append((cells + slopes / 2)[left_cells])
                right.append((cells - slopes / 2)[right_cells])
            if self.riemann == 'godunov' and len(states) == 1:
                interfaces = [self._godunov(axis, left[0], right[0], parameters)]
            else:
                before, after = self._sides(parameters, axis)
                speed = np.maximum(self._speed(axis, left, before), self._speed(axis, right, after))
                left_flux, right_flux = self._flux(axis, left, before), self._flux(axis, right, after)
                interfaces = [
                    (first + second) / 2 - speed / 2 * (upper - lower)
                    for first, second, lower, upper in zip(left_flux, right_flux, left, right, strict=True)
                ]
            for index, flux in enumerate(interfaces):
                walled = self._walled(flux, axis, n)
                totals[index] = totals[index] - (walled[right_faces] - walled[left_faces]) / self.spacings[axis]
        rates = np.zeros_like(x)
        for start, total in zip(self.starts, totals, strict=True):
            rates[start : start + self.size] = total.ravel()
        self.values_ = rates

    def _walled(self, flux: Array, axis: int, n: int) -> Array:
        """Return the flux through the interfaces of an axis, zero at the two ends of an interval class axis."""
        if not self.edges or self.edges[axis] != 'reflecting':
            return flux
        result = np.array(flux)
        result[self._window(axis, 0, 1)] = 0.0
        result[self._window(axis, n, n + 1)] = 0.0
        return result

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
        parameters = self._parameters(x, p)
        return max(float(self._speed(axis, states, parameters).max()) for axis in range(len(self.dimensions)))

    def drift_of(self, axis: int, states: Sequence[Array], x: Array, p: Array, t: float) -> list[Array]:
        """Return the coefficient of the linear drift of every state along an axis.

        Args:
            axis:
                The axis.

            states:
                The conserved states.

            x:
                The state vector.

            p:
                The parameter vector.

            t:
                The time.

        Returns:
            drifts:
                The coefficient of every state, in lattice shape where it varies.
        """
        values = ExpressionPrinter._values_of(self.drifts[axis](t, *states, *self._parameters(x, p, t)))
        return [np.asarray(item, dtype=float) for item in values]

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
