"""Compile conservation laws of finite-volume fields into flux differences with limited reconstruction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, solve

from ..core import Expression, ValidationError, symbol
from ..model import Field, Flux, LatticeSpace
from ._kernels import Reference, Source
from ._printing import numpy_function

type Array = npt.NDArray[np.float64]

TIME = symbol('t')


def _limited(backward: Array, forward: Array, limiter: str | None) -> Array:
    """Return the limited slope of every cell from its backward and forward differences."""
    if limiter is None:
        return np.zeros_like(forward)
    same = backward * forward > 0
    if limiter == 'minmod':
        return np.where(same, np.sign(forward) * np.minimum(np.abs(backward), np.abs(forward)), 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(same, 2 * backward * forward / (backward + forward), 0.0)


@dataclass(eq=False)
class FiniteVolume:
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

    owner: str
    starts: tuple[int, ...]
    dimensions: tuple[int, ...]
    spacings: tuple[float, ...]
    boundaries: tuple[tuple[str, float], ...]
    fluxes: tuple[Callable[..., Any], ...]
    jacobians: tuple[Callable[..., Any], ...]
    parameters: tuple[Reference, ...]
    limiter: str | None
    riemann: str
    critical: tuple[Callable[..., Any], ...] = ()

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
        entries = self.jacobians[axis](0.0, *states, *parameters)
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
        values = self.fluxes[axis](0.0, *states, *parameters)
        return [np.broadcast_to(np.asarray(item, dtype=float), shape) for item in values]

    def _godunov(self, axis: int, left: Array, right: Array, parameters: list[float]) -> Array:
        """Return the exact Godunov flux of one state: the minimum flux between rising states, else the maximum."""
        low, high = np.minimum(left, right), np.maximum(left, right)
        candidates = [self._flux(axis, [left], parameters)[0], self._flux(axis, [right], parameters)[0]]
        for root in self.critical[axis](0.0, *parameters):
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

    def rates(self, x: Array, p: Array) -> Array:
        """Return the rates of the conserved states, minus the divergence of the numerical fluxes.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            rates:
                A vector of the size of the state vector with the rates of the conserved states.
        """
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
                slopes = _limited(backward, forward, self.limiter)
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


def compile_volumes(
    laws: Sequence[tuple[Flux, Reference]],
    references: Mapping[Expression, Reference],
    place: Field,
    owner: str,
) -> FiniteVolume:
    """Return the conservation laws of one finite-volume field.

    Args:
        laws:
            The flux laws of the field with the references of their targets.

        references:
            The references of the symbols of the field's namespace.

        place:
            The field.

        owner:
            The path of the field.

    Returns:
        volumes:
            The compiled conservation laws.

    Raises:
        ValidationError:
            The field is not a finite-volume field on a lattice, or the fluxes are not one per dimension.
    """
    space = place.space
    if place.discretization != 'finite-volume' or not isinstance(space, LatticeSpace):
        error_msg = f'{owner} declares fluxes, which need a finite-volume field on a lattice.'
        raise ValidationError(error_msg)
    states = [_symbol(law.target, references) for law, _ in laws]
    n_dim = space.n_dim
    expressions = []
    for law, _ in laws:
        components = law.fluxes
        if len(components) != n_dim:
            error_msg = f'A flux of {owner} has {len(components)} components for a {n_dim}D lattice.'
            raise ValidationError(error_msg)
        expressions.append([SR(item) for item in components])
    parameters = sorted(
        {
            item
            for row in expressions
            for value in row
            for item in value.variables()
            if item not in states and item in references
        },
        key=str,
    )
    unknown = {item for row in expressions for value in row for item in value.variables() if item not in states} - set(
        parameters,
    )
    if unknown:
        error_msg = f'The fluxes of {owner} read {", ".join(map(str, unknown))}, which are not states or parameters.'
        raise ValidationError(error_msg)
    columns = list(zip(*expressions, strict=True))
    fluxes = tuple(numpy_function(TIME, [*states, *parameters], list(column)) for column in columns)
    jacobians = tuple(
        numpy_function(TIME, [*states, *parameters], [value.diff(state) for value in column for state in states])
        for column in columns
    )
    first = laws[0][0]
    critical: tuple[Callable[..., Any], ...] = ()
    if first.riemann == 'godunov':
        if len(states) != 1:
            error_msg = f'The fluxes of {owner} ask for the Godunov flux, which is exact for one state only.'
            raise ValidationError(error_msg)
        roots = []
        for column in columns:
            found = [item.rhs() for item in solve(column[0].diff(states[0]) == 0, states[0])]
            roots.append(numpy_function(TIME, parameters, found) if found else numpy_function(TIME, parameters, []))
        critical = tuple(roots)
    boundaries = tuple(
        (kind, float(value) if not callable(value) else 0.0)
        for kind, value in (place.boundary_of(reference.entry.name.rpartition('.')[2]) for _, reference in laws)
    )
    return FiniteVolume(
        owner,
        tuple(reference.entry.start for _, reference in laws),
        space.dimensions,
        space.spacings,
        boundaries,
        fluxes,
        jacobians,
        tuple(references[item] for item in parameters),
        first.limiter,
        first.riemann,
        critical,
    )


def _symbol(target: Expression, references: Mapping[Expression, Reference]) -> Expression:
    """Return the member symbol of a flux target after checking it is a member state."""
    reference = references.get(target)
    if reference is None or not reference.is_state or reference.source is not Source.MEMBER:
        error_msg = f'A flux targets {target}, which is not a state of a field.'
        raise ValidationError(error_msg)
    return target
