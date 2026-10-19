"""Integrate stiff fields with implicit-explicit, Crank-Nicolson, and fourth-order exponential time differencing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import splu

from ..base import Form, SolverError, symbol
from ..compile import Problem, Vector, mesh_laplacian, numpy_function, spectral_multiplier
from ..model import LatticeSpace, Mesh, Neighborhood, NetworkSpace
from ._base import Stepper, register_solver

CONTOUR_POINTS = 64
LATTICE_BOUNDARIES = {'periodic': 'periodic', 'zero-flux': 'reflecting', 'fixed': 'fixed'}
TIME = symbol('t')

type Complex = npt.NDArray[np.complex128]


def _coefficient(term: Any, p: Vector) -> Any:  # noqa: ANN401
    """Return the value of a linear term's coefficient, a number or one value per member."""
    symbols = sorted(term.references, key=str)
    values = []
    for item in symbols:
        reference = term.references[item]
        column = reference.entry.view(p)[:, reference.column]
        values.append(float(column[0]) if reference.entry.members is None else column)
    return numpy_function(TIME, symbols, [term.coefficient])(0.0, *values)[0]


def _rows(reference: Any) -> npt.NDArray[np.intp]:  # noqa: ANN401
    """Return the flat state index of a component for every member."""
    return reference.flat_indices


def _lattice_operator(name: str, axis: int, place: Any, reference: Any) -> csr_matrix:  # noqa: ANN401
    """Return the stencil matrix of a finite-difference operator on a field state, without its boundary values."""
    space = place.space
    size = place.size
    kind, _ = place.boundary_of(reference.entry.name.rpartition('.')[2])

    def shift(offset: Any) -> csr_matrix:  # noqa: ANN401
        lattice = LatticeSpace(space.dimensions, boundary=LATTICE_BOUNDARIES[kind])
        table = lattice.index_table(Neighborhood((offset,)))[:, 0]
        inside = table < size
        return csr_matrix((np.ones(inside.sum()), (np.flatnonzero(inside), table[inside])), shape=(size, size))

    if isinstance(space, NetworkSpace):
        adjacency = csr_matrix(nx.to_scipy_sparse_array(space.graph, nodelist=list(space.graph.nodes), weight=None))
        if name != 'laplacian':
            error_msg = f'Only Laplacians act on fields on networks, but a term takes the {name}.'
            raise SolverError(error_msg)
        return adjacency - diags(np.asarray(adjacency.sum(axis=1)).ravel())
    units = [tuple(int(other == index) for other in range(space.n_dim)) for index in range(space.n_dim)]
    spacings = space.spacings
    if name == 'laplacian':
        total = csr_matrix((size, size))
        for unit, step in zip(units, spacings, strict=True):
            total = total + (shift(unit) + shift(tuple(-item for item in unit)) - 2 * identity(size)) / step**2
        return total
    if name == 'partial':
        unit = units[axis]
        return (shift(unit) - shift(tuple(-item for item in unit))) / (2 * spacings[axis])
    error_msg = f'The {name} of a finite-difference field has no stencil. Use a spectral field.'
    raise SolverError(error_msg)


@dataclass
class _SpectralBlock:
    """Linear terms of the states of one spectral field, a small matrix per Fourier mode."""

    dimensions: tuple[int, ...]
    rows: list[npt.NDArray[np.intp]]
    matrix: Complex

    def transform(self, x: Vector) -> Complex:
        """Return the Fourier modes of every state of the block, stacked on the last axis."""
        return np.stack([np.fft.rfftn(np.reshape(x[rows], self.dimensions)) for rows in self.rows], axis=-1)

    def inverse(self, modes: Complex, out: Vector) -> None:
        """Write the states of Fourier modes stacked on the last axis."""
        for index, rows in enumerate(self.rows):
            out[rows] = np.fft.irfftn(modes[..., index], s=self.dimensions, axes=range(len(self.dimensions))).ravel()

    def apply(self, operator: Complex, x: Vector, out: Vector) -> None:
        """Write a matrix of every mode applied to the block's states."""
        self.inverse(np.einsum('...ij,...j->...i', operator, self.transform(x)), out)


def _term_matrix(term: Any, coefficient: Any) -> csr_matrix:  # noqa: ANN401
    """Return the sparse matrix of a linear term of a finite-difference or finite element field."""
    place = term.place
    matrix = identity(place.size, format='csr')
    for name, axis in reversed(term.chain):
        operator = (
            _mesh_operator(place.space, name)
            if isinstance(place.space, Mesh)
            else _lattice_operator(name, axis, place, term.source)
        )
        matrix = operator @ matrix
    return diags(np.broadcast_to(np.asarray(coefficient, dtype=float), place.size)) @ matrix


def _sparse(triplets: list[tuple[Any, Any, csr_matrix]], problem: Problem) -> csr_matrix | None:
    """Return the sparse matrix of linear terms over the whole state vector, without rows of fixed nodes."""
    if not triplets:
        return None
    size = problem.states.size
    rows, columns, values = [], [], []
    for target, source, matrix in triplets:
        block = matrix.tocoo()
        rows.append(_rows(target)[block.row])
        columns.append(_rows(source)[block.col])
        values.append(block.data)
    sparse = csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(size, size))
    if problem.constrained.size:
        keep = np.ones(size)
        keep[problem.constrained] = 0.0
        sparse = diags(keep) @ sparse
    return sparse


def _spectral_block(terms: list[tuple[Any, complex]]) -> '_SpectralBlock':
    """Return the Fourier block of the linear terms of the states of one spectral field."""
    place = terms[0][0].place
    states: dict[tuple[str, int], Any] = {}
    for term, _ in terms:
        for reference in (term.target, term.source):
            states.setdefault((reference.entry.name, reference.column), reference)
    order = {key: index for index, key in enumerate(states)}
    dimensions, spacings = place.space.dimensions, place.space.spacings
    matrix = np.zeros((*np.fft.rfftn(np.zeros(dimensions)).shape, len(states), len(states)), dtype=complex)
    for term, coefficient in terms:
        multiplier: Any = 1.0
        for name, axis in term.chain:
            multiplier = multiplier * spectral_multiplier(name, axis, dimensions, spacings)
        target = order[(term.target.entry.name, term.target.column)]
        source = order[(term.source.entry.name, term.source.column)]
        matrix[..., target, source] += coefficient * multiplier
    return _SpectralBlock(dimensions, [_rows(reference) for reference in states.values()], matrix)


@dataclass
class SplitOperator:
    """Linear part of the rates of a problem, in Fourier blocks for spectral fields and a sparse matrix otherwise.

    Args:
        size:
            The size of the state vector.

        blocks:
            The spectral blocks.

        sparse:
            The sparse matrix of the other linear terms, over the whole state vector.

        covered:
            The state indices of the spectral blocks.
    """

    size: int
    blocks: list[_SpectralBlock] = field(default_factory=list)
    sparse: csr_matrix | None = None
    covered: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    _solvers: dict[float, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, problem: Problem, p: Vector) -> 'SplitOperator':
        """Return the linear part of a problem's rates at parameter values.

        Args:
            problem:
                The compiled problem.

            p:
                The parameter vector.

        Returns:
            operator:
                The linear part.
        """
        operator = cls(problem.states.size)
        groups: dict[int, list[Any]] = {}
        triplets: list[tuple[Any, Any, csr_matrix]] = []
        for term in problem.linear:
            coefficient = _coefficient(term, p)
            if term.place.discretization != 'spectral':
                triplets.append((term.target, term.source, _term_matrix(term, coefficient)))
            elif not np.ndim(coefficient) or np.ptp(coefficient) == 0:
                groups.setdefault(id(term.place), []).append((term, complex(np.ravel(coefficient)[0])))
        operator.sparse = _sparse(triplets, problem)
        operator.blocks = [_spectral_block(terms) for terms in groups.values()]
        rows = [row for block in operator.blocks for row in block.rows]
        operator.covered = np.concatenate([np.zeros(0, dtype=np.intp), *rows]).astype(np.intp)
        return operator

    def apply(self, x: Vector) -> Vector:
        """Return the linear part applied to a state vector.

        Args:
            x:
                The state vector.

        Returns:
            values:
                The linear part of the rates.
        """
        out = np.zeros(self.size)
        if self.sparse is not None:
            out += self.sparse @ x
        for block in self.blocks:
            block.apply(block.matrix, x, out)
        return out

    def solve(self, factor: float, rhs: Vector) -> Vector:
        """Return the solution `y` of `(I - factor L) y = rhs`.

        Args:
            factor:
                The factor of the linear part, such as the step for implicit Euler.

            rhs:
                The right-hand side.

        Returns:
            solution:
                The solution.
        """
        if factor not in self._solvers:
            sparse = None
            if self.sparse is not None:
                sparse = splu((identity(self.size, format='csc') - factor * self.sparse).tocsc())
            inverses = [np.linalg.inv(np.eye(block.matrix.shape[-1]) - factor * block.matrix) for block in self.blocks]
            self._solvers[factor] = (sparse, inverses)
        sparse, inverses = self._solvers[factor]
        out = sparse.solve(rhs) if sparse is not None else rhs.copy()
        for block, inverse in zip(self.blocks, inverses, strict=True):
            block.apply(inverse, rhs, out)
        return out


def _mesh_operator(space: Mesh, name: str) -> csr_matrix:
    """Return the finite element Laplacian of a mesh."""
    if name != 'laplacian':
        error_msg = f'Only Laplacians act on finite element fields, but a term takes the {name}.'
        raise SolverError(error_msg)
    return mesh_laplacian(space)


class _SplitStepper(Stepper):
    """Base of steppers that treat the linear part of the rates of fields apart from the nonlinear remainder."""

    form: ClassVar[Form] = Form.ODE
    implicit: ClassVar[bool] = True

    def __init__(self, dt: float = 0.01) -> None:
        self.dt = dt

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the stepper and check that the problem has linear terms to treat apart.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator, unused.

        Raises:
            SolverError:
                The problem is not an ODE, has no linear terms of fields, or has delays or events.
        """
        super().initialize(problem, rng)
        if not problem.linear:
            error_msg = (
                f'{type(self).__name__} splits the rates of fields into linear and nonlinear parts, but the problem '
                'has no linear terms with constant coefficients. Use ScipyODE or RK4.'
            )
            raise SolverError(error_msg)
        if problem.delays or problem.events:
            error_msg = f'{type(self).__name__} does not integrate delays or events. Use ScipyODE or RK4.'
            raise SolverError(error_msg)
        self._parameters: Vector | None = None

    def _prepare(self, p: Vector) -> None:
        """Build the linear part for parameter values when they change."""
        if self._parameters is None or not np.array_equal(p, self._parameters):
            self.operator_ = SplitOperator.build(self.problem_, p)
            self._parameters = p.copy()
            self._built(p)

    def _built(self, p: Vector) -> None:
        """Hook that runs after the linear part is rebuilt."""

    def _remainder(self, t: float, x: Vector, p: Vector) -> Vector:
        """Return the nonlinear remainder of the rates, the rates minus their linear part."""
        return self.problem_.drift(t, x, p) - self.operator_.apply(x)  # type: ignore[misc]

    def _step(self, t: float, x: Vector, p: Vector, h: float) -> Vector:
        """Return the state after one step."""
        raise NotImplementedError

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate from time `t` to time `until` in equal steps no longer than `dt`.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The state vector at time `until`.
        """
        if until <= t:
            return x
        self._prepare(p)
        n_steps = max(1, math.ceil((until - t) / self.dt - 1e-9))
        step = (until - t) / n_steps
        state = x.copy()
        for index in range(n_steps):
            state = self._step(t + index * step, state, p, step)
        return state


class IMEXEuler(_SplitStepper):
    """Implicit-explicit Euler: the linear part implicit and the nonlinear remainder explicit, of first order.

    Every step solves `(I - h L) x_{n+1} = x_n + h N(x_n)`, with sparse solves for finite differences and finite
    elements and division per Fourier mode for spectral fields.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.models.fields import heat_diffusion
        >>> from skcomplex.core.solvers import IMEXEuler
        >>> problem = compile_system(heat_diffusion((16, 16)))
        >>> stepper = IMEXEuler(dt=1.0)
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> state = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 10.0)
        >>> bool(np.isclose(state.sum(), problem.initial_state.sum()))
        True
    """

    def _step(self, t: float, x: Vector, p: Vector, h: float) -> Vector:
        """Return the state after one implicit-explicit Euler step."""
        return self.operator_.solve(h, x + h * self._remainder(t, x, p))


class IMEX2(_SplitStepper):
    """Crank-Nicolson Adams-Bashforth scheme of second order: trapezoidal linear part and extrapolated remainder.

    Every step solves `(I - h L / 2) x_{n+1} = (I + h L / 2) x_n + h (3 N_n - N_{n-1}) / 2`, starting with one
    Crank-Nicolson step with an Euler remainder.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> from skcomplex.core.solvers import IMEX2
        >>> IMEX2(dt=0.1).dt
        0.1
    """

    def _built(self, p: Vector) -> None:
        """Forget the previous remainder when the linear part changes."""
        self._previous: Vector | None = None

    def _step(self, t: float, x: Vector, p: Vector, h: float) -> Vector:
        """Return the state after one Crank-Nicolson Adams-Bashforth step."""
        remainder = self._remainder(t, x, p)
        extrapolated = remainder if self._previous is None else 1.5 * remainder - 0.5 * self._previous
        self._previous = remainder
        return self.operator_.solve(h / 2, x + h / 2 * self.operator_.apply(x) + h * extrapolated)


class CrankNicolson(_SplitStepper):
    """Crank-Nicolson for the linear part with a Heun predictor-corrector for the remainder, of second order.

    The predictor takes a Crank-Nicolson step with the remainder at the start, and the corrector repeats it with the
    remainder averaged over the start and the predicted end. For linear fields it is the trapezoidal rule.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> from skcomplex.core.solvers import CrankNicolson
        >>> CrankNicolson(dt=0.5).dt
        0.5
    """

    def _step(self, t: float, x: Vector, p: Vector, h: float) -> Vector:
        """Return the state after one Crank-Nicolson step with a predicted remainder."""
        explicit = x + h / 2 * self.operator_.apply(x)
        start = self._remainder(t, x, p)
        predicted = self.operator_.solve(h / 2, explicit + h * start)
        average = (start + self._remainder(t + h, predicted, p)) / 2
        return self.operator_.solve(h / 2, explicit + h * average)


def _phi(z: Complex, h: float) -> list[Complex]:
    """Return `e^z`, `e^(z/2)`, and the coefficients `Q`, `f1`, `f2`, `f3` of ETDRK4 by contour integrals.

    The integrals average over a full circle of radius 1 around every eigenvalue, which avoids the cancellation of the
    formulas near zero for complex eigenvalues as well as real ones.
    """
    roots = np.exp(2j * np.pi * (np.arange(1, CONTOUR_POINTS + 1) - 0.5) / CONTOUR_POINTS)
    points = z[..., None] + roots
    exponential = np.exp(points)
    q = h * np.mean((np.exp(points / 2) - 1) / points, axis=-1)
    f1 = h * np.mean((-4 - points + exponential * (4 - 3 * points + points**2)) / points**3, axis=-1)
    f2 = h * np.mean((2 + points + exponential * (-2 + points)) / points**3, axis=-1)
    f3 = h * np.mean((-4 - 3 * points - points**2 + exponential * (4 - points)) / points**3, axis=-1)
    return [np.exp(z), np.exp(z / 2), q, f1, f2, f3]


class ETDRK4(_SplitStepper):
    """Fourth-order exponential time differencing Runge-Kutta of Cox and Matthews, for spectral fields.

    The linear part is integrated exactly per Fourier mode and the remainder by a fourth-order scheme, with the
    coefficients computed by contour integrals as Kassam and Trefethen propose. Coupled states of a field use a small
    matrix exponential per mode. States outside spectral fields follow classical RK4.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> from skcomplex.core.solvers import ETDRK4
        >>> ETDRK4(dt=0.25).dt
        0.25
    """

    def _built(self, p: Vector) -> None:
        """Check that the linear part is spectral and forget its coefficients."""
        if self.operator_.sparse is not None:
            error_msg = 'ETDRK4 needs the linear part on spectral fields. Use IMEX2 or CrankNicolson.'
            raise SolverError(error_msg)
        self._coefficients: dict[float, list[list[Complex]]] = {}

    def _matrices(self, h: float) -> list[list[Complex]]:
        """Return the six ETDRK4 matrices of every block for a step."""
        if h not in self._coefficients:
            blocks = []
            for block in self.operator_.blocks:
                values, vectors = np.linalg.eig(h * block.matrix)
                inverse = np.linalg.inv(vectors)
                blocks.append(
                    [np.einsum('...ij,...j,...jk->...ik', vectors, item, inverse) for item in _phi(values, h)],
                )
            self._coefficients[h] = blocks
        return self._coefficients[h]

    def _map(self, index: int, h: float, x: Vector) -> Vector:
        """Return one of the six ETDRK4 maps applied to a state vector, with its L = 0 limit outside blocks."""
        scalars = [1.0, 1.0, h / 2, h / 6, h / 6, h / 6]
        out = scalars[index] * x
        for block, matrices in zip(self.operator_.blocks, self._matrices(h), strict=True):
            block.apply(matrices[index], x, out)
        return out

    def _step(self, t: float, x: Vector, p: Vector, h: float) -> Vector:
        """Return the state after one ETDRK4 step."""
        n_x = self._remainder(t, x, p)
        half = self._map(1, h, x)
        a = half + self._map(2, h, n_x)
        n_a = self._remainder(t + h / 2, a, p)
        b = half + self._map(2, h, n_a)
        n_b = self._remainder(t + h / 2, b, p)
        c = self._map(1, h, a) + self._map(2, h, 2 * n_b - n_x)
        n_c = self._remainder(t + h, c, p)
        return self._map(0, h, x) + self._map(3, h, n_x) + 2 * self._map(4, h, n_a + n_b) + self._map(5, h, n_c)


class SSPRK(Stepper):
    """Strong-stability-preserving Runge-Kutta of order 2 or 3, with steps from the CFL condition of conservation laws.

    The schemes of Shu and Osher are convex combinations of Euler steps, so they keep the total variation bound of a
    limited finite-volume scheme. With finite-volume fields the step is `cfl * h / s`, for the cell size `h` and the
    largest wave speed `s`, and otherwise `dt`.

    Args:
        order:
            2 or 3.

        cfl:
            The Courant number.

        dt:
            The largest step, and the step without conservation laws.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.models.fluids import burgers
        >>> from skcomplex.core.solvers import SSPRK
        >>> problem = compile_system(burgers(64))
        >>> stepper = SSPRK()
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> state = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)
        >>> bool(np.isclose(state.sum(), problem.initial_state.sum()))
        True
    """

    form: ClassVar[Form] = Form.ODE

    def __init__(self, order: int = 3, cfl: float = 0.4, dt: float = 0.01) -> None:
        self.order = order
        self.cfl = cfl
        self.dt = dt

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the stepper and check its order.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator, unused.

        Raises:
            SolverError:
                The order is not 2 or 3, or the problem is not an ODE.
        """
        super().initialize(problem, rng)
        if self.order not in {2, 3}:
            error_msg = f'SSPRK has order 2 or 3, not {self.order}.'
            raise SolverError(error_msg)
        self.spacing_ = min((min(volume.spacings) for volume in problem.volumes), default=None)

    def _step_size(self, x: Vector, p: Vector) -> float:
        """Return the step allowed by the CFL condition, at most `dt`."""
        speed = self.problem_.max_speed(x, p)
        if speed is None or self.spacing_ is None or speed <= 0:
            return self.dt
        return min(self.dt, self.cfl * self.spacing_ / speed)

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate from time `t` to time `until` with steps of the CFL condition.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The state vector at time `until`.
        """
        drift = self.problem_.drift
        state, time = x.copy(), t
        while drift is not None and time < until - 1e-12 * max(1.0, abs(until)):
            h = min(self._step_size(state, p), until - time)
            first = state + h * drift(time, state, p)
            if self.order == 2:  # noqa: PLR2004
                state = (state + first + h * drift(time + h, first, p)) / 2
            else:
                second = 0.75 * state + 0.25 * (first + h * drift(time + h, first, p))
                state = state / 3 + 2 / 3 * (second + h * drift(time + h / 2, second, p))
            time += h
        return state


register_solver('ssp-rk', SSPRK)
register_solver('imex-euler', IMEXEuler)
register_solver('imex2', IMEX2)
register_solver('crank-nicolson', CrankNicolson)
register_solver('etdrk4', ETDRK4)
