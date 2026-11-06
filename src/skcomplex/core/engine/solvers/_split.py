"""Base of steppers that treat the linear part of the rates of fields apart from the nonlinear remainder."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import SuperLU, splu

from ...components.elements import FieldElement
from ...components.laws import BoundaryLaw, FluxLaw, LinearProgramLaw, RateLaw, TransferLaw
from ...components.spaces import (
    CayleySpace,
    ContinuousSpace,
    ContinuousTime,
    LatticeNeighborhood,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
)
from ...errors import SolverError
from ...specs import SupportSpec
from ...types import Array, ComplexArray, FormKind, OrderKind
from .._utils import symbol
from ..compile._field_lowering import FieldLowering
from ..problems import ExpressionPrinter, LinearTermSpec, ProblemSpec, ReferenceSpec
from ..problems._law_kernel import LATTICE_BOUNDARIES
from ..problems._steps import step_count
from ..types import SpectralBlock
from ._solver import Solver

TIME = symbol('t')


class SplitSolver(Solver):
    """Base of steppers that treat the linear part of the rates of fields apart from the nonlinear remainder.

    The linear part is held in Fourier blocks for spectral fields, a small matrix per Fourier mode, and in a sparse
    matrix over the whole state vector for the other fields. Every block is the shape of the field, the state indices
    of its states, and its matrices.

    Args:
        dt:
            The largest internal time step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Attributes:
        sparse_:
            The sparse matrix of the linear terms outside spectral fields, or `None` without them.

        blocks_:
            The spectral blocks.
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={LinearProgramLaw, RateLaw, FluxLaw, BoundaryLaw, TransferLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

    @classmethod
    def _is_implicit(cls) -> bool:
        """Return whether the solver treats diffusion implicitly, so explicit stability limits do not apply."""
        return True

    def __init__(
        self,
        dt: float = 0.01,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the stepper and check that the problem has linear terms to treat apart.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem is not an ODE, has no linear terms of fields, or has delays or events.
        """
        super()._prepare(context)
        problem = self.problem_
        if not problem.linear:
            error_msg = (
                f'{type(self).__name__} splits the rates of fields into linear and nonlinear parts, but the problem '
                'has no linear terms with constant coefficients. Use ScipyODESolver or RK4Solver.'
            )
            raise SolverError(error_msg)
        if problem.delays or problem.events:
            error_msg = f'{type(self).__name__} does not integrate delays or events. Use ScipyODESolver or RK4Solver.'
            raise SolverError(error_msg)
        self._parameters: Array | None = None
        self._last: tuple[float, Array, float] | None = None

    def _continues(self, t: float, x: Array, h: float) -> bool:
        """Check whether a call continues the time, state, and step of the previous call."""
        if self._last is None:
            return False
        time, state, step = self._last
        tolerance = 1e-9 * max(1.0, abs(t))
        return abs(t - time) <= tolerance and abs(h - step) <= tolerance and np.array_equal(x, state)

    def _forget(self) -> None:
        """Hook that forgets the steps of earlier calls, which multistep schemes override."""

    def _build_operator(self, p: Array) -> None:
        """Build the linear part for parameter values when they change."""
        if self._parameters is None or not np.array_equal(p, self._parameters):
            groups: dict[int, list[tuple[LinearTermSpec, complex]]] = {}
            triplets: list[tuple[ReferenceSpec, ReferenceSpec, csr_matrix]] = []
            for term in self.problem_.linear:
                coefficient = self._coefficient(term, p)
                if term.place.discretization != 'spectral':
                    triplets.append((term.target, term.source, self._term_matrix(term, coefficient)))
                elif not np.ndim(coefficient) or np.ptp(coefficient) == 0:
                    groups.setdefault(id(term.place), []).append((term, complex(np.ravel(coefficient)[0])))
            self.sparse_ = self._sparse(triplets, self.problem_)
            self.blocks_ = [self._spectral_block(terms) for terms in groups.values()]
            self._inverses: dict[float, tuple[SuperLU | None, list[ComplexArray]]] = {}
            self._parameters = p.copy()
            self._built(p)

    def _built(self, p: Array) -> None:
        """Hook that runs after the linear part is rebuilt."""

    def _remainder(self, t: float, x: Array, p: Array) -> Array:
        """Return the nonlinear remainder of the rates, the rates minus their linear part."""
        return self._drift()(t, x, p) - self._linear(x)

    def _linear(self, x: Array) -> Array:
        """Return the linear part of the rates at a state vector.

        Args:
            x:
                The state vector.

        Returns:
            values:
                The linear part of the rates.
        """
        out = np.zeros(self.problem_.states.size)
        if self.sparse_ is not None:
            out += self.sparse_ @ x
        for block in self.blocks_:
            self._apply_block(block, block[2], x, out)
        return out

    def _solve_linear(self, factor: float, rhs: Array) -> Array:
        """Return the solution `y` of `(I - factor L) y = rhs`, for the linear part `L` of the rates.

        Args:
            factor:
                The factor of the linear part, such as the step for implicit Euler.

            rhs:
                The right-hand side.

        Returns:
            solution:
                The solution.
        """
        if factor not in self._inverses:
            sparse = None
            if self.sparse_ is not None:
                size = self.problem_.states.size
                sparse = splu((identity(size, format='csc') - factor * self.sparse_).tocsc())
            inverses = [np.linalg.inv(np.eye(matrix.shape[-1]) - factor * matrix) for _, _, matrix in self.blocks_]
            self._inverses[factor] = (sparse, inverses)
        sparse, inverses = self._inverses[factor]
        out = sparse.solve(rhs) if sparse is not None else rhs.copy()
        for block, inverse in zip(self.blocks_, inverses, strict=True):
            self._apply_block(block, inverse, rhs, out)
        return out

    @staticmethod
    def _apply_block(block: SpectralBlock, operator: ComplexArray, x: Array, out: Array) -> None:
        """Write a matrix of every Fourier mode of a spectral block applied to the states of the block.

        Args:
            block:
                The spectral block.

            operator:
                One matrix per Fourier mode.

            x:
                The state vector the block reads.

            out:
                The vector the block writes its states into.
        """
        dimensions, rows, _ = block
        modes = np.stack([np.fft.rfftn(np.reshape(x[item], dimensions)) for item in rows], axis=-1)
        applied = np.einsum('...ij,...j->...i', operator, modes)
        for index, item in enumerate(rows):
            out[item] = np.fft.irfftn(applied[..., index], s=dimensions, axes=range(len(dimensions))).ravel()

    def _step(self, t: float, x: Array, p: Array, h: float) -> Array:
        """Return the state after one step."""
        raise NotImplementedError

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
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
        self._build_operator(p)
        n_steps = step_count(t, until, self.dt)
        step = (until - t) / n_steps
        if not self._continues(t, x, step):
            self._forget()
        state = x.copy()
        for index in range(n_steps):
            time = t + (index + 1) * step
            following = self._step(t + index * step, state, p, step)
            state = self._bounded(time, following, p)
            if not np.array_equal(state, following):
                self._forget()
        self._last = (until, state.copy(), step)
        return state

    @staticmethod
    def _lattice_operator(name: str, axis: int, place: FieldElement, reference: ReferenceSpec) -> csr_matrix:
        """Return the stencil matrix of a finite-difference operator on a field state, without its boundary values."""
        space = place.space
        size = place.size_
        kind, _ = place._get_boundary(reference.entry.name.rpartition('.')[2])
        if isinstance(space, NetworkSpace):
            adjacency = csr_matrix(nx.to_scipy_sparse_array(space.graph, nodelist=list(space.graph.nodes), weight=None))
            if name != 'laplacian':
                error_msg = f'Only Laplacians act on fields on networks, but a term takes the {name}.'
                raise SolverError(error_msg)
            return adjacency - diags(np.asarray(adjacency.sum(axis=1)).ravel())
        if not isinstance(space, LatticeSpace):
            error_msg = f'Finite-difference fields live on lattices or networks, not on a {type(space).__name__}.'
            raise SolverError(error_msg)
        shape = space.shape_

        def shift(offset: tuple[int, ...]) -> csr_matrix:
            lattice = LatticeSpace(shape, boundary=LATTICE_BOUNDARIES[kind]).prepare()
            table = lattice._index_table(LatticeNeighborhood((offset,)))[:, 0]
            inside = table < size
            return csr_matrix((np.ones(inside.sum()), (np.flatnonzero(inside), table[inside])), shape=(size, size))

        units = [tuple(int(other == index) for other in range(space.n_dim_)) for index in range(space.n_dim_)]
        spacings = space.spacings_
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

    @staticmethod
    def _term_matrix(term: LinearTermSpec, coefficient: float | Array) -> csr_matrix:
        """Return the sparse matrix of a linear term of a finite-difference or finite element field."""
        place = term.place
        matrix = identity(place.size_, format='csr')
        for name, axis in reversed(term.chain):
            operator = (
                SplitSolver._mesh_operator(place.space, name)
                if isinstance(place.space, MeshSpace)
                else SplitSolver._lattice_operator(name, axis, place, term.source)
            )
            matrix = operator @ matrix
        return diags(np.broadcast_to(np.asarray(coefficient, dtype=float), place.size_)) @ matrix

    @staticmethod
    def _sparse(
        triplets: list[tuple[ReferenceSpec, ReferenceSpec, csr_matrix]],
        problem: ProblemSpec,
    ) -> csr_matrix | None:
        """Return the sparse matrix of linear terms over the whole state vector, without rows of fixed nodes."""
        if not triplets:
            return None
        size = problem.states.size
        rows, columns, values = [], [], []
        for target, source, matrix in triplets:
            block = matrix.tocoo()
            rows.append(SplitSolver._rows(target)[block.row])
            columns.append(SplitSolver._rows(source)[block.col])
            values.append(block.data)
        sparse = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(size, size),
        )
        if problem.constrained.size:
            keep = np.ones(size)
            keep[problem.constrained] = 0.0
            sparse = diags(keep) @ sparse
        return sparse

    @staticmethod
    def _spectral_block(terms: list[tuple[LinearTermSpec, complex]]) -> SpectralBlock:
        """Return the Fourier block of the linear terms of the states of one spectral field."""
        place = terms[0][0].place
        states: dict[tuple[str, int], ReferenceSpec] = {}
        for term, _ in terms:
            for reference in (term.target, term.source):
                states.setdefault((reference.entry.name, reference.column), reference)
        order = {key: index for index, key in enumerate(states)}
        space = place.space
        if not isinstance(space, LatticeSpace):
            error_msg = f'Spectral fields live on lattices, not on a {type(space).__name__}.'
            raise SolverError(error_msg)
        dimensions, spacings = space.shape_, space.spacings_
        matrix = np.zeros((*np.fft.rfftn(np.zeros(dimensions)).shape, len(states), len(states)), dtype=complex)
        for term, coefficient in terms:
            multiplier: complex | ComplexArray = 1.0
            for name, axis in term.chain:
                multiplier = multiplier * FieldLowering._spectral_multiplier(name, axis, dimensions, spacings)
            target = order[(term.target.entry.name, term.target.column)]
            source = order[(term.source.entry.name, term.source.column)]
            matrix[..., target, source] += coefficient * multiplier
        return dimensions, [SplitSolver._rows(reference) for reference in states.values()], matrix

    @staticmethod
    def _mesh_operator(space: MeshSpace, name: str) -> csr_matrix:
        """Return the finite element Laplacian of a mesh."""
        if name != 'laplacian':
            error_msg = f'Only Laplacians act on finite element fields, but a term takes the {name}.'
            raise SolverError(error_msg)
        return FieldLowering._mesh_laplacian(space)

    @staticmethod
    def _coefficient(term: LinearTermSpec, p: Array) -> float | Array:
        """Return the value of a linear term's coefficient, a number or one value per member."""
        symbols = sorted(term.references, key=str)
        values = []
        for item in symbols:
            reference = term.references[item]
            column = reference.entry.view(p)[:, reference.column]
            values.append(float(column[0]) if reference.entry.members is None else column)
        function = ExpressionPrinter(time=TIME, compiled=True).prepare(symbols).run([term.coefficient]).code_
        if not callable(function):
            error_msg = f'The printer returned a {type(function).__name__}, not a compiled function.'
            raise TypeError(error_msg)
        coefficients = function(0.0, *values)
        if not isinstance(coefficients, list):
            error_msg = f'The compiled function returned a {type(coefficients).__name__}, not a list of values.'
            raise TypeError(error_msg)
        coefficient = np.asarray(coefficients[0], dtype=float)
        return float(coefficient) if coefficient.ndim == 0 else coefficient

    @staticmethod
    def _rows(reference: ReferenceSpec) -> npt.NDArray[np.intp]:
        """Return the flat state index of a component for every member."""
        return reference.flat_indices
