"""Advance quantum problems with QuTiP by unitary, Lindblad, and quantum trajectory evolution."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp

from ...core.laws import DissipatorLaw, GeneratorLaw
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError
from ...types import Array
from ..compile.passes._quantum_compiler import DEFAULT_MAX_DIMENSION
from ..problems import Form, OrderKind, State, SupportSpec
from ..problems._quantum_kernel import QuantumKernel
from ..problems._quantum_problem import QuantumProblem
from ._base import BaseSolver


class QuantumSolver(BaseSolver[QuantumProblem]):
    """Base class of solvers that advance quantum problems with a stepwise QuTiP solver.

    The solver is created at the first frame and again whenever the parameters change, from the current state, so a
    parameter update or a schedule applies from the next frame on. Coefficients that depend on time stay functions of
    time, and the others are folded into constant matrices.

    Args:
        atol:
            The absolute tolerance of the integrator.

        rtol:
            The relative tolerance of the integrator.

        nsteps:
            The largest number of internal steps between two frames.

        max_dimension:
            The largest dimension of the Hilbert space the solver accepts, which compiling checks.
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={GeneratorLaw, DissipatorLaw},
            orders={OrderKind.INTEGER},
        )

    @classmethod
    def _get_form(cls) -> Form:
        """Return the form the solver solves."""
        return Form.QUANTUM

    def __init__(
        self,
        atol: float = 1e-12,
        rtol: float = 1e-10,
        nsteps: int = 100_000,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.nsteps = nsteps
        self.max_dimension = max_dimension
        self.random_state = random_state
        self.until = until

    def _operator(self, terms: Sequence[QuantumKernel], p: Array) -> qutip.Qobj | qutip.QobjEvo:
        """Return the QuTiP operator of terms, constant or as a function of time, at fixed parameters."""
        problem = self.quantum_
        dims = [list(problem.dimensions)] * 2
        size = math.prod(problem.dimensions)
        constant = qutip.Qobj(sp.csr_array((size, size), dtype=complex), dims=dims)
        varying: list[Any] = []
        for term in terms:
            matrix = qutip.Qobj(term.matrix, dims=dims)
            if term.time_dependent:
                varying.append([matrix, self._coefficient(term, p)])
            else:
                constant = constant + complex(term.run(State(0.0, np.zeros(0), p))) * matrix
        return qutip.QobjEvo([constant, *varying]) if varying else constant

    def _ket(self, vector: npt.NDArray[np.complex128]) -> qutip.Qobj:
        """Return a state vector as a QuTiP ket with the dimensions of the problem."""
        dimensions = list(self.quantum_.dimensions)
        return qutip.Qobj(vector.reshape(-1, 1), dims=[dimensions, [1] * len(dimensions)])

    def _start_state(self, x: Array) -> qutip.Qobj:
        """Return the QuTiP state a solver starts from."""
        problem = self.quantum_
        state = problem._state_of(x)
        if problem.pure:
            return self._ket(state)
        return qutip.Qobj(state, dims=[list(problem.dimensions)] * 2)

    def _stored(self, state: qutip.Qobj) -> Array:
        """Return a QuTiP state as a state vector of the problem."""
        problem = self.quantum_
        entry = problem.states.entry(problem.states.names[0])
        values = state.full()
        if problem.pure:
            return entry.encode(values.ravel())
        if values.shape[1] == 1:
            values = np.outer(values, values.conj())
        return entry.encode(values)

    @property
    def _options(self) -> dict[str, Any]:
        """Options of the QuTiP solvers."""
        return {'atol': self.atol, 'rtol': self.rtol, 'nsteps': self.nsteps}

    def _create(self, p: Array) -> Any:  # noqa: ANN401
        """Return the QuTiP solver at fixed parameters.

        Args:
            p:
                The parameter vector.

        Returns:
            solver:
                The QuTiP solver of the subclass.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def _start(self, solver: Any, x: Array, t: float) -> None:  # noqa: ANN401
        """Start a QuTiP solver from a state at a time."""
        solver.start(self._start_state(x), t)

    def _unsupported(self, problem: QuantumProblem) -> list[str]:
        """Return a description of every part of a quantum problem the solver does not run."""
        del problem
        return []

    def _prepare(self, problem: QuantumProblem) -> None:
        """Check that the problem is a quantum problem the solver runs, and create its QuTiP solver at the first frame.

        Args:
            problem:
                The compiled problem.

        Raises:
            SolverError:
                The problem is not quantum, or has dissipators or a density matrix the solver does not run.
        """
        super()._prepare(problem)
        unsupported = self._unsupported(problem) if isinstance(problem, QuantumProblem) else ['a classical problem']
        if unsupported:
            error_msg = f'{type(self).__name__} cannot run this system. Unsupported: {"; ".join(unsupported)}.'
            raise SolverError(error_msg)
        self.quantum_: QuantumProblem = problem
        self.solver_: Any = None
        self.parameters_: Array | None = None

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Advance the state from time `t` to time `until` with the QuTiP solver.

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
        if self.solver_ is None or self.parameters_ is None or not np.array_equal(p, self.parameters_):
            self.solver_ = self._create(p)
            self.parameters_ = p.copy()
            self._start(self.solver_, x, t)
        return self._stored(self.solver_.step(until))

    @staticmethod
    def _coefficient(term: QuantumKernel, p: Array) -> Callable[[float], complex]:
        """Return the coefficient of a term as a function of time, at fixed parameters."""
        parameters = p.copy()

        def value(t: float) -> complex:
            return term.run(State(t, np.zeros(0), parameters))

        return value
