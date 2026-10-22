"""Advance quantum problems with QuTiP by unitary, Lindblad, and quantum trajectory evolution."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp

from ..base import ComponentKind, DomainKind, Form, LawKind, OrderKind, SolverError, SpaceKind, Supports, TimeKind
from ..compile import DEFAULT_MAX_DIMENSION, Problem, QuantumProblem, QuantumTerm, Vector
from ._base import Stepper, register_solver

MAX_SEED = 2**32
PURITY_TOLERANCE = 1e-8


def _coefficient(term: QuantumTerm, p: Vector) -> Callable[[float], complex]:
    """Return the coefficient of a term as a function of time, at fixed parameters."""
    parameters = p.copy()

    def value(t: float) -> complex:
        return term.value(t, parameters)

    return value


class QuantumSolver(Stepper):
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

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.CONTINUOUS},
        space=set(SpaceKind),
        laws={LawKind.GENERATOR, LawKind.DISSIPATOR},
        domains={DomainKind.HILBERT},
        components={ComponentKind.ELEMENT, ComponentKind.GROUP, ComponentKind.SITE_GROUP},
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.QUANTUM

    def __init__(
        self,
        atol: float = 1e-12,
        rtol: float = 1e-10,
        nsteps: int = 100_000,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.nsteps = nsteps
        self.max_dimension = max_dimension

    def _operator(self, terms: Sequence[QuantumTerm], p: Vector) -> qutip.Qobj | qutip.QobjEvo:
        """Return the QuTiP operator of terms, constant or as a function of time, at fixed parameters."""
        problem = self.quantum_
        dims = [list(problem.dimensions)] * 2
        constant = qutip.Qobj(sp.csr_array((problem.dimension, problem.dimension), dtype=complex), dims=dims)
        varying: list[Any] = []
        for term in terms:
            matrix = qutip.Qobj(term.matrix, dims=dims)
            if term.time_dependent:
                varying.append([matrix, _coefficient(term, p)])
            else:
                constant = constant + complex(term.value(0.0, p)) * matrix
        return qutip.QobjEvo([constant, *varying]) if varying else constant

    def _ket(self, vector: npt.NDArray[np.complex128]) -> qutip.Qobj:
        """Return a state vector as a QuTiP ket with the dimensions of the problem."""
        dimensions = list(self.quantum_.dimensions)
        return qutip.Qobj(vector.reshape(-1, 1), dims=[dimensions, [1] * len(dimensions)])

    def _start_state(self, x: Vector) -> qutip.Qobj:
        """Return the QuTiP state a solver starts from."""
        problem = self.quantum_
        state = problem.state_of(x)
        if problem.pure:
            return self._ket(state)
        return qutip.Qobj(state, dims=[list(problem.dimensions)] * 2)

    def _stored(self, state: qutip.Qobj) -> Vector:
        """Return a QuTiP state as a state vector of the problem."""
        problem = self.quantum_
        values = state.full()
        if problem.pure:
            return problem.encode(values.ravel())
        if values.shape[1] == 1:
            values = np.outer(values, values.conj())
        return problem.encode(values)

    @property
    def _options(self) -> dict[str, Any]:
        """Options of the QuTiP solvers."""
        return {'atol': self.atol, 'rtol': self.rtol, 'nsteps': self.nsteps}

    def _create(self, p: Vector) -> Any:  # noqa: ANN401
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

    def _start(self, solver: Any, x: Vector, t: float) -> None:  # noqa: ANN401
        """Start a QuTiP solver from a state at a time."""
        solver.start(self._start_state(x), t)

    def _unsupported(self, problem: QuantumProblem) -> list[str]:
        """Return a description of every part of a quantum problem the solver does not run."""
        del problem
        return []

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Check that the problem is a quantum problem the solver runs, and create its QuTiP solver at the first frame.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator that seeds quantum jumps.

        Raises:
            SolverError:
                The problem is not quantum, or has dissipators or a density matrix the solver does not run.
        """
        super().initialize(problem, rng)
        unsupported = self._unsupported(problem) if isinstance(problem, QuantumProblem) else ['a classical problem']
        if unsupported:
            error_msg = f'{type(self).__name__} cannot run this system. Unsupported: {"; ".join(unsupported)}.'
            raise SolverError(error_msg)
        self.quantum_: QuantumProblem = problem  # type: ignore[assignment]
        self.solver_: Any = None
        self.parameters_: Vector | None = None

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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


class UnitaryEvolution(QuantumSolver):
    """Solver of closed quantum systems that evolves a pure state by the Schrödinger equation with QuTiP's `SESolver`.

    Args:
        atol:
            The absolute tolerance of the integrator.

        rtol:
            The relative tolerance of the integrator.

        nsteps:
            The largest number of internal steps between two frames.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.solvers import UnitaryEvolution
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.core.laws import Generator
        >>> from skcomplex.core.system import System, expectation
        >>> from skcomplex.models.physics.realizations import Qubit
        >>> from operator import attrgetter
        >>> Rabi = Qubit.define('Rabi', laws=lambda s: [Generator(s.x / 2)])
        >>> system = System(components=[Rabi('q')])
        >>> z = [expectation('z', attrgetter('q.z'))]
        >>> simulation = Simulation(system, solver=UnitaryEvolution(), dt=np.pi, observables=z).run(until=np.pi)
        >>> float(round(simulation.trajectory_['z'][-1], 6))
        -1.0
    """

    def _unsupported(self, problem: QuantumProblem) -> list[str]:
        """Return the dissipators or the density matrix of a problem, which unitary evolution does not run."""
        if problem.dissipators:
            return ['dissipators, which need the lindblad or quantum_trajectories solver']
        return [] if problem.pure else ['a density matrix, which needs the lindblad solver']

    def _create(self, p: Vector) -> Any:  # noqa: ANN401
        """Return QuTiP's Schrödinger solver at fixed parameters."""
        return qutip.SESolver(self._operator(self.quantum_.hamiltonian, p), options=self._options)


class LindbladEvolution(QuantumSolver):
    """Solver of the Lindblad master equation with QuTiP's `MESolver`, and of pure closed systems with `SESolver`.

    It is the default solver of the quantum form. A system without dissipators that starts in a pure state follows the
    Schrödinger equation, which is the master equation without dissipators.

    Args:
        atol:
            The absolute tolerance of the integrator.

        rtol:
            The relative tolerance of the integrator.

        nsteps:
            The largest number of internal steps between two frames.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.solvers import LindbladEvolution
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.core.laws import Dissipator
        >>> from skcomplex.core.system import System, expectation
        >>> from skcomplex.models.physics.realizations import Qubit
        >>> from operator import attrgetter
        >>> Decaying = Qubit.define('Decaying', laws=lambda s: [Dissipator(s.minus, 0.5)])
        >>> system = System(components=[Decaying('q')])
        >>> z = [expectation('z', attrgetter('q.z'))]
        >>> simulation = Simulation(system, solver=LindbladEvolution(), dt=1.0, observables=z).run(until=2.0)
        >>> bool(abs((1 + simulation.trajectory_['z'][-1]) / 2 - np.exp(-1.0)) < 1e-6)
        True
    """

    def _create(self, p: Vector) -> Any:  # noqa: ANN401
        """Return QuTiP's master equation solver, or its Schrödinger solver for a pure closed problem."""
        problem = self.quantum_
        hamiltonian = self._operator(problem.hamiltonian, p)
        if problem.pure:
            return qutip.SESolver(hamiltonian, options=self._options)
        collapses = [self._operator(terms, p) for terms in problem.dissipators]
        return qutip.MESolver(hamiltonian, c_ops=collapses, options=self._options)


class QuantumTrajectories(QuantumSolver):
    """Solver that samples the master equation by quantum jumps of a pure state with QuTiP's `MCSolver`.

    Every run is one trajectory, whose jumps are drawn from the simulation's random number generator, so a seeded run
    is reproducible. A problem stored as a density matrix records the projector on the pure state of every frame.

    Args:
        atol:
            The absolute tolerance of the integrator.

        rtol:
            The relative tolerance of the integrator.

        nsteps:
            The largest number of internal steps between two frames.

    Raises:
        SolverError:
            The initial state is a mixed density matrix.

    Examples:
        >>> from skcomplex.core.solvers import QuantumTrajectories
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.core.laws import Dissipator
        >>> from skcomplex.core.system import System, expectation
        >>> from skcomplex.models.physics.realizations import Qubit
        >>> from operator import attrgetter
        >>> Decaying = Qubit.define('Decaying', laws=lambda s: [Dissipator(s.minus, 1.0)])
        >>> system = System(components=[Decaying('q')])
        >>> z = [expectation('z', attrgetter('q.z'))]
        >>> simulation = Simulation(system, solver=QuantumTrajectories(), dt=1.0, random_state=0, observables=z)
        >>> simulation = simulation.run(until=20.0)
        >>> float(simulation.trajectory_['z'][-1])
        -1.0
    """

    def _create(self, p: Vector) -> Any:  # noqa: ANN401
        """Return QuTiP's Monte Carlo solver at fixed parameters."""
        problem = self.quantum_
        hamiltonian = self._operator(problem.hamiltonian, p)
        collapses = [self._operator(terms, p) for terms in problem.dissipators]
        return qutip.MCSolver(hamiltonian, c_ops=collapses, options=self._options)

    def _start_state(self, x: Vector) -> qutip.Qobj:
        """Return the pure state a trajectory starts from, the eigenvector of a pure density matrix."""
        problem = self.quantum_
        state = problem.state_of(x)
        if problem.pure:
            return self._ket(state)
        weights, vectors = np.linalg.eigh(state)
        if abs(weights[-1] - 1) > PURITY_TOLERANCE:
            error_msg = (
                f'Quantum trajectories start from a pure state, but the state has purity {np.sum(weights**2):.6g}. '
                'Use the lindblad solver for mixed states.'
            )
            raise SolverError(error_msg)
        return self._ket(vectors[:, -1])

    def _start(self, solver: Any, x: Vector, t: float) -> None:  # noqa: ANN401
        """Start the Monte Carlo solver from a state at a time with a seed drawn from the random number generator."""
        solver.start(self._start_state(x), t, seed=int(self.rng_.integers(MAX_SEED)))


register_solver('unitary', UnitaryEvolution)
register_solver('lindblad', LindbladEvolution, default=True)
register_solver('quantum_trajectories', QuantumTrajectories)
