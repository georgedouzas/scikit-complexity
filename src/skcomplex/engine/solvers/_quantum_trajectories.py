"""Solver that samples the master equation by quantum jumps of a pure state with QuTiP's `MCSolver`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import qutip

from ...core.errors import SolverError
from ...core.types import Vector
from ._quantum import QuantumSolver

MAX_SEED = 2**32


_PURITY_TOLERANCE = 1e-8


class QuantumTrajectoriesSolver(QuantumSolver):
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
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from sage.all__sagemath_symbolics import CC, SR
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.laws import DissipatorLaw, GeneratorLaw
        >>> from skcomplex.core.laws.utils import local_operator
        >>> from skcomplex.core.specs import HilbertSpec, ObservableSpec, StateSpec
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.engine.solvers import QuantumTrajectoriesSolver
        >>> operators = {'x': (np.array([[0, 1], [1, 0]]), True), 'z': (np.diag([1, -1]), True)}
        >>> operators['minus'] = (np.array([[0, 0], [1, 0]]), False)
        >>> QUBIT = HilbertSpec(2, operators, basis={'up': 0, 'down': 1})
        >>> class QubitElement(Element):
        ...     def __init__(self, label, psi: Annotated[object, StateSpec(domain=CC**2)] = 'up', hilbert=QUBIT):
        ...         super().__init__(label)
        ...         self.psi = psi
        ...         self.hilbert = hilbert
        ...
        ...     def _get_laws(self, s):
        ...         return [DissipatorLaw(local_operator(s.psi, SR.symbol('minus')), 1.0)]
        >>> z = [ObservableSpec('z', element='q', variable='z')]
        >>> system = System(elements=[QubitElement('q')])
        >>> solver = QuantumTrajectoriesSolver()
        >>> simulation = Simulation(
        ...     system, solver=solver, dt=1.0, random_state=0, observables=z,
        ... ).check().run(until=20.0)
        >>> float(simulation.trajectory_.values('q.z')[-1])
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
        state = problem._state_of(x)
        if problem.pure:
            return self._ket(state)
        weights, vectors = np.linalg.eigh(state)
        if abs(weights[-1] - 1) > _PURITY_TOLERANCE:
            error_msg = (
                f'Quantum trajectories start from a pure state, but the state has purity {np.sum(weights**2):.6g}. '
                'Use the lindblad solver for mixed states.'
            )
            raise SolverError(error_msg)
        return self._ket(vectors[:, -1])

    def _start(self, solver: Any, x: Vector, t: float) -> None:  # noqa: ANN401
        """Start the Monte Carlo solver from a state at a time with a seed drawn from the random number generator."""
        solver.start(self._start_state(x), t, seed=int(self.rng_.integers(MAX_SEED)))
