"""Solver that samples the master equation by jumps of a pure state.

It uses QuTiP's `MCSolver`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import qutip
from qutip.solver import solver_base

from ...errors import SolverError
from ...types import Array
from ._operator import OperatorSolver

MAX_SEED = 2**32


_PURITY_TOLERANCE = 1e-8


class JumpTrajectoriesSolver(OperatorSolver):
    """Solver that samples the master equation by jumps of a pure state.

    It uses QuTiP's `MCSolver`. Every run is one trajectory, whose jumps are drawn from the simulation's random number
    generator, so a seeded run is reproducible. A problem stored as a density matrix records the projector on the pure
    state of every frame.

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
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.laws import DissipatorLaw, GeneratorLaw
        >>> from skcomplex.core.components.laws.utils import local_operator
        >>> from skcomplex.core.specs import HilbertSpec, ObservableSpec, StateSpec
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.engine.solvers import JumpTrajectoriesSolver
        >>> operators = {'x': (np.array([[0, 1], [1, 0]]), True), 'z': (np.diag([1, -1]), True)}
        >>> operators['minus'] = (np.array([[0, 0], [1, 0]]), False)
        >>> QUBIT = HilbertSpec(2, operators, basis={'up': 0, 'down': 1})
        >>> class TwoLevelElement(Element):
        ...     def __init__(self, label, psi: Annotated[object, StateSpec(domain=CC**2)] = 'up', hilbert=QUBIT):
        ...         super().__init__(label)
        ...         self.psi = psi
        ...         self.hilbert = hilbert
        ...
        ...     def _get_laws(self, s):
        ...         return [DissipatorLaw(local_operator(s.psi, SR.symbol('minus')), 1.0)]
        >>> z = [ObservableSpec('z', element='q', variable='z')]
        >>> system = System(elements=[TwoLevelElement('q')])
        >>> solver = JumpTrajectoriesSolver()
        >>> trajectory = simulate(system, until=20.0, solver=solver, dt=1.0, random_state=0, observables=z)
        >>> float(trajectory.values('q.z')[-1])
        -1.0
    """

    def _create(self, p: Array) -> solver_base.Solver:
        """Return QuTiP's Monte Carlo solver at fixed parameters."""
        problem = self.operator_
        generator = self._operator(problem.generator, p)
        collapses = [self._operator(terms, p) for terms in problem.dissipators]
        return qutip.MCSolver(generator, c_ops=collapses, options=self._options)

    def _start_state(self, x: Array) -> qutip.Qobj:
        """Return the pure state a trajectory starts from.

        It is the eigenvector of a pure density matrix.
        """
        problem = self.operator_
        state = problem._state_of(x)
        if problem.pure:
            return self._ket(state)
        weights, vectors = np.linalg.eigh(state)
        if abs(weights[-1] - 1) > _PURITY_TOLERANCE:
            error_msg = (
                f'Operator trajectories start from a pure state, but the state has purity {np.sum(weights**2):.6g}. '
                'Use the lindblad solver for mixed states.'
            )
            raise SolverError(error_msg)
        return self._ket(vectors[:, -1])

    def _start(self, solver: solver_base.Solver, x: Array, t: float) -> None:
        """Start the Monte Carlo solver from a state at a time.

        Its seed is drawn from the random number generator.
        """
        solver.start(self._start_state(x), t, seed=int(self.rng_.integers(MAX_SEED)))
