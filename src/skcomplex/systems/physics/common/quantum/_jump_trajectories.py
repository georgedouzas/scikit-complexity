"""Solver that samples the master equation by jumps of a pure state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import qutip
from qutip.solver import solver_base

from .....core.errors import SolverError
from .....core.types import Array
from ._operator import OperatorSolver

MAX_SEED = 2**32
PURITY_TOLERANCE = 1e-8


class JumpTrajectoriesSolver(OperatorSolver):
    """A solver that samples the master equation by random jumps of a pure state.

    It uses QuTiP's `MCSolver`. Every run is one trajectory, whose jumps are drawn from the simulation's random number
    generator, so a seeded run is reproducible. A problem stored as a density matrix records the projector on the pure
    state of every frame.

    Choose it for large open quantum systems, where averaging pure states costs less than the density matrix. It appears
    in quantum optics, such as the photon counts of a cavity.

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
        >>> import numpy as np
        >>> from sage.all__sagemath_symbolics import SR
        >>> from skcomplex.core.components import Observable
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws.utils import local_operator
        >>> from skcomplex.core.specs import Role
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.quantum import DissipatorLaw, JumpTrajectoriesSolver, QubitElement
        >>> class Decay(Interaction):
        ...     def __init__(self, label, *, target: Role = 'q', rate=1.0):
        ...         super().__init__(label)
        ...         self.target = target
        ...         self.rate = rate
        ...
        ...     def _prepare(self, s):
        ...         super()._prepare(s)
        ...         self.laws_ = [DissipatorLaw(local_operator(s.target.psi, SR.symbol('minus')), self.rate)]
        >>> z = [Observable('z', element='q', variable='z')]
        >>> solver = JumpTrajectoriesSolver()
        >>> system = System(elements=[QubitElement('q')], interactions=[Decay('decay')], solver=solver).prepare(z)
        >>> float(system.simulate(until=20.0, dt=1.0, random_state=0, observables=z).get_values('q.z')[-1])
        -1.0
    """

    def _create(self, p: Array) -> solver_base.Solver:
        """Return QuTiP's Monte Carlo solver at fixed parameters."""
        problem = self.operator_
        generator = self._get_operator(problem.generator, p)
        collapses = [self._get_operator(terms, p) for terms in problem.dissipators]
        return qutip.MCSolver(generator, c_ops=collapses, options=self._options)

    def _start_state(self, x: Array) -> qutip.Qobj:
        """Return the pure state a trajectory starts from.

        It is the eigenvector of a pure density matrix.
        """
        problem = self.operator_
        state = problem._get_state_of(x)
        if problem.pure:
            return self._get_ket(state)
        weights, vectors = np.linalg.eigh(state)
        if abs(weights[-1] - 1) > PURITY_TOLERANCE:
            error_msg = (
                f'Operator trajectories start from a pure state, but the state has purity {np.sum(weights**2):.6g}. '
                'Use the lindblad solver for mixed states.'
            )
            raise SolverError(error_msg)
        return self._get_ket(vectors[:, -1])

    def _start(self, solver: solver_base.Solver, x: Array, t: float) -> None:
        """Start the Monte Carlo solver from a state at a time.

        Its seed is drawn from the random number generator.
        """
        solver.start(self._start_state(x), t, seed=int(self.rng_.integers(MAX_SEED)))
