"""Solver of the linear evolution `dx/dt = -i H x` of a pure state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import qutip
from qutip.solver import solver_base

from .....core.engine.problems import OperatorProblemSpec
from .....core.types import Array
from ._operator import OperatorSolver


class UnitarySolver(OperatorSolver):
    """A solver of the linear evolution `dx/dt = -i H x` of a pure state.

    It uses QuTiP's `SESolver`.

    Choose it for closed quantum systems. It appears in quantum mechanics, such as spin chains and quantum walks.

    Args:
        atol:
            The absolute tolerance of the integrator.

        rtol:
            The relative tolerance of the integrator.

        nsteps:
            The largest number of internal steps between two frames.

    Examples:
        >>> import numpy as np
        >>> from sage.all__sagemath_symbolics import SR
        >>> from skcomplex.core.components import Observable
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws.utils import local_operator
        >>> from skcomplex.core.types import Role
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.quantum import GeneratorLaw, QubitElement, UnitarySolver
        >>> class Flip(Interaction):
        ...     def __init__(self, label, *, target: Role = 'q'):
        ...         super().__init__(label)
        ...         self.target = target
        ...
        ...     def _prepare(self, s):
        ...         super()._prepare(s)
        ...         self.laws_ = [GeneratorLaw(local_operator(s.target.psi, SR.symbol('x')) / 2)]
        >>> z = [Observable('z', element='q', variable='z')]
        >>> system = System(elements=[QubitElement('q')], interactions=[Flip('flip')], solver=UnitarySolver())
        >>> trajectory = system.prepare(z).simulate(until=np.pi, dt=np.pi, observables=z)
        >>> float(round(trajectory.get_values('q.z')[-1], 6))
        -1.0
    """

    def _get_unsupported(self, problem: OperatorProblemSpec) -> list[str]:
        """Return the dissipators or the density matrix of a problem.

        Unitary evolution does not run them.
        """
        if problem.dissipators:
            return ['dissipators, which need the lindblad or jump_trajectories solver']
        return [] if problem.pure else ['a density matrix, which needs the lindblad solver']

    def _create(self, p: Array) -> solver_base.Solver:
        """Return QuTiP's Schrödinger solver at fixed parameters."""
        return qutip.SESolver(self._get_operator(self.operator_.generator, p), options=self._options)
