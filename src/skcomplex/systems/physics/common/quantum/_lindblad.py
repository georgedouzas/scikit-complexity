"""Solver of the Lindblad master equation with QuTiP's `MESolver`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import qutip
from qutip.solver import solver_base

from .....core.types import Array
from ._operator import OperatorSolver


class LindbladSolver(OperatorSolver):
    """A solver of the Lindblad master equation with the `MESolver` of QuTiP.

    Pure closed systems are solved with `SESolver`. It is the default solver of quantum systems. A system without
    dissipators that starts in a pure state follows the linear evolution `dx/dt = -i H x`, which is the master equation
    without dissipators.

    Choose it for open quantum systems small enough to hold the density matrix. It appears in quantum optics and in the
    decoherence of qubits.

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
        >>> from skcomplex.systems.physics.common.quantum import DissipatorLaw, LindbladSolver, QubitElement
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
        >>> decay = Decay('decay', rate=0.5)
        >>> system = System(elements=[QubitElement('q')], interactions=[decay], solver=LindbladSolver()).prepare(z)
        >>> values = system.simulate(until=2.0, dt=1.0, observables=z).get_values('q.z')
        >>> bool(abs((1 + values[-1]) / 2 - np.exp(-1.0)) < 1e-6)
        True
    """

    def _create(self, p: Array) -> solver_base.Solver:
        """Return QuTiP's master equation solver.

        A pure closed problem gets its Schrödinger solver.
        """
        problem = self.operator_
        generator = self._get_operator(problem.generator, p)
        if problem.pure:
            return qutip.SESolver(generator, options=self._options)
        collapses = [self._get_operator(terms, p) for terms in problem.dissipators]
        return qutip.MESolver(generator, c_ops=collapses, options=self._options)
