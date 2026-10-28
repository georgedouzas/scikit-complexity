"""Solver of closed quantum systems that evolves a pure state by the Schrödinger equation with QuTiP's `SESolver`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import qutip

from ...core.types import Vector
from ..compile.passes._quantum_problem import QuantumProblem
from ._quantum import QuantumSolver


class UnitarySolver(QuantumSolver):
    """Solver of closed quantum systems that evolves a pure state by the Schrödinger equation with QuTiP's `SESolver`.

    Args:
        atol:
            The absolute tolerance of the integrator.

        rtol:
            The relative tolerance of the integrator.

        nsteps:
            The largest number of internal steps between two frames.

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
        >>> from skcomplex.engine.solvers import UnitarySolver
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
        ...         return [GeneratorLaw(local_operator(s.psi, SR.symbol('x')) / 2)]
        >>> z = [ObservableSpec('z', element='q', variable='z')]
        >>> system = System(elements=[QubitElement('q')])
        >>> simulation = Simulation(system, solver=UnitarySolver(), dt=np.pi, observables=z)
        >>> float(round(simulation.run(until=np.pi).trajectory_.values('q.z')[-1], 6))
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
