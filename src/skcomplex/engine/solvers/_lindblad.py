"""Solver of the Lindblad master equation with QuTiP's `MESolver`, and of pure closed systems with `SESolver`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import qutip

from ...types import Array
from ._quantum import QuantumSolver


class LindbladSolver(QuantumSolver):
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
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from sage.all__sagemath_symbolics import CC, SR
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.laws import DissipatorLaw, GeneratorLaw
        >>> from skcomplex.core.laws.utils import local_operator
        >>> from skcomplex.specs import HilbertSpec, ObservableSpec, StateSpec
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.engine.solvers import LindbladSolver
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
        ...         return [DissipatorLaw(local_operator(s.psi, SR.symbol('minus')), 0.5)]
        >>> z = [ObservableSpec('z', element='q', variable='z')]
        >>> system = System(elements=[QubitElement('q')])
        >>> from skcomplex.engine.compile import QuantumCompiler
        >>> compiler = QuantumCompiler()
        >>> simulation = Simulation(system, compiler=compiler, solver=LindbladSolver(), dt=1.0, observables=z).check()
        >>> values = simulation.run(until=2.0).trajectory_.values('q.z')
        >>> bool(abs((1 + values[-1]) / 2 - np.exp(-1.0)) < 1e-6)
        True
    """

    def _create(self, p: Array) -> Any:  # noqa: ANN401
        """Return QuTiP's master equation solver, or its Schrödinger solver for a pure closed problem."""
        problem = self.quantum_
        hamiltonian = self._operator(problem.hamiltonian, p)
        if problem.pure:
            return qutip.SESolver(hamiltonian, options=self._options)
        collapses = [self._operator(terms, p) for terms in problem.dissipators]
        return qutip.MESolver(hamiltonian, c_ops=collapses, options=self._options)
