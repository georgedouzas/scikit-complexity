"""Solver of the linear evolution `dx/dt = -i H x` of a pure state.

It uses QuTiP's `SESolver`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import qutip
from qutip.solver import solver_base

from ...types import Array
from ..problems._operator_problem_spec import OperatorProblemSpec
from ._operator import OperatorSolver


class UnitarySolver(OperatorSolver):
    """Solver of the linear evolution `dx/dt = -i H x` of a pure state.

    It uses QuTiP's `SESolver`.

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
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.laws import DissipatorLaw, GeneratorLaw
        >>> from skcomplex.core.components.laws.utils import local_operator
        >>> from skcomplex.core.specs import HilbertSpec, ObservableSpec, StateSpec
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.engine.solvers import UnitarySolver
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
        ...         return [GeneratorLaw(local_operator(s.psi, SR.symbol('x')) / 2)]
        >>> z = [ObservableSpec('z', element='q', variable='z')]
        >>> system = CompositeSystem(elements=[TwoLevelElement('q')])
        >>> trajectory = simulate(system, until=np.pi, solver=UnitarySolver(), dt=np.pi, observables=z)
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
