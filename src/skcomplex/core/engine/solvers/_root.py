"""Solve the equations of a static system once.

They are solved by the compiled solution or numerically from the initial state.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from scipy.optimize import root

from ...components.laws import EquationLaw, LinearProgramLaw
from ...components.spaces import CayleySpace, ContinuousSpace, LatticeSpace, MeshSpace, NetworkSpace, StaticTime
from ...errors import SolverError
from ...specs import RunStateSpec, SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems import ProblemSpec
from ..problems._jacobian_kernel import JacobianKernel
from ._solver import Solver


class RootSolver(Solver):
    """Solver of the equations of a static system.

    It solves them once for the one frame of the run. The compiled solution gives the states when Sage finds a unique
    explicit one. Otherwise `scipy.optimize.root` solves the residuals of the equations with their exact Jacobian, from
    the initial state as the guess. A system without equations solves to its initial state.

    Args:
        method:
            The method of `scipy.optimize.root`.

        tolerance:
            The largest residual of a solution.

        random_state:
            The seed or generator of the solver, which draws nothing.

        until:
            The end time of the run the solver prepares for, which a static system does not read.

    Attributes:
        initial_state_:
            The solved state vector, the one frame of the run.

    Examples:
        >>> from sage.all__sagemath_symbolics import cos
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.laws import EquationLaw
        >>> from skcomplex.core.components.spaces import StaticTime
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class FixedPointElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.x = x
        ...
        ...     def _get_laws(self, s):
        ...         return [EquationLaw(s.x == cos(s.x), target=s.x)]
        >>> system = CompositeSystem(elements=[FixedPointElement('a')], time=StaticTime())
        >>> trajectory = simulate(system, until=1.0)
        >>> round(float(trajectory.values('a.x')[0]), 7)
        0.7390851
    """

    def __init__(
        self,
        *,
        method: str = 'hybr',
        tolerance: float = 1e-9,
        random_state: int | np.random.Generator | None = None,
        until: float = np.inf,
    ) -> None:
        super().__init__(random_state=random_state, until=until)
        self.method = method
        self.tolerance = tolerance

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={StaticTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={LinearProgramLaw, EquationLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.STATIC

    def _residuals(self, unknowns: Array, x: Array, p: Array) -> Array:
        """Return the residuals of the equations at values of their unknowns."""
        state = x.copy()
        state[self.problem_.unknowns] = unknowns
        return np.asarray(self._drift()(0.0, state, p), dtype=float)[self.problem_.unknowns]

    def _kernel(self) -> JacobianKernel:
        """Return the kernel of the exact Jacobian of the residuals.

        An error is raised for a problem without one.
        """
        kernel = self.problem_.jacobian
        if kernel is None:
            error_msg = f'{type(self).__name__} solves with the exact Jacobian, but the problem has none.'
            raise SolverError(error_msg)
        return kernel

    def _jacobian(self, unknowns: Array, x: Array, p: Array) -> Array:
        """Return the exact Jacobian of the residuals in their unknowns."""
        indices = self.problem_.unknowns
        state = x.copy()
        state[indices] = unknowns
        matrix = self._kernel().run(RunStateSpec(0.0, state, p)).values_
        return np.asarray(matrix.toarray(), dtype=float)[np.ix_(indices, indices)]

    def _solved(self, x: Array, p: Array) -> Array:
        """Return the state vector that solves the equations.

        It starts from an initial state and parameters.
        """
        problem = self.problem_
        indices = problem.unknowns
        if not indices.size:
            return np.asarray(problem.update(0.0, x, p), dtype=float) if problem.update is not None else x
        self._kernel().prepare(problem)
        result = root(
            self._residuals,
            x[indices],
            args=(x, p),
            jac=self._jacobian,
            method=self.method,
            tol=self.tolerance,
        )
        residual = float(np.max(np.abs(self._residuals(result.x, x, p))))
        if not result.success or not residual <= self.tolerance:
            equations = ', '.join(f'{name}: {value} == 0' for name, value in problem.equations.items() if value != 0)
            error_msg = (
                f'The numerical solve of the equations {equations} did not converge: the largest residual is '
                f'{residual:.6g}, above the tolerance {self.tolerance:g}. Give an initial state closer to a real '
                'solution, or check that one exists.'
            )
            raise SolverError(error_msg)
        solution = x.copy()
        solution[indices] = result.x
        return solution

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the solver for a problem and solve its equations.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem has another form, or its numerical solve does not converge.
        """
        super()._prepare(context)
        self.initial_state_ = self._solved(self.problem_.initial_state, self.problem_.initial_parameters)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Return the state unchanged.

        The state was solved for the one frame of the run.

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
                The state vector.
        """
        return x
