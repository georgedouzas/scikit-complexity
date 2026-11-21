"""Solve the equations of a static system once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from scipy.optimize import root

from ...components.laws import ConstraintLaw
from ...components.laws._linear_program import _LinearProgramLaw
from ...components.spaces import BaseSpace
from ...components.spaces._static_time import _StaticTime
from ...enums import FormKind, OrderKind
from ...errors import SolverError
from ...types import Array
from ..backends import SupportSpec
from ..problems import ProblemSpec, RunStateSpec
from ..problems._jacobian_kernel import JacobianKernel
from ._solver import Solver


class RootSolver(Solver):
    """A solver of the equations of a static system, solved once.

    It solves the constraint laws once for the one frame of the run, such as the outputs 3.33 and 2.33 at which two
    firms of a market both make their best reply. It takes the exact solution when Sage finds a unique one, and
    otherwise solves the equations with `scipy.optimize.root` and their exact Jacobian from the starting values. A
    system without equations keeps its starting values. It finds the equilibria of markets, the balance of forces in a
    truss, and the steady state of a reaction.

    Read more in the user guide, under Solvers.

    Args:
        method:
            The method of `scipy.optimize.root`, or `'auto'` for `'hybr'` with a second try by `'lm'` from the same
            starting values when it does not converge, as an exponential diode from zero volts needs.

        tolerance:
            The largest residual of a solution.

        random_state:
            The seed or generator of the solver, which draws nothing.

        until:
            The end time of the run the solver prepares for, which a static system does not read.

    Attributes:
        initial_state_:
            The solved state vector, the one frame of the run.

    See Also:
        MinimizeSolver: Finds the minimum of an objective.

    Examples:
        >>> from skcomplex.systems.economics.markets import OligopolyEquilibriaSystem
        >>> trajectory = OligopolyEquilibriaSystem().prepare().simulate()
        >>> [round(float(trajectory.get_values(name)[0]), 4) for name in ('cournot.q_i', 'cournot.q_j')]
        [3.3333, 2.3333]
    """

    def __init__(
        self,
        *,
        method: str = 'auto',
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
            time={_StaticTime},
            space={None, BaseSpace},
            laws={_LinearProgramLaw, ConstraintLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.STATIC

    def _get_residuals(self, unknowns: Array, x: Array, p: Array) -> Array:
        """Return the residuals of the equations at values of their unknowns."""
        state = x.copy()
        state[self.problem_.unknowns] = unknowns
        return np.asarray(self._get_drift()(0.0, state, p), dtype=float)[self.problem_.unknowns]

    def _get_kernel(self) -> JacobianKernel:
        """Return the kernel of the exact Jacobian of the residuals.

        An error is raised for a problem without one.
        """
        kernel = self.problem_.jacobian
        if kernel is None:
            error_msg = f'{type(self).__name__} solves with the exact Jacobian, but the problem has none.'
            raise SolverError(error_msg)
        return kernel

    def _get_jacobian(self, unknowns: Array, x: Array, p: Array) -> Array:
        """Return the exact Jacobian of the residuals in their unknowns."""
        indices = self.problem_.unknowns
        state = x.copy()
        state[indices] = unknowns
        matrix = self._get_kernel().run(RunStateSpec(0.0, state, p)).values_
        return np.asarray(matrix.toarray(), dtype=float)[np.ix_(indices, indices)]

    def _get_solved(self, x: Array, p: Array) -> Array:
        """Return the state vector that solves the equations.

        It starts from an initial state and parameters.
        """
        problem = self.problem_
        indices = problem.unknowns
        if not indices.size:
            return np.asarray(problem.update(0.0, x, p), dtype=float) if problem.update is not None else x
        self._get_kernel().prepare(problem)
        methods = ('hybr', 'lm') if self.method == 'auto' else (self.method,)
        for method in methods:
            result = root(
                self._get_residuals,
                x[indices],
                args=(x, p),
                jac=self._get_jacobian,
                method=method,
                tol=self.tolerance,
            )
            residual = float(np.max(np.abs(self._get_residuals(result.x, x, p))))
            if result.success and residual <= self.tolerance:
                break
        else:
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
        self.initial_state_ = self._get_solved(self.problem_.initial_state, self.problem_.initial_parameters)

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
