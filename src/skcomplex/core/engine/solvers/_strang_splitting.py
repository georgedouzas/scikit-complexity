"""Alternate transport and collision steps of a field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from dataclasses import replace

import numpy as np
from sklearn.base import clone

from ...components.laws import BaseLaw, RateLaw
from ...components.spaces import LatticeSpace, PhaseSpace
from ...components.spaces._continuous_time import _ContinuousTime
from ...enums import FormKind, OrderKind
from ...errors import SolverError
from ...types import Array
from ..backends import SupportSpec
from ..problems import ProblemSpec, RunStateSpec
from ..problems._bilinear_law import BilinearLaw
from ..problems._steps import step_count
from ._rk4 import RK4Solver
from ._solver import Solver


class StrangSplittingSolver(Solver):
    """A solver that puts a full collision step between two half transport steps.

    Every step takes half a step of transport, a full step of collisions, and another half step of transport, which is
    of order 2. The transport solver moves the flux differences of the finite volume fields, such as a gas streaming
    along its positions, and the collision solver the source rates, such as its relaxation towards equilibrium, which
    the problem keeps apart. Choose it for kinetic equations with collisions, in models of rarefied gases, plasmas, and
    traffic.

    Read more in the user guide, under Solvers.

    Args:
        transport:
            The solver of the flux differences, or `None` for `RK4Solver`.

        collision:
            The solver of the source rates, or `None` for `RK4Solver`.

        dt:
            The largest internal time step, the length of one split step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Attributes:
        transport_:
            The prepared solver of the flux differences.

        collision_:
            The prepared solver of the source rates.

    See Also:
        SemiLagrangianSolver: Moves transport with any step.
        SSPRKSolver: Moves transport with shocks.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BGKEquationSystem
        >>> solver = BGKEquationSystem(positions=8, velocities=16).prepare().system_.solver_
        >>> type(solver).__name__, type(solver.transport).__name__, type(solver.collision).__name__
        ('StrangSplittingSolver', 'SSPRKSolver', 'RK4Solver')
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_ContinuousTime},
            space={PhaseSpace, LatticeSpace},
            laws={RateLaw, BilinearLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

    @classmethod
    def _is_implicit(cls) -> bool:
        """Return whether explicit stability limits do not apply.

        The inner solvers answer for it.
        """
        return True

    def __init__(
        self,
        transport: Solver | None = None,
        collision: Solver | None = None,
        dt: float = 0.01,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.transport = transport
        self.collision = collision
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _get_flux_differences(self, t: float, x: Array, p: Array) -> Array:
        """Return the rates of the flux differences of the finite-volume fields.

        They are zero for the other states.
        """
        rates = np.zeros(x.size)
        for volume in self.problem_.volumes:
            rates = rates + np.asarray(volume.run(RunStateSpec(t, x, p)).values_, dtype=float)
        return rates

    def _get_inner(self, solver: Solver | None, law: type[BaseLaw], dt: float, problem: ProblemSpec) -> Solver:
        """Return an inner solver prepared on a part of the problem.

        An error is raised for a law it does not advance.
        """
        inner = RK4Solver(dt=dt) if solver is None else clone(solver)
        if inner._get_support().get_missing('laws', {law}):
            error_msg = f'StrangSplittingSolver advances {law._get_name()} with {type(inner).__name__}, which lacks it.'
            raise SolverError(error_msg)
        return inner.prepare(problem)

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the two inner solvers on the transport and the collision parts.

        These are the parts of the problem.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem has no flux or no source rates, or an inner solver does not advance its part of the problem.
        """
        super()._prepare(context)
        problem = self.problem_
        if not problem.volumes or problem.sources is None:
            error_msg = 'StrangSplittingSolver splits flux differences from source rates, but the problem has no flux.'
            raise SolverError(error_msg)
        plain = replace(problem, accelerated=None, jacobian=None, sources=None)
        transport = replace(plain, drift=self._get_flux_differences)
        collision = replace(plain, drift=problem.sources, volumes=(), linear=())
        self.transport_ = self._get_inner(self.transport, RateLaw, self.dt / 2, transport)
        self.collision_ = self._get_inner(self.collision, RateLaw, self.dt, collision)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Advance from time `t` to time `until` in equal split steps no longer than `dt`.

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
                The state vector at time `until`.
        """
        steps = step_count(t, until, self.dt)
        h = (until - t) / steps
        state = x
        for index in range(steps):
            time = t + index * h
            state = self.transport_._advance(time, state, p, time + h / 2)
            state = self.collision_._advance(time, state, p, time + h)
            state = self.transport_._advance(time + h / 2, state, p, time + h)
        return state
