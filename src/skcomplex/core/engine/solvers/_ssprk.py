"""Strong-stability-preserving Runge-Kutta of order 2 or 3."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ...components.laws import RateLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.spaces import BaseSpace
from ...components.spaces._continuous_time import _ContinuousTime
from ...enums import FormKind, OrderKind
from ...errors import SolverError
from ...types import Array
from ..backends import SupportSpec
from ..problems import ProblemSpec
from ._solver import Solver

SECOND_ORDER = 2


class SSPRKSolver(Solver):
    """A solver by strong stability preserving Runge-Kutta steps of order 2 or 3.

    Each step of the schemes of Shu and Osher is an average of Euler steps, so a field with shocks, such as a traffic
    jam, gains no new wiggles and keeps its total. With finite volume fields the step is the smaller of `cfl * h / s`,
    for the cell size `h` and the largest wave speed `s`, and the stability limit of the diffusion, and otherwise `dt`.
    Choose it for conservation laws with shocks, in models of traffic, gas flows, and shallow water.

    Read more in the user guide, under Solvers.

    Args:
        order:
            2 or 3.

        cfl:
            The Courant number.

        dt:
            The largest step, and the step without conservation laws.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    See Also:
        RK4Solver: Steps smooth flows with the classical Runge-Kutta rule.

    Examples:
        >>> from skcomplex.core.engine.solvers import SSPRKSolver
        >>> from skcomplex.systems.physics.fluids import BurgersSystem
        >>> u = BurgersSystem(64, solver=SSPRKSolver()).prepare().simulate(until=1, dt=1).get_values('line.u')
        >>> round(float(u[0].sum()), 6), round(float(u[-1].sum()), 6)
        (96.0, 96.0)
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_ContinuousTime},
            space={None, BaseSpace},
            laws={RateLaw, _BoundaryLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

    def __init__(
        self,
        order: int = 3,
        cfl: float = 0.4,
        dt: float = 0.01,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.order = order
        self.cfl = cfl
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the stepper and check its order.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The order is not 2 or 3, or the problem is not an ODE.
        """
        super()._prepare(context)
        problem = self.problem_
        if self.order not in {2, 3}:
            error_msg = f'SSPRKSolver has order 2 or 3, not {self.order}.'
            raise SolverError(error_msg)
        self.spacing_ = min((min(volume.spacings) for volume in problem.volumes), default=None)

    def _step_size(self, x: Array, p: Array) -> float:
        """Return the step allowed by the CFL condition and the diffusion, at most `dt`.

        The diffusion is that of finite-volume fields.
        """
        problem = self.problem_
        step = min([self.dt, *(limit.step(p) for limit in problem.diffusive_limits)])
        speed = max((volume.get_max_speed(x, p) for volume in problem.volumes), default=None)
        if speed is None or self.spacing_ is None or speed <= 0:
            return step
        return min(step, self.cfl * self.spacing_ / speed)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate from time `t` to time `until` with steps of the CFL condition.

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
        drift = self.problem_.drift
        state, time = x.copy(), t
        while drift is not None and time < until - 1e-12 * max(1.0, abs(until)):
            h = min(self._step_size(state, p), until - time)
            first = state + h * drift(time, state, p)
            if self.order == SECOND_ORDER:
                state = (state + first + h * drift(time + h, first, p)) / 2
            else:
                second = 0.75 * state + 0.25 * (first + h * drift(time + h, first, p))
                state = state / 3 + 2 / 3 * (second + h * drift(time + h / 2, second, p))
            time += h
            state = self._get_bounded(time, state, p)
        return state
