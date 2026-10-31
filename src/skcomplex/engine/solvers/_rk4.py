"""Classical fourth-order Runge-Kutta integrator of ordinary differential equations with a fixed internal step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable

import numpy as np

from ...core.laws import (
    BoundaryLaw,
    CrossingLaw,
    DestructionLaw,
    EncounterLaw,
    EquationLaw,
    FluxLaw,
    JumpLaw,
    ProliferationLaw,
    RateLaw,
    TransferLaw,
)
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError
from ...specs import SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems._steps import step_count
from ..types import Interpolant
from ._base import BaseSolver
from ._history import History
from ._solver import Solver


class RK4Solver(Solver):
    """Classical fourth-order Runge-Kutta integrator of ordinary differential equations with a fixed internal step.

    With the Numba backend every internal step of a frame runs in compiled code.

    Args:
        dt:
            The largest internal time step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
        >>> from skcomplex.engine.solvers import RK4Solver
        >>> from skcomplex.specs import StepSpec
        >>> problem = compile_system(
        ...     System(
        ...         elements=[PopulationElement('a', n=1)],
        ...         interactions=[ExponentialGrowthInteraction('g', target='a', r=1)],
        ...     ),
        ... )
        >>> stepper = RK4Solver(dt=0.001, random_state=0).check().prepare(problem)
        >>> step = StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)
        >>> round(float(stepper.run(step).state_[0]), 10)
        2.7182818285
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={
                JumpLaw,
                RateLaw,
                FluxLaw,
                EquationLaw,
                CrossingLaw,
                BoundaryLaw,
                TransferLaw,
                EncounterLaw,
                ProliferationLaw,
                DestructionLaw,
            },
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

    @classmethod
    def _is_renewing(cls) -> bool:
        """Return whether the solver fires renewal laws."""
        return True

    def __init__(
        self,
        dt: float = 1e-3,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _step(self, time: float, state: Array, p: Array, lengths: float | Array) -> Array:
        """Return the state after one Runge-Kutta step and the boundary laws, with one step length or one per state."""
        drift = self._drift()

        def field(time: float, state: Array) -> Array:
            return drift(time, state, p)

        return self._bounded(time + float(np.max(lengths)), self._rk4_step(field, time, state, lengths), p)

    def _advance_hybrid(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate with fixed RK4 steps, interpolating every step for delays and locating events in it."""
        problem = self.problem_
        drift = self._drift()
        history = self.history_ = History._open_history(problem, self.history_, t, x)

        def take(time: float, state: Array, step: float, ceiling: float) -> tuple[Array, Interpolant]:
            def field(time: float, state: Array) -> Array:
                time = min(time, ceiling)
                return drift(time, state, History._with_delays(problem, history, time, state, p))

            following = self._rk4_step(field, time, state, step)
            interpolant = self._hermite(
                time,
                state,
                field(time, state),
                t1=time + step,
                x1=following,
                f1=field(time + step, following),
            )
            return following, interpolant

        return BaseSolver._advance_fixed_steps(
            t,
            x,
            p,
            until,
            dt=self.dt,
            locator=self.locator_,
            history=history,
            bounded=self._bounded,
            take=take,
        )

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate the drift from time `t` to time `until` in equal steps no longer than `dt`.

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

        Raises:
            SolverError:
                The internal step is not positive.
        """
        if not self.dt > 0:
            error_msg = f'dt = {self.dt!r} must be positive.'
            raise SolverError(error_msg)
        drift = self.problem_.drift
        if until <= t or drift is None:
            return x
        if self.problem_.renewals:
            return self._advance_renewals(t, x, p, until, dt=self.dt, step=self._step)
        if BaseSolver._is_hybrid(self.problem_, t, until):
            return self._advance_hybrid(t, x, p, until)
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.rk4 is not None:
            return accelerated.rk4(t, x, p, until, self.dt)
        n_steps = step_count(t, until, self.dt)
        step = (until - t) / n_steps
        state = x.copy()
        for index in range(n_steps):
            state = self._step(t + index * step, state, p, step)
        return state

    @staticmethod
    def _rk4_step(field: Callable[[float, Array], Array], time: float, state: Array, step: float | Array) -> Array:
        """Return the state after one classical Runge-Kutta step, with one step length or one per state component.

        The step spans the longest of the step lengths in time.

        Args:
            field:
                The drift `field(t, x)`.

            time:
                The time at the start of the step.

            state:
                The state at the start of the step.

            step:
                The step length, or one step length per state component.

        Returns:
            state:
                The state at the end of the step, before the boundary laws.
        """
        span = float(np.max(step))
        k1 = field(time, state)
        k2 = field(time + span / 2, state + step / 2 * k1)
        k3 = field(time + span / 2, state + step / 2 * k2)
        k4 = field(time + span, state + step * k3)
        return state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def _hermite(t0: float, x0: Array, f0: Array, *, t1: float, x1: Array, f1: Array) -> Interpolant:
        """Return the cubic Hermite interpolant between two states with their derivatives.

        Args:
            t0:
                The first time.

            x0:
                The first state.

            f0:
                The derivative at the first time.

            t1:
                The second time.

            x1:
                The second state.

            f1:
                The derivative at the second time.

        Returns:
            interpolant:
                The function of time.
        """
        span = t1 - t0

        def evaluate(time: float) -> Array:
            s = (time - t0) / span if span else 0.0
            return (
                (2 * s**3 - 3 * s**2 + 1) * x0
                + (s**3 - 2 * s**2 + s) * span * f0
                + (-2 * s**3 + 3 * s**2) * x1
                + (s**3 - s**2) * span * f1
            )

        return evaluate
