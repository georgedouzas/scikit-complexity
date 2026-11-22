"""Euler-Maruyama integrator of stochastic differential equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ...components.laws import ConstraintLaw, EventLaw, JumpLaw, NoiseLaw, RateLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.spaces import BaseSpace
from ...components.spaces._continuous_time import _ContinuousTime
from ...enums import NoiseKind, OrderKind
from ...errors import SolverError
from ...types import Array
from ..backends import SupportSpec
from ..problems._steps import step_count
from ._base import BaseSolver
from ._sde import SDESolver


class EulerMaruyamaSolver(SDESolver):
    """A solver that steps stochastic differential equations by Euler-Maruyama.

    Every step of length `h` sets `x + f(x) h + g(x) dW`, the drift `f` plus the noise `g` times a normal increment `dW`
    of variance `h`, such as a stock price of 100 that ends near 131 after a year in one run. Choose it as the default
    for noise in continuous time, in stock prices, diffusing particles, and noisy gene expression.

    Read more in the user guide, under Solvers.

    Args:
        dt:
            The largest internal time step.

    See Also:
        HeunSolver: Averages the drift over each step.
        MilsteinSolver: Corrects noise whose size depends on the state.

    Examples:
        >>> from skcomplex.core.engine.solvers import EulerMaruyamaSolver
        >>> from skcomplex.systems.economics.finance import GeometricBrownianMotionSystem
        >>> system = GeometricBrownianMotionSystem(solver=EulerMaruyamaSolver(dt=0.01)).prepare()
        >>> system.simulate(until=1, dt=1, random_state=0).get_values('asset.price').round(2).tolist()
        [100.0, 131.0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_ContinuousTime},
            space={None, BaseSpace},
            laws={
                JumpLaw,
                RateLaw,
                ConstraintLaw,
                EventLaw,
                _BoundaryLaw,
                NoiseLaw,
            },
            noise=frozenset(NoiseKind) - {NoiseKind.FRACTIONAL},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _is_renewing(cls) -> bool:
        """Return whether the solver fires renewal laws."""
        return True

    def _increment(
        self,
        time: float,
        state: Array,
        p: Array,
        *,
        step: float | Array,
        noise: Array,
        end: float,
    ) -> Array:
        """Return the state after one Euler-Maruyama step with given Wiener increments.

        The state is taken before the boundary laws.

        Args:
            time:
                The time at the start of the step.

            state:
                The state at the start of the step.

            p:
                The parameter vector.

            step:
                The step length, or one step length per state component.

            noise:
                The Wiener increments, one per noise amplitude.

            end:
                The time at the end of the step, capped before the next breakpoint.

        Returns:
            state:
                The state at the end of the step.
        """
        problem = self.problem_
        following = state + self._get_drift()(time, state, p) * step
        np.add.at(following, problem.noise_rows, self._get_diffusion()(time, state, p) * noise)
        return following

    def _step(self, time: float, state: Array, p: Array, lengths: Array) -> Array:
        """Return the state after one Euler-Maruyama step and the boundary laws.

        There is one step length per state.
        """
        noise = self._draw_increments(1.0) * np.sqrt(lengths[self.problem_.noise_rows])
        end = time + float(np.max(lengths))
        return self._get_bounded(end, self._increment(time, state, p, step=lengths, noise=noise, end=end), p)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate the drift and the noise from time `t` to time `until`.

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
        drift, diffusion = self.problem_.drift, self.problem_.diffusion
        if until <= t or drift is None or diffusion is None:
            return x
        if self.problem_.renewals:
            return self._advance_renewals(t, x, p, until, dt=self.dt, step=self._step)
        if BaseSolver._is_hybrid(self.problem_, t, until):
            return self._advance_hybrid(t, x, p, until)
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.euler_maruyama is not None:
            return accelerated.euler_maruyama(t, x, p, until, self.dt, self.rng_)
        n_steps = step_count(t, until, self.dt)
        step = (until - t) / n_steps
        state = x.copy()
        for index in range(n_steps):
            time = t + index * step
            noise = self._draw_increments(math.sqrt(step))
            following = self._increment(time, state, p, step=step, noise=noise, end=time + step)
            state = self._get_bounded(time + step, following, p)
        return state
