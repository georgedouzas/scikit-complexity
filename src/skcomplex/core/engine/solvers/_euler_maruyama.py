"""Euler-Maruyama integrator of stochastic differential equations with a fixed internal step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ...components.laws import BoundaryLaw, CrossingLaw, EquationLaw, FluxLaw, JumpLaw, NoiseLaw, RateLaw, TransferLaw
from ...components.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError
from ...specs import SupportSpec
from ...types import Array, NoiseKind, OrderKind
from ..problems._steps import step_count
from ._base import BaseSolver
from ._sde import SDESolver


class EulerMaruyamaSolver(SDESolver):
    """Euler-Maruyama integrator of stochastic differential equations with a fixed internal step.

    Args:
        dt:
            The largest internal time step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.economics.finance import AssetElement, GeometricBrownianMotionInteraction
        >>> from skcomplex.core.engine.solvers import EulerMaruyamaSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> system = System(
        ...     elements=[AssetElement('stock', price=100)],
        ...     interactions=[GeometricBrownianMotionInteraction('gbm', target='stock', mu=0.05, sigma=0.2)],
        ... )
        >>> problem = simulate(system, until=0).problem
        >>> stepper = EulerMaruyamaSolver(dt=1e-3, random_state=0).prepare(problem)
        >>> bool(stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_[0] > 0)
        True
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
                NoiseLaw,
            },
            noise=frozenset(NoiseKind),
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
        """Return the state after one Euler-Maruyama step with given Wiener increments, before the boundary laws.

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
        following = state + self._drift()(time, state, p) * step
        np.add.at(following, problem.noise_rows, self._diffusion()(time, state, p) * noise)
        return following

    def _step(self, time: float, state: Array, p: Array, lengths: Array) -> Array:
        """Return the state after one Euler-Maruyama step and the boundary laws, with one step length per state."""
        noise = self._draw_increments(1.0) * np.sqrt(lengths[self.problem_.noise_rows])
        end = time + float(np.max(lengths))
        return self._bounded(end, self._increment(time, state, p, step=lengths, noise=noise, end=end), p)

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
            state = self._bounded(time + step, following, p)
        return state
