"""Predictor-corrector scheme for SDEs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...components.laws import ConstraintLaw, EventLaw, NoiseLaw, RateLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.laws._linear_program import _LinearProgramLaw
from ...components.spaces import BaseSpace
from ...components.spaces._continuous_time import _ContinuousTime
from ...enums import NoiseKind, OrderKind
from ...types import Array
from ..backends import SupportSpec
from ._sde import SDESolver


class HeunSolver(SDESolver):
    """A solver of stochastic differential equations by a predictor and a corrector.

    The predictor `X~ = X + f(X) h + g(X) dW` gives the corrector `X + (f(X) + f(X~)) h / 2 + g(X) dW`, which averages
    the drift over the step and keeps the noise of the start, so it is of strong order 1 for noise of fixed size. Choose
    it for smoother paths than Euler-Maruyama at the same step, in models of Brownian particles, noisy oscillators, and
    prices.

    Read more in the user guide, under Solvers.

    Args:
        dt:
            The fixed internal step.

    See Also:
        EulerMaruyamaSolver: The default solver of noise in continuous time.

    Examples:
        >>> from skcomplex.core.engine.solvers import HeunSolver
        >>> from skcomplex.systems.economics.finance import GeometricBrownianMotionSystem
        >>> system = GeometricBrownianMotionSystem(solver=HeunSolver(dt=0.01)).prepare()
        >>> system.simulate(until=1, dt=1, random_state=0).get_values('asset.price').round(2).tolist()
        [100.0, 131.01]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_ContinuousTime},
            space={None, BaseSpace},
            laws={_LinearProgramLaw, RateLaw, ConstraintLaw, EventLaw, _BoundaryLaw, NoiseLaw},
            noise=frozenset(NoiseKind) - {NoiseKind.FRACTIONAL},
            orders=frozenset({OrderKind.INTEGER}),
        )

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
        """Return the state after one Heun step with given Wiener increments.

        The corrector drift is taken at the end.

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
        problem, field = self.problem_, self._get_drift()
        drift = field(time, state, p)
        diffusion = np.zeros_like(state)
        np.add.at(diffusion, problem.noise_rows, self._get_diffusion()(time, state, p) * noise)
        predictor = state + drift * step + diffusion
        return state + (drift + field(end, predictor, p)) * step / 2 + diffusion
