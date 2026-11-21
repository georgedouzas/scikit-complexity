"""Derivative-free Milstein scheme of Platen for SDEs with diagonal noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...components.laws import ConstraintLaw, EventLaw, NoiseLaw, RateLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.spaces import BaseSpace
from ...components.spaces._continuous_time import _ContinuousTime
from ...enums import NoiseKind, OrderKind
from ...types import Array
from ..backends import SupportSpec
from ._sde import SDESolver


class MilsteinSolver(SDESolver):
    """A solver of stochastic differential equations by the Milstein scheme of Platen.

    It takes a helper value `Y = X + f h + g sqrt(h)` and corrects the noise by
    `(g(Y) - g(X)) (dW^2 - h) / (2 sqrt(h))`, which needs no derivative of the noise and is of strong order 1. Choose it
    for noise whose size depends on the state, such as a price whose shocks grow with it, where it is closer to the
    exact path than Euler-Maruyama, in models of interest rates, populations, and stock prices.

    Read more in the user guide, under Solvers.

    Args:
        dt:
            The fixed internal step.

    See Also:
        EulerMaruyamaSolver: The default solver of noise in continuous time.

    Examples:
        >>> from skcomplex.core.engine.solvers import MilsteinSolver
        >>> from skcomplex.systems.economics.finance import GeometricBrownianMotionSystem
        >>> system = GeometricBrownianMotionSystem(solver=MilsteinSolver(dt=0.01)).prepare()
        >>> system.simulate(until=1, dt=1, random_state=0).get_values('asset.price').round(2).tolist()
        [100.0, 130.8]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_ContinuousTime},
            space={None, BaseSpace},
            laws={RateLaw, ConstraintLaw, EventLaw, _BoundaryLaw, NoiseLaw},
            noise=frozenset({NoiseKind.DIAGONAL}),
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
        """Return the state after one derivative-free Milstein step.

        The step takes given Wiener increments.

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
        problem, diffusion = self.problem_, self._get_diffusion()
        drift = self._get_drift()(time, state, p)
        amplitudes = diffusion(time, state, p)
        root = np.sqrt(step)
        support = state + drift * step
        np.add.at(support, problem.noise_rows, amplitudes * root)
        correction = (diffusion(time, support, p) - amplitudes) * (noise**2 - step) / (2 * root)
        following = state + drift * step
        np.add.at(following, problem.noise_rows, amplitudes * noise + correction)
        return following
