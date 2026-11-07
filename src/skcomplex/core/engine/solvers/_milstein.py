"""Derivative-free Milstein scheme of Platen for SDEs with diagonal noise, of strong order 1."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...components.laws import BoundaryLaw, CrossingLaw, EquationLaw, FluxLaw, NoiseLaw, RateLaw, TransferLaw
from ...components.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...specs import SupportSpec
from ...types import Array, NoiseKind, OrderKind
from ._sde import SDESolver


class MilsteinSolver(SDESolver):
    """Derivative-free Milstein scheme of Platen for SDEs with diagonal noise, of strong order 1.

    It takes a supporting value `Y = X + f h + g sqrt(h)` and corrects the noise with
    `(g(Y) - g(X)) (dW^2 - h) / (2 sqrt(h))`, which needs no derivative of the noise.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.compile import PassCompiler
        >>> from skcomplex.systems.economics.finance import GeometricBrownianMotionSystem
        >>> from skcomplex.core.engine.solvers import MilsteinSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> problem = PassCompiler().prepare(GeometricBrownianMotionSystem().prepare().system_).run().problem_
        >>> stepper = MilsteinSolver(dt=0.01, random_state=0).prepare(problem)
        >>> bool(stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_[0] > 0)
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={RateLaw, FluxLaw, EquationLaw, CrossingLaw, BoundaryLaw, TransferLaw, NoiseLaw},
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
        """Return the state after one derivative-free Milstein step with given Wiener increments.

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
        problem, diffusion = self.problem_, self._diffusion()
        drift = self._drift()(time, state, p)
        amplitudes = diffusion(time, state, p)
        root = np.sqrt(step)
        support = state + drift * step
        np.add.at(support, problem.noise_rows, amplitudes * root)
        correction = (diffusion(time, support, p) - amplitudes) * (noise**2 - step) / (2 * root)
        following = state + drift * step
        np.add.at(following, problem.noise_rows, amplitudes * noise + correction)
        return following
