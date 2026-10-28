"""Derivative-free Milstein scheme of Platen for SDEs with diagonal noise, of strong order 1."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ...core.laws import BoundaryLaw, CrossingLaw, EquationLaw, FluxLaw, NoiseLaw, RateLaw, TransferLaw
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...types import Array
from ..problems import NoiseKind, OrderKind, SupportSpec
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
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.models.economics.finance import geometric_brownian_motion
        >>> from skcomplex.engine.solvers import MilsteinSolver
        >>> problem = compile_system(geometric_brownian_motion())
        >>> stepper = MilsteinSolver(dt=0.01).prepare(problem, np.random.default_rng(0))
        >>> bool(stepper.run(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0] > 0)
        True
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={ContinuousTime},
        space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
        laws={RateLaw, FluxLaw, EquationLaw, CrossingLaw, BoundaryLaw, TransferLaw, NoiseLaw},
        noise={NoiseKind.DIAGONAL},
        orders={OrderKind.INTEGER},
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
        problem = self.problem_
        drift = problem.drift(time, state, p)  # type: ignore[misc]
        amplitudes = problem.diffusion(time, state, p)  # type: ignore[misc]
        root = np.sqrt(step)
        support = state + drift * step
        np.add.at(support, problem.noise_rows, amplitudes * root)
        correction = (problem.diffusion(time, support, p) - amplitudes) * (noise**2 - step) / (2 * root)  # type: ignore[misc]
        following = state + drift * step
        np.add.at(following, problem.noise_rows, amplitudes * noise + correction)
        return following
