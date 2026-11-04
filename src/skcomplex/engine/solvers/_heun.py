"""Predictor-corrector scheme for SDEs that averages the drift over the step and keeps the Itô noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...core.laws import (
    BoundaryLaw,
    CrossingLaw,
    EquationLaw,
    FluxLaw,
    LinearProgramLaw,
    NoiseLaw,
    RateLaw,
    TransferLaw,
)
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...specs import SupportSpec
from ...types import Array, NoiseKind, OrderKind
from ._sde import SDESolver


class HeunSolver(SDESolver):
    """Predictor-corrector scheme for SDEs that averages the drift over the step and keeps the Itô noise.

    The predictor `X~ = X + f(X) h + g(X) dW` gives the corrector `X + (f(X) + f(X~)) h / 2 + g(X) dW`, which has strong
    order 1 for additive noise and integrates the Itô form for any noise.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.models.economics.finance import geometric_brownian_motion
        >>> from skcomplex.engine.solvers import HeunSolver
        >>> from skcomplex.specs import StepSpec
        >>> problem = compile_system(geometric_brownian_motion())
        >>> stepper = HeunSolver(dt=0.01, random_state=0).check().prepare(problem)
        >>> bool(stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_[0] > 0)
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={LinearProgramLaw, RateLaw, FluxLaw, EquationLaw, CrossingLaw, BoundaryLaw, TransferLaw, NoiseLaw},
            noise=frozenset(NoiseKind),
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
        """Return the state after one Heun step with given Wiener increments, with the corrector drift at the end.

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
        problem, field = self.problem_, self._drift()
        drift = field(time, state, p)
        diffusion = np.zeros_like(state)
        np.add.at(diffusion, problem.noise_rows, self._diffusion()(time, state, p) * noise)
        predictor = state + drift * step + diffusion
        return state + (drift + field(end, predictor, p)) * step / 2 + diffusion
