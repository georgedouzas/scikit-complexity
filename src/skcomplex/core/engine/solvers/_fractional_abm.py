"""Fractional Adams-Bashforth-Moulton predictor-corrector of Diethelm, Ford, and Freed, of order `1 + alpha`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from scipy.special import gamma

from ...types import Array
from ._caputo import CaputoSolver
from ._rows import _INITIAL_ROWS, _Rows


class FractionalABMSolver(CaputoSolver):
    """Fractional Adams-Bashforth-Moulton predictor-corrector of Diethelm, Ford, and Freed, of order `1 + alpha`.

    The scheme solves the equivalent Volterra equation `x(t) = x_0 + (1 / Gamma(alpha)) int_0^t (t - s)^(alpha - 1)
    f(s, x(s)) ds`. A rectangle rule over the drifts of the whole run predicts the next state, and a trapezoidal rule
    with the drift at the prediction corrects it. States of order 1 take Heun steps.

    Args:
        dt:
            The fixed step. Frames must fall on multiples of it.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Attributes:
        orders_:
            The order of every state component.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.engine.solvers import FractionalABMSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> class RelaxingElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec(order=0.5)] = 1.0):
        ...         super().__init__(label)
        ...         self.x = x
        ...
        ...     def _get_laws(self, s):
        ...         return [RateLaw(s.x, -s.x)]
        >>> problem = simulate(System(elements=[RelaxingElement('a')]), until=0).problem
        >>> stepper = FractionalABMSolver(dt=0.001, random_state=0).prepare(problem)
        >>> step = StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)
        >>> round(float(stepper.run(step).state_[0]), 2)
        0.43
    """

    def __init__(
        self,
        dt: float = 0.01,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _restart(self, t: float, x: Array) -> None:
        """Start the history of drifts and the weights of the predictor and the corrector."""
        del t, x
        self._drifts: _Rows | None = None
        self._resize(_INITIAL_ROWS)
        alpha = self._unique
        self._predictor_scale = self._per_state(self.dt**alpha / gamma(alpha + 1))
        self._corrector_scale = self._per_state(self.dt**alpha / gamma(alpha + 2))

    def _resize(self, count: int) -> None:
        """Compute the weights of the predictor and the corrector up to a lag."""
        self._predictor = self._predictor_weights(self._unique, count)
        self._corrector = self._corrector_weights(self._unique, count)

    def _carry(self, x: Array) -> None:
        """Take the states of order 1 that changed between frames, which the Heun steps start from."""
        ordinary = ~self._fractional
        self._state[ordinary] = x[ordinary]

    def _step(self, p: Array) -> None:
        """Advance the run by one predictor-corrector step."""
        drift = self._drift()
        step = self._step_index
        time = self._origin + step * self.dt
        current = drift(time, self._state, p)
        if self._drifts is None:
            self._drifts = _Rows(current, None)
        else:
            self._drifts.append(current)
        if step + 1 >= self._predictor.shape[0]:
            self._resize(2 * (step + 1))
        drifts = self._drifts.rows(0, step + 1)
        ordinary = ~self._fractional
        predicted = self._initial + self._predictor_scale * self._convolve(self._predictor[step::-1], drifts)
        predicted[ordinary] = self._state[ordinary] + self.dt * current[ordinary]
        alpha = self._unique
        first = self._per_state(step ** (alpha + 1) - (step - alpha) * (step + 1) ** alpha) * drifts[0]
        past = self._convolve(self._corrector[step - 1 :: -1], drifts[1:]) if step else 0.0
        following = drift(time + self.dt, predicted, p)
        corrected = self._initial + self._corrector_scale * (following + first + past)
        corrected[ordinary] = self._state[ordinary] + self.dt / 2 * (current[ordinary] + following[ordinary])
        self._state = self._bounded(time + self.dt, corrected, p)

    @staticmethod
    def _predictor_weights(alpha: Array, count: int) -> Array:
        """Return the weights `(m + 1)^alpha - m^alpha` of the predictor at lags 0 to `count`, one column per order."""
        lags = np.arange(count + 1, dtype=float)[:, None]
        return (lags + 1) ** alpha[None, :] - lags ** alpha[None, :]

    @staticmethod
    def _corrector_weights(alpha: Array, count: int) -> Array:
        """Return the corrector weights at lags 0 to `count` for every value but the first, one column per order."""
        lags = np.arange(count + 1, dtype=float)[:, None]
        power = alpha[None, :] + 1
        return (lags + 2) ** power + lags**power - 2 * (lags + 1) ** power
