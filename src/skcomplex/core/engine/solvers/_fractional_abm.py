"""Fractional Adams-Bashforth-Moulton predictor-corrector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from scipy.special import gamma

from ...types import Array
from ._caputo import CaputoSolver
from ._rows import INITIAL_ROWS, _Rows


class FractionalABMSolver(CaputoSolver):
    """A solver of fractional equations by the Adams-Bashforth-Moulton method.

    It solves `x(t) = x_0 + (1 / Gamma(alpha)) int_0^t (t - s)^(alpha - 1) f(s, x(s)) ds`, the same equation written as
    an integral over the past, as Diethelm, Ford, and Freed propose: a rectangle rule over the whole run predicts the
    next state, and a trapezoidal rule corrects it, so it is of order `1 + alpha`. States of order 1 take Heun steps.
    Choose it for fractional equations of moderate size, such as the relaxation of a polymer from 1 to 0.43 in one time
    unit, the spread of a drug in tissue, and epidemics with memory.

    Read more in the user guide, under Solvers.

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

    See Also:
        L1Solver: An explicit scheme of order 1.
        GrunwaldLetnikovSolver: An explicit scheme of order 1.

    Examples:
        >>> from skcomplex.core.engine.solvers import FractionalABMSolver
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalRelaxationSystem
        >>> system = FractionalRelaxationSystem(order=0.5, solver=FractionalABMSolver(dt=0.001)).prepare()
        >>> system.simulate(until=1, dt=1).get_values('relaxation.x').round(2).tolist()
        [1.0, 0.43]
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
        self._resize(INITIAL_ROWS)
        alpha = self._unique
        self._predictor_scale = self._get_per_state(self.dt**alpha / gamma(alpha + 1))
        self._corrector_scale = self._get_per_state(self.dt**alpha / gamma(alpha + 2))

    def _resize(self, count: int) -> None:
        """Compute the weights of the predictor and the corrector up to a lag."""
        self._predictor = self._get_predictor_weights(self._unique, count)
        self._corrector = self._get_corrector_weights(self._unique, count)

    def _carry(self, x: Array) -> None:
        """Take the states of order 1 that changed between frames.

        The Heun steps start from them.
        """
        ordinary = ~self._fractional
        self._state[ordinary] = x[ordinary]

    def _step(self, p: Array) -> None:
        """Advance the run by one predictor-corrector step."""
        drift = self._get_drift()
        step = self._step_index
        time = self._origin + step * self.dt
        current = drift(time, self._state, p)
        if self._drifts is None:
            self._drifts = _Rows(current, None)
        else:
            self._drifts.append(current)
        if step + 1 >= self._predictor.shape[0]:
            self._resize(2 * (step + 1))
        drifts = self._drifts.get_rows(0, step + 1)
        ordinary = ~self._fractional
        predicted = self._initial + self._predictor_scale * self._convolve(self._predictor[step::-1], drifts)
        predicted[ordinary] = self._state[ordinary] + self.dt * current[ordinary]
        alpha = self._unique
        first = self._get_per_state(step ** (alpha + 1) - (step - alpha) * (step + 1) ** alpha) * drifts[0]
        past = self._convolve(self._corrector[step - 1 :: -1], drifts[1:]) if step else 0.0
        following = drift(time + self.dt, predicted, p)
        corrected = self._initial + self._corrector_scale * (following + first + past)
        corrected[ordinary] = self._state[ordinary] + self.dt / 2 * (current[ordinary] + following[ordinary])
        self._state = self._get_bounded(time + self.dt, corrected, p)

    @staticmethod
    def _get_predictor_weights(alpha: Array, count: int) -> Array:
        """Return the weights `(m + 1)^alpha - m^alpha` of the predictor.

        They cover lags 0 to `count`, with one column per order.
        """
        lags = np.arange(count + 1, dtype=float)[:, None]
        return (lags + 1) ** alpha[None, :] - lags ** alpha[None, :]

    @staticmethod
    def _get_corrector_weights(alpha: Array, count: int) -> Array:
        """Return the corrector weights at lags 0 to `count`.

        They cover every value but the first, with one column per order.
        """
        lags = np.arange(count + 1, dtype=float)[:, None]
        power = alpha[None, :] + 1
        return (lags + 2) ** power + lags**power - 2 * (lags + 1) ** power
