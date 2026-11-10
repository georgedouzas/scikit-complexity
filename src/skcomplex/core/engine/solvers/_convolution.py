"""Base class of explicit schemes that weigh the displacements of the states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from abc import abstractmethod

import numpy as np

from ...errors import SolverError
from ...types import Array
from ._caputo import CaputoSolver
from ._rows import INITIAL_ROWS, _Rows

SLOWEST_MODE = 1e-7
FASTEST_MODE = 60.0
MODE_SPACING = 0.35


class ConvolutionSolver(CaputoSolver):
    """An abstract solver of explicit schemes that weigh past steps of the states.

    The displacements are taken from the start of a run. A step sets `g dt^alpha f(t_(k-1), x_(k-1)) = sum_(j=0)^k c_j
    (x_(k-j) - x_0)` with `c_0 = 1`. With a memory window, the displacements older than the window enter through
    exponential modes fitted to the weights.
    """

    def __init__(
        self,
        dt: float = 0.01,
        memory: float | None = None,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.dt = dt
        self.memory = memory
        self.random_state = random_state
        self.until = until

    @abstractmethod
    def _get_weights(self, alpha: Array, count: int) -> Array:
        """Return the weights `c_0` to `c_count`, one column per order.

        Args:
            alpha:
                The distinct orders.

            count:
                The largest lag.

        Returns:
            weights:
                The weights, one row per lag.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_tail(self, alpha: Array, rates: Array, window: int) -> Array:
        """Return the coefficients of the exponential modes past a window.

        The modes give the weights past the window, with one column per order.

        Args:
            alpha:
                The distinct orders.

            rates:
                The decay rates of the modes per step.

            window:
                The number of steps of the window.

        Returns:
            coefficients:
                The coefficients, one row per mode.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_gain(self, alpha: Array) -> Array:
        """Return the factor `g` of the drift of every order.

        Args:
            alpha:
                The distinct orders.

        Returns:
            gain:
                The factor of every order.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def _restart(self, t: float, x: Array) -> None:
        """Start the history of displacements, the weights, and the tail modes.

        The tail modes are the exponential modes of the tail.
        """
        del t
        if self.memory is not None and not self.memory > 0:
            error_msg = f'memory = {self.memory!r} must be positive or None.'
            raise SolverError(error_msg)
        self._window = None if self.memory is None else max(1, round(self.memory / self.dt))
        self._rows = _Rows(np.zeros(x.size), self._window)
        self._table = self._get_weights(self._unique, INITIAL_ROWS if self._window is None else self._window)
        self._scale = self._get_per_state(self._get_gain(self._unique) * self.dt**self._unique)
        if self._window is not None:
            rates = self._get_tail_rates(self._window)
            self._decay = np.exp(-rates)[:, None]
            self._modes = self._get_tail(self._unique, rates, self._window)[:, self._groups]
            self._amplitudes = np.zeros((rates.size, x.size))

    def _carry(self, x: Array) -> None:
        """Replace the last displacement of the states of order 1 by the given state."""
        ordinary = ~self._fractional
        self._state[ordinary] = x[ordinary]
        self._rows.get_row(self._step_index)[ordinary] = x[ordinary] - self._initial[ordinary]

    def _get_history_sum(self, step: int) -> Array:
        """Return the weighted sum of the displacements before a step.

        It includes the tail of a memory window.
        """
        window = self._window
        lags = step - 1 if window is None else min(step - 1, window)
        if lags >= self._table.shape[0]:
            self._table = self._get_weights(self._unique, 2 * lags)
        total = self._convolve(self._table[lags:0:-1], self._rows.get_rows(step - lags, step)) if lags else 0.0
        if window is None or step <= window + 1:
            return np.asarray(total + np.zeros(self._state.size))
        self._amplitudes = self._decay * self._amplitudes + self._rows.get_row(step - window - 1)
        return np.asarray(total + np.sum(self._modes * self._amplitudes, axis=0))

    def _step(self, p: Array) -> None:
        """Advance the displacements by one explicit step."""
        step = self._step_index + 1
        time = self._origin + self._step_index * self.dt
        drift = self._get_drift()(time, self._state, p)
        displacement = self._scale * drift - self._get_history_sum(step)
        self._state = self._get_bounded(time + self.dt, self._initial + displacement, p)
        self._rows.append(self._state - self._initial)

    @staticmethod
    def _get_tail_rates(window: int) -> Array:
        """Return the decay rates per step of the exponential modes of the tail.

        These modes hold the history older than a window.
        """
        exponents = np.arange(math.log(SLOWEST_MODE), math.log(FASTEST_MODE / (window + 1)), MODE_SPACING)
        return np.exp(exponents)
