"""Base class of explicit schemes that weigh the displacements of the states from the start of a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from abc import abstractmethod

import numpy as np

from ...core.errors import SolverError
from ...core.types import Vector
from ._caputo import CaputoSolver
from ._rows import _INITIAL_ROWS, _Rows

_SLOWEST_MODE = 1e-7


_FASTEST_MODE = 60.0


_MODE_SPACING = 0.35


class ConvolutionSolver(CaputoSolver):
    """Base class of explicit schemes that weigh the displacements of the states from the start of a run.

    A step sets `g dt^alpha f(t_(k-1), x_(k-1)) = sum_(j=0)^k c_j (x_(k-j) - x_0)` with `c_0 = 1`. With a memory
    window, the displacements older than the window enter through exponential modes fitted to the weights.
    """

    def __init__(self, dt: float = 0.01, memory: float | None = None) -> None:
        self.dt = dt
        self.memory = memory

    @abstractmethod
    def _weights(self, alpha: Vector, count: int) -> Vector:
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
    def _tail(self, alpha: Vector, rates: Vector, window: int) -> Vector:
        """Return the coefficients of the exponential modes that give the weights past a window, one column per order.

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
    def _gain(self, alpha: Vector) -> Vector:
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

    def _restart(self, t: float, x: Vector) -> None:
        """Start the history of displacements, the weights, and the exponential modes of the tail."""
        del t
        if self.memory is not None and not self.memory > 0:
            error_msg = f'memory = {self.memory!r} must be positive or None.'
            raise SolverError(error_msg)
        self._window = None if self.memory is None else max(1, round(self.memory / self.dt))
        self._rows = _Rows(np.zeros(x.size), self._window)
        self._table = self._weights(self._unique, _INITIAL_ROWS if self._window is None else self._window)
        self._scale = self._per_state(self._gain(self._unique) * self.dt**self._unique)
        if self._window is not None:
            rates = self._tail_rates(self._window)
            self._decay = np.exp(-rates)[:, None]
            self._modes = self._tail(self._unique, rates, self._window)[:, self._groups]
            self._amplitudes = np.zeros((rates.size, x.size))

    def _carry(self, x: Vector) -> None:
        """Replace the last displacement of the states of order 1 by the given state."""
        ordinary = ~self._fractional
        self._state[ordinary] = x[ordinary]
        self._rows.row(self._step_index)[ordinary] = x[ordinary] - self._initial[ordinary]

    def _history_sum(self, step: int) -> Vector:
        """Return the weighted sum of the displacements before a step, with the tail of a memory window."""
        window = self._window
        lags = step - 1 if window is None else min(step - 1, window)
        if lags >= self._table.shape[0]:
            self._table = self._weights(self._unique, 2 * lags)
        total = self._convolve(self._table[lags:0:-1], self._rows.rows(step - lags, step)) if lags else 0.0
        if window is None or step <= window + 1:
            return np.asarray(total + np.zeros(self._state.size))
        self._amplitudes = self._decay * self._amplitudes + self._rows.row(step - window - 1)
        return np.asarray(total + np.sum(self._modes * self._amplitudes, axis=0))

    def _step(self, p: Vector) -> None:
        """Advance the displacements by one explicit step."""
        step = self._step_index + 1
        time = self._origin + self._step_index * self.dt
        drift = self.problem_.drift(time, self._state, p)  # type: ignore[misc]
        displacement = self._scale * drift - self._history_sum(step)
        self._state = self._bounded(time + self.dt, self._initial + displacement, p)
        self._rows.append(self._state - self._initial)

    @staticmethod
    def _tail_rates(window: int) -> Vector:
        """Return the decay rates per step of the exponential modes that hold the history older than a window."""
        exponents = np.arange(math.log(_SLOWEST_MODE), math.log(_FASTEST_MODE / (window + 1)), _MODE_SPACING)
        return np.exp(exponents)
