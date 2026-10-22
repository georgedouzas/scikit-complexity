"""Monitor the totals of conserved quantities during a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ..base import ConservationError, SimulationError
from ..compile import Problem, StockFlow, Vector
from ._frames import Frame
from ._observers import Observer


class ConservationMonitor(Observer):
    """Observer that checks at every frame that each quantity changes only by its flows with the environment.

    The total of a quantity is the sum of the states that hold it. The monitor compares its change since the first
    frame with the accumulated net inflow from the environment, which the compiler integrates as a hidden state, and
    raises an error when they differ by more than the tolerance, such as after a boundary law clips a stock.

    Args:
        rtol:
            The relative tolerance, a fraction of the largest absolute total seen.

        atol:
            The absolute tolerance.

    Attributes:
        totals_:
            The total of every quantity at the latest frame.

    Examples:
        >>> from skcomplex.core.runtime import ConservationMonitor, Simulation
        >>> from skcomplex.models import create
        >>> monitor = ConservationMonitor()
        >>> _ = Simulation(create('bathtub'), dt=1, observers=[monitor]).run(10)
        >>> round(monitor.totals_['water'], 6)
        25.0
    """

    def __init__(self, *, rtol: float = 1e-6, atol: float = 1e-9) -> None:
        self.rtol = rtol
        self.atol = atol

    def _totals(self, state: Vector) -> dict[str, float]:
        """Return the total of every quantity minus its accumulated inflow from the environment."""
        layout = self.problem_.states
        totals = {}
        for quantity, names in self.stock_flow_.quantities.items():
            total = sum(float(np.sum(layout.read(state, name))) for name in names)
            self.totals_[quantity] = total
            boundary = self.stock_flow_.boundary.get(quantity)
            totals[quantity] = total - (float(np.sum(layout.read(state, boundary))) if boundary is not None else 0.0)
        return totals

    def on_start(self, problem: Problem) -> None:
        """Keep the quantities of the problem.

        Args:
            problem:
                The compiled problem.

        Raises:
            SimulationError:
                The problem has no stock-flow description, such as a partitioned problem.
        """
        if problem.stock_flow is None:
            error_msg = 'The conservation monitor needs a problem compiled as one partition with its stock flows.'
            raise SimulationError(error_msg)
        self.problem_: Problem = problem
        self.stock_flow_: StockFlow = problem.stock_flow
        self.totals_: dict[str, float] = {}
        self.initial_: dict[str, float] | None = None
        self.scale_: dict[str, float] = {}

    def on_frame(self, frame: Frame) -> None:
        """Check the totals of a frame against the first frame.

        Args:
            frame:
                The frame.

        Raises:
            ConservationError:
                A total changed by more than its inflow from the environment.
        """
        balances = self._totals(frame.state)
        if self.initial_ is None:
            self.initial_ = balances
        for quantity, balance in balances.items():
            scale = max(self.scale_.get(quantity, 0.0), abs(self.totals_[quantity]))
            self.scale_[quantity] = scale
            drift = balance - self.initial_[quantity]
            if abs(drift) > self.atol + self.rtol * scale:
                error_msg = (
                    f'The total of {quantity!r} drifted by {drift:.6g} at time {frame.time:g} beyond its flows with '
                    'the environment. Change the states that hold it only with flows.'
                )
                raise ConservationError(error_msg)


class InvariantMonitor(Observer):
    """Observer that checks at every frame that every invariant of the laws stays within its tolerance.

    Every run attaches one when its problem declares invariants.

    Attributes:
        initial_:
            The value of every invariant at the first frame.
    """

    def on_start(self, problem: Problem) -> None:
        """Keep the invariants of the problem.

        Args:
            problem:
                The compiled problem.
        """
        self.invariants_ = dict(problem.stock_flow.invariants) if problem.stock_flow is not None else {}
        self.initial_: dict[str, float] = {}

    def on_frame(self, frame: Frame) -> None:
        """Check the invariants of a frame against the first frame.

        Args:
            frame:
                The frame.

        Raises:
            ConservationError:
                An invariant changed by more than its tolerance.
        """
        for path, tolerance in self.invariants_.items():
            value = float(np.sum(frame.observables[path]))
            initial = self.initial_.setdefault(path, value)
            if abs(value - initial) > tolerance:
                error_msg = (
                    f'The invariant {path} changed from {initial:.6g} to {value:.6g} at frame {frame.step}, time '
                    f'{frame.time:.6g}, beyond its tolerance {tolerance:g}.'
                )
                raise ConservationError(error_msg)
