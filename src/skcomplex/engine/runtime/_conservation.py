"""Monitor the totals of conserved quantities during a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

import numpy as np

from ...core.base import BaseProblem, ConservationError, SimulationError, Vector
from ...core.problems import Problem, StockFlow
from ._frames import Frame
from ._observers import BaseObserver


class ConservationObserver(BaseObserver):
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
        problem_:
            The compiled problem of the run.

        stock_flow_:
            The quantities of the problem, with the states that hold them and their flows with the environment.

        totals_:
            The total of every quantity at the latest frame.

        initial_:
            The total of every quantity minus its inflow from the environment at the first frame, or `None`.

        scale_:
            The largest absolute total of every quantity seen so far.

    Examples:
        >>> from skcomplex.engine.runtime import ConservationObserver, Simulation
        >>> from skcomplex.models import create
        >>> monitor = ConservationObserver()
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

    def _fit(self, problem: BaseProblem) -> None:
        """Keep the quantities of the problem.

        Raises:
            SimulationError:
                The problem has no stock-flow description, such as a partitioned problem.
        """
        stock_flow = problem.stock_flow if isinstance(problem, Problem) else None
        if stock_flow is None:
            error_msg = 'The conservation monitor needs a problem compiled as one partition with its stock flows.'
            raise SimulationError(error_msg)
        self.problem_: Problem = cast('Problem', problem)
        self.stock_flow_: StockFlow = stock_flow
        self.totals_: dict[str, float] = {}
        self.initial_: dict[str, float] | None = None
        self.scale_: dict[str, float] = {}

    def _run(self, frame: Frame | None = None) -> None:
        """Check the totals of a frame against the first frame.

        Raises:
            ConservationError:
                A total changed by more than its inflow from the environment.
        """
        if frame is None:
            return
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


class InvariantObserver(BaseObserver):
    """Observer that checks at every frame that every invariant of the laws stays within its tolerance.

    Every run attaches one when its problem declares invariants.

    Attributes:
        invariants_:
            The tolerance of every invariant, by the path of its observable.

        initial_:
            The value of every invariant at the first frame.
    """

    def _fit(self, problem: BaseProblem) -> None:
        """Keep the invariants of the problem."""
        stock_flow = problem.stock_flow if isinstance(problem, Problem) else None
        self.invariants_ = dict(stock_flow.invariants) if stock_flow is not None else {}
        self.initial_: dict[str, float] = {}

    def _run(self, frame: Frame | None = None) -> None:
        """Check the invariants of a frame against the first frame.

        Raises:
            ConservationError:
                An invariant changed by more than its tolerance.
        """
        if frame is None:
            return
        for path, tolerance in self.invariants_.items():
            value = float(np.sum(frame.observables[path]))
            initial = self.initial_.setdefault(path, value)
            if abs(value - initial) > tolerance:
                error_msg = (
                    f'The invariant {path} changed from {initial:.6g} to {value:.6g} at frame {frame.step}, time '
                    f'{frame.time:.6g}, beyond its tolerance {tolerance:g}.'
                )
                raise ConservationError(error_msg)
