"""Monitor the totals of conserved quantities during a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ...errors import ConservationError, SimulationError, ValidationError
from ...types import Array
from ..problems import BaseProblemSpec, FrameSpec, TransferNetworkSpec
from ._base import BaseObserver


class ConservationObserver(BaseObserver):
    """Observer that checks at every frame that each quantity changes only by its flows with the environment.

    The total of a quantity is the sum of the states that hold it. The monitor compares its change since the first
    frame with the accumulated net inflow from the environment, which the compiler integrates as a hidden state, and
    raises an error when they differ by more than the tolerance, such as after a boundary law clips a state.

    Args:
        rtol:
            The relative tolerance, a fraction of the largest absolute total seen.

        atol:
            The absolute tolerance.

    Attributes:
        problem_:
            The compiled problem of the run.

        transfer_network_:
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
        >>> _ = Simulation(create('bathtub'), dt=1, observers=[monitor]).check().prepare().run(10)
        >>> round(monitor.totals_['water'], 6)
        25.0
    """

    def __init__(self, *, rtol: float = 1e-6, atol: float = 1e-9) -> None:
        self.rtol = rtol
        self.atol = atol

    def _check(self) -> None:
        """Raise an error unless both tolerances are nonnegative numbers."""
        for name, value in (('rtol', self.rtol), ('atol', self.atol)):
            if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0:
                error_msg = f'{name} = {value!r} must be a nonnegative number.'
                raise ValidationError(error_msg)

    def _totals(self, state: Array) -> dict[str, float]:
        """Return the total of every quantity minus its accumulated inflow from the environment."""
        layout = self.problem_.states
        totals = {}
        for quantity, names in self.transfer_network_.quantities.items():
            total = sum(float(np.sum(layout.read(state, name))) for name in names)
            self.totals_[quantity] = total
            boundary = self.transfer_network_.boundary.get(quantity)
            totals[quantity] = total - (float(np.sum(layout.read(state, boundary))) if boundary is not None else 0.0)
        return totals

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the quantities of the problem.

        Args:
            context:
                The compiled problem the run advances.

        Raises:
            SimulationError:
                No problem is passed, or the problem has no transfer network, such as a partitioned problem.
        """
        if context is None:
            error_msg = 'The conservation monitor prepares on a compiled problem. Pass the problem to `prepare`.'
            raise SimulationError(error_msg)
        problem = context
        transfer_network = problem.transfer_network
        if transfer_network is None:
            error_msg = 'The conservation monitor needs a problem compiled as one partition with its transfer network.'
            raise SimulationError(error_msg)
        self.problem_ = problem
        self.transfer_network_: TransferNetworkSpec = transfer_network
        self.totals_: dict[str, float] = {}
        self.initial_: dict[str, float] | None = None
        self.scale_: dict[str, float] = {}

    def _run(self, frame: FrameSpec | None = None) -> None:
        """Check the totals of a frame against the first frame.

        Args:
            frame:
                The recorded frame, or `None` at the last run, which checks nothing.

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
