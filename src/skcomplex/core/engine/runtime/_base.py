"""Define the bases of the runtime."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import time
from typing import Self

from ...base import BaseConfig
from ..problems import BaseProblemSpec, FrameSpec


class BaseObserver(BaseConfig[BaseProblemSpec, FrameSpec]):
    """An abstract observer, which receives the frames of a run one by one.

    A run calls `prepare(problem)` with the compiled problem before the first frame, `run(frame)` with every frame, such
    as the frame of time 0 that holds the starting values, and `run(final=True)` once the run stops, which frees what
    `prepare` took, such as an open file. A subclass writes any of the private methods `_prepare`, `_run`, and
    `_release`. Observers record the course of an epidemic, check that the money of an economy is conserved, and draw a
    flock as it moves.

    Read more in the user guide, under Observe.

    See Also:
        RecorderObserver: Keeps every frame.
        ConservationObserver: Checks that conserved quantities hold.
        InvariantObserver: Checks that observables stay near their first value.

    Examples:
        >>> from skcomplex.core.engine.runtime import BaseObserver, RecorderObserver
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> recorder = RecorderObserver()
        >>> _ = LogisticGrowthSystem().prepare().simulate(until=2, dt=1, observers=[recorder])
        >>> isinstance(recorder, BaseObserver), len(recorder.frames_)
        (True, 3)
    """

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Prepare for a run of the compiled problem.

        By default, prepare nothing.
        """

    def _run(self, inputs: FrameSpec | None) -> None:
        """React to a frame, or to the end of the run without one.

        By default, do nothing.
        """

    def prepare(self, problem: BaseProblemSpec | None = None) -> Self:
        """Prepare the checked observer for a run of a compiled problem.

        Args:
            problem:
                The compiled problem the run advances.

        Returns:
            self:
                The prepared observer.
        """
        return super().prepare(problem)

    def run(self, frame: FrameSpec | None = None, final: bool = False) -> Self:
        """React to a frame.

        At the last call, it releases what preparing acquired.

        Args:
            frame:
                The frame, or `None` for the call that ends the run.

            final:
                Whether this is the last call of the run.

        Returns:
            self:
                The observer.
        """
        super().run(frame, final)
        return self


class BaseClock(BaseConfig[None, float]):
    """An abstract clock, which paces a run against wall time.

    A clock decides how long a run waits between frames. `prepare()` marks the wall time at the start of a run, and
    `run(elapsed)` waits until the frame `elapsed` time units after the start is due, so `RealTimeClock(speed=100.0)`
    shows 100 time units per second. A subclass writes the private method `_run`. Clocks pace a live view of a pendulum,
    a demo of a traffic jam, and a game against a model of a market.

    Read more in the user guide, under Observe.

    Attributes:
        started_:
            The wall time at which the run started, in seconds of `time.perf_counter`.

        max_lag_:
            The largest delay of a frame behind its due time, in seconds.

    See Also:
        AsFastAsPossibleClock: Never waits.
        RealTimeClock: Runs at a fixed speed against wall time.

    Examples:
        >>> from skcomplex.core.engine.runtime import BaseClock, RealTimeClock
        >>> clock = RealTimeClock(speed=100.0).prepare()
        >>> isinstance(clock, BaseClock)
        True
    """

    def _prepare(self, context: None) -> None:
        """Mark the wall time at which the run starts."""
        self.started_ = time.perf_counter()
        self.max_lag_ = 0.0

    def _run(self, inputs: float | None) -> None:
        """Wait until a frame that is `inputs` simulated time units after the start is due.

        By default, it never waits.

        Args:
            inputs:
                The simulated time since the start of the run.
        """

    def prepare(self, context: None = None) -> Self:
        """Mark the wall time at which the run of the checked clock starts.

        Args:
            context:
                Nothing: a clock prepares on no context.

        Returns:
            self:
                The prepared clock.
        """
        return super().prepare(context)

    def run(self, elapsed: float | None = None, final: bool = False) -> Self:
        """Wait until the frame `elapsed` simulated time units after the start is due.

        Args:
            elapsed:
                The simulated time since the start of the run.

            final:
                Whether this is the last call of the run.

        Returns:
            self:
                The clock.
        """
        super().run(elapsed, final)
        return self
