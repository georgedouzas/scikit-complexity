"""Define the bases of the runtime.

Observers receive the frames of a simulation, and clocks pace them.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import time
from typing import Self

from ...base import BaseConfig
from ..problems import BaseProblemSpec, FrameSpec


class BaseObserver(BaseConfig[BaseProblemSpec, FrameSpec]):
    """Base class of objects that receive the frames of a simulation.

    An observer follows the common API of the engine. A simulation checks it, calls `prepare(problem)` with the
    compiled problem before the first frame, `run(frame)` with every frame, and `run(final=True)` without a frame once
    the run stops, which releases what preparing acquired. Subclasses override any of the hooks `_prepare`, `_run`,
    and `_release`.

    Examples:
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import BaseObserver, simulate
        >>> from skcomplex.systems.biology.common.populations import PopulationElement
        >>> class CountingObserver(BaseObserver):
        ...     def _prepare(self, context):
        ...         self.count_ = 0
        ...
        ...     def _run(self, inputs):
        ...         self.count_ += inputs is not None
        >>> observer = CountingObserver()
        >>> _ = simulate(System(elements=[PopulationElement('a')]), until=3, dt=1, observers=[observer])
        >>> observer.count_
        4
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
    """Base class of clocks.

    A clock decides how long to wait between frames. A clock follows the common API of the engine. It is prepared at the
    start of a run with `prepare()`, which marks the wall time, and `run(elapsed)` waits until a frame that is `elapsed`
    simulated time units after the start is due. Subclasses implement `_run`.

    Attributes:
        started_:
            The wall time at which the run started, in seconds of `time.perf_counter`.

        max_lag_:
            The largest delay of a frame behind its due time, in seconds.
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
