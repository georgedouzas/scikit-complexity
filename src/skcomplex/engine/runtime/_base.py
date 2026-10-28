"""Define the bases of the runtime: observers, which receive the frames of a simulation, and clocks, which pace them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import time
from typing import Self

from ...base import BaseConfig
from ..problems import BaseProblem
from ._frames import Frame


class BaseObserver(BaseConfig):
    """Base class of objects that receive the frames of a simulation.

    An observer follows the common API of the engine. A simulation checks it, calls `prepare(problem)` with the
    compiled problem before the first frame, `run(frame)` with every frame, and `run(final=True)` without a frame once
    the run stops, which releases what preparing acquired. Subclasses override any of the hooks `_prepare`, `_run`,
    and `_release`.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import BaseObserver, Simulation
        >>> from skcomplex.models.biology.ecology import PopulationElement
        >>> class CountingObserver(BaseObserver):
        ...     def _prepare(self, problem):
        ...         self.count_ = 0
        ...
        ...     def _run(self, frame=None):
        ...         self.count_ += frame is not None
        >>> observer = CountingObserver().check()
        >>> _ = Simulation(System(elements=[PopulationElement('a')]), dt=1, observers=[observer]).check().run(until=3)
        >>> observer.count_
        4
    """

    def _prepare(self, problem: BaseProblem) -> None:
        """Prepare for a run of the compiled problem; prepare nothing by default."""

    def _run(self, frame: Frame | None = None) -> None:
        """React to a frame, or to the end of the run without one; do nothing by default."""

    def prepare(self, problem: BaseProblem) -> Self:
        """Prepare the checked observer for a run of a compiled problem.

        Args:
            problem:
                The compiled problem the run advances.

        Returns:
            self:
                The prepared observer.
        """
        return super().prepare(problem)

    def run(self, frame: Frame | None = None, *, final: bool = False) -> Self:
        """React to a frame, and release what preparing acquired at the last call.

        Args:
            frame:
                The frame, or `None` for the call that ends the run.

            final:
                Whether this is the last call of the run.

        Returns:
            self:
                The observer.
        """
        super().run(frame, final=final)
        return self


class BaseClock(BaseConfig):
    """Base class of clocks, which decide how long to wait between frames.

    A clock follows the common API of the engine. It is checked and prepared at the start of a run with `prepare()`,
    which marks the wall time, and `run(elapsed)` waits until a frame that is `elapsed` simulated time units after the
    start is due. Subclasses implement `_run`.

    Attributes:
        started_:
            The wall time at which the run started, in seconds of `time.perf_counter`.

        max_lag_:
            The largest delay of a frame behind its due time, in seconds.
    """

    def _prepare(self, context: None = None) -> None:
        """Mark the wall time at which the run starts."""
        self.started_ = time.perf_counter()
        self.max_lag_ = 0.0

    def _run(self, elapsed: float) -> None:
        """Wait until a frame that is `elapsed` simulated time units after the start is due; never wait by default.

        Args:
            elapsed:
                The simulated time since the start of the run.
        """

    def prepare(self) -> Self:
        """Mark the wall time at which the run of the checked clock starts.

        Returns:
            self:
                The prepared clock.
        """
        return super().prepare()

    def run(self, elapsed: float) -> Self:
        """Wait until a frame that is `elapsed` simulated time units after the start is due.

        Args:
            elapsed:
                The simulated time since the start of the run.

        Returns:
            self:
                The clock.
        """
        super().run(elapsed)
        return self
