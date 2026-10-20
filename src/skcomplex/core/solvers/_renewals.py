"""Keep the next renewal times of every member and fire renewals, alone or between the steps of other solvers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable

import numpy as np

from ..compile import Problem, Vector
from ._hybrid import EventLocator

type Step = Callable[[float, Vector, Vector, Vector], Vector]


class RenewalClock:
    """Next renewal time of every member of every renewal law, with the renewals that fire as time passes.

    Members of one component renew in time order across its renewal laws. Every firing pass draws for all the members
    it renews at once, so the stream of random values depends only on the random state.

    Args:
        problem:
            The compiled problem.

        rng:
            The random number generator.

        locator:
            The event locator whose records receive the records of fired renewals.
    """

    def __init__(
        self,
        problem: Problem,
        rng: np.random.Generator,
        locator: EventLocator,
    ) -> None:
        self.problem = problem
        self.rng = rng
        self.locator = locator
        self.following: list[Vector] = []
        keys = [law.states[:, 0].tobytes() if law.states.size else id(law) for law in problem.renewals]
        self.rivals = [[other for other, key in enumerate(keys) if key == own] for own in keys]

    def begin(self, t: float, x: Vector, p: Vector) -> None:
        """Draw the first renewal time of every member, once, from the states at the start.

        Args:
            t:
                The start time.

            x:
                The state vector.

            p:
                The parameter vector.
        """
        if not self.following:
            self.following = [law.start(t, x, p, self.rng) for law in self.problem.renewals]

    def earliest(self) -> float:
        """Return the earliest pending renewal time of any member.

        Returns:
            time:
                The earliest renewal time, or infinity without renewal laws.
        """
        return min((float(np.min(times)) for times in self.following if times.size), default=math.inf)

    def _fire_pass(self, limits: list[Vector], x: Vector, p: Vector) -> tuple[Vector, bool]:
        """Fire once every member whose next renewal is at most its limit and comes first among its laws."""
        state, fired = x, False
        for index, law in enumerate(self.problem.renewals):
            times = self.following[index]
            first = np.min([self.following[other] for other in self.rivals[index]], axis=0)
            due = np.flatnonzero((times <= limits[index]) & (times <= first))
            if not due.size:
                continue
            state, following, record = law.fire(times[due], due, state, p, self.rng)
            times[due] = following
            self.locator.records.append(record)
            fired = True
        return state, fired

    def fire_until(self, until: float, x: Vector, p: Vector) -> Vector:
        """Fire every renewal up to a time, each member in time order, and return the state after them.

        Args:
            until:
                The last time a renewal fires at.

            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            state:
                The state after the renewals.
        """
        limits = [np.full(law.members, until) for law in self.problem.renewals]
        state, fired = x, True
        while fired:
            state, fired = self._fire_pass(limits, state, p)
        return state

    def _member_stops(self, end: float, size: int) -> Vector:
        """Return the time every state component advances to: its member's next renewal, or the end."""
        stops = np.full(size, end)
        for law, times in zip(self.problem.renewals, self.following, strict=True):
            if law.states.size:
                stops[law.states] = np.minimum(stops[law.states], np.minimum(times, end)[:, np.newaxis])
        return stops

    def advance_members(self, t: float, end: float, x: Vector, p: Vector, step: Step) -> Vector:
        """Advance members that do not interact, each with its own steps that end at its renewals.

        Args:
            t:
                The current time.

            end:
                The time to advance to.

            x:
                The state vector.

            p:
                The parameter vector.

            step:
                The step `step(t, x, p, h)` with one step length per state component.

        Returns:
            state:
                The state at `end`.
        """
        clock = np.full(x.size, t)
        state = x
        while True:
            stops = self._member_stops(end, x.size)
            lengths = stops - clock
            if np.any(lengths > 0):
                state = step(float(np.min(clock)), state, p, lengths)
            clock = stops
            if self.earliest() > end:
                return state
            limits = [
                clock[law.states[:, 0]] if law.states.size else np.full(law.members, end)
                for law in self.problem.renewals
            ]
            state, _ = self._fire_pass(limits, state, p)

    def advance_split(self, t: float, end: float, x: Vector, p: Vector, step: Step) -> Vector:
        """Advance interacting members together, splitting the step at every renewal time inside it.

        Args:
            t:
                The current time.

            end:
                The time to advance to.

            x:
                The state vector.

            p:
                The parameter vector.

            step:
                The step `step(t, x, p, h)` with one step length per state component.

        Returns:
            state:
                The state at `end`.
        """
        time, state = t, x
        while (renewal := self.earliest()) <= end:
            if renewal > time:
                state = step(time, state, p, np.full(x.size, renewal - time))
                time = renewal
            state = self.fire_until(renewal, state, p)
        if end > time:
            state = step(time, state, p, np.full(x.size, end - time))
        return state
