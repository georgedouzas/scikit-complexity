"""Locates events between two times of a step, applies resets, and records them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from scipy.optimize import brentq

from ...errors import SolverError
from ...specs import BaseRecordSpec, EventRecordSpec, RunStateSpec
from ...types import Array
from ..problems import ProblemSpec
from ..problems._event_kernel import EventKernel
from ..types import Interpolant

SEPARATION = 1e-10


class EventLocator:
    """Locates events between two times of a step, applies resets, and records them.

    Args:
        problem:
            The compiled problem.
    """

    def __init__(self, problem: ProblemSpec) -> None:
        self.problem = problem
        self.records: list[BaseRecordSpec] = []
        self.stopped_at: float | None = None
        self._last: dict[tuple[str, int | None], float] = {}
        self._slots = [
            (index, member)
            for index, event in enumerate(problem.events)
            for member in (range(event.members) if event.members else (None,))
        ]
        self._slot_of = {slot: position for position, slot in enumerate(self._slots)}

    @staticmethod
    def _member_event(event: EventKernel, member: int) -> EventKernel:
        """Return the event of one member of a group, whose guard and reset are those of the member."""
        reset = event.reset

        def guard(t: float, x: Array, p: Array) -> float:
            return float(np.asarray(event.guard(t, x, p))[member])

        def apply(t: float, x: Array, p: Array) -> Array:
            return x.copy() if reset is None else reset(t, x, p, member)

        return EventKernel(event.label, guard, apply, event.direction, event.terminal).check().prepare()

    @staticmethod
    def _crossed(direction: int, g0: Array, g1: Array) -> Array:
        """Return whether guards cross zero in their direction between two values, one per member."""
        upwards = (g0 < 0) & (g1 >= 0)
        downwards = (g0 > 0) & (g1 <= 0)
        return np.asarray((direction >= 0) & upwards | (direction <= 0) & downwards)

    def crossing(
        self,
        t0: float,
        t1: float,
        x0: Array,
        x1: Array,
        *,
        p: Array,
        interpolant: Interpolant,
        include_hidden: bool = True,
    ) -> tuple[float, int] | None:
        """Return the earliest event crossing in a step, with its time and the index of its event.

        An event of the members of a group crosses once per member, and each member has its own index.

        Args:
            t0:
                The start of the step.

            t1:
                The end of the step.

            x0:
                The state at the start.

            x1:
                The state at the end.

            p:
                The parameter vector.

            interpolant:
                The state as a function of time within the step.

            include_hidden:
                Whether to locate hidden events, which only restart adaptive integrators at switches of the drift.
                Fixed-step integrators step over those switches, so they leave them out.

        Returns:
            crossing:
                The time and the event index, or `None` when no guard crosses zero in its direction.
        """
        earliest: tuple[float, int] | None = None
        for index, event in enumerate(self.problem.events):
            if event.hidden and not include_hidden:
                continue
            g0 = np.atleast_1d(np.asarray(event.guard(t0, x0, p), dtype=float))
            g1 = np.atleast_1d(np.asarray(event.guard(t1, x1, p), dtype=float))
            for member in np.flatnonzero(self._crossed(event.direction, g0, g1)):
                slot = int(member) if event.members else None
                single = event if slot is None else self._member_event(event, slot)
                time = (
                    t1
                    if g1[member] == 0
                    else self._past_root(single, float(g0[member]), t0, t1, p=p, interpolant=interpolant)
                )
                if earliest is None or time < earliest[0]:
                    earliest = (time, self._slot_of[(index, slot)])
        return earliest

    @staticmethod
    def _past_root(
        event: EventKernel,
        g0: float,
        t0: float,
        t1: float,
        *,
        p: Array,
        interpolant: Interpolant,
    ) -> float:
        """Return the root of a guard in a step, moved just past it so the guard has already changed sign there."""

        def guard(s: float) -> float:
            return float(event.guard(s, interpolant(s), p))

        time = brentq(guard, t0, t1, xtol=1e-13, rtol=1e-12)
        nudge = 1e-15 * max(1.0, abs(time))
        while time < t1 and np.sign(guard(time)) == np.sign(g0):
            time, nudge = min(t1, time + nudge), 2 * nudge
        return time

    def apply(self, time: float, index: int, state: Array, p: Array) -> Array:
        """Apply an event's reset at its time, record it, and stop the run when it is terminal.

        Args:
            time:
                The time of the crossing.

            index:
                The index of the event, as the crossing returns it.

            state:
                The state at the crossing.

            p:
                The parameter vector.

        Returns:
            state:
                The state after the reset.

        Raises:
            SolverError:
                The event fires again within the minimum separation, which signals chattering.
        """
        position, member = self._slots[index]
        event = self.problem.events[position]
        if event.hidden:
            return state.copy()
        last = self._last.get((event.label, member))
        if last is not None and time - last < SEPARATION * max(1.0, abs(time)):
            error_msg = (
                f'Event {event.label} fired again at t = {time} right after its previous firing, which signals '
                'chattering. Change the reset so that the guard leaves zero, or end the run with a terminal event.'
            )
            raise SolverError(error_msg)
        self._last[(event.label, member)] = time
        single = event if member is None else self._member_event(event, member)
        after = single.run(RunStateSpec(time, state, p)).values_
        self.records.append(EventRecordSpec(event.label, time, state.copy(), after.copy(), member))
        if event.terminal:
            self.stopped_at = time
        return after

    @staticmethod
    def _find_bounds(problem: ProblemSpec, t: float, until: float) -> list[float]:
        """Return the breakpoints between two times followed by the end time.

        Args:
            problem:
                The compiled problem.

            t:
                The start time.

            until:
                The end time.

        Returns:
            bounds:
                The times a step must not cross, in increasing order.
        """
        return [*problem.breakpoints.between(t, until), until]

    @staticmethod
    def _find_left_limit(bound: float) -> float:
        """Return the last time before a bound, where inputs that jump at the bound still take their earlier value.

        Args:
            bound:
                The bound.

        Returns:
            time:
                The largest float below the bound.
        """
        return float(np.nextafter(bound, -math.inf))

    @staticmethod
    def _find_smallest_delay(problem: ProblemSpec) -> float:
        """Return the smallest delay of a problem, or infinity without delays.

        Args:
            problem:
                The compiled problem.

        Returns:
            delay:
                The smallest delay.
        """
        return min((delay.minimum for delay in problem.delays), default=math.inf)
