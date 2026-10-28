"""Compile renewal laws into the waiting times and resets of every member, and record their firings."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...core.errors import SolverError
from ...core.laws.utils import Draw
from ...core.types import Indices, Vector
from ..problems import Entry, Reference, RenewalRecord, Source
from ._renewal_law import RenewalLaw
from ._specs import KernelSpec
from ._stochastic import DrawLowering

type Drawn = dict[str, Vector]
_READ_SOURCES = frozenset({Source.SINGLE, Source.MEMBER})


@dataclass(frozen=True)
class _MemberFunction:
    """Expressions of the states and parameters of renewing members and of the values drawn for them."""

    function: Callable[..., list[Any]]
    readers: tuple[Reference | str, ...]

    def evaluate(self, times: Vector, x: Vector, p: Vector, fired: Indices, drawn: Drawn) -> list[Vector]:
        """Return the value of every expression for every renewing member."""
        arguments = []
        for reader in self.readers:
            if isinstance(reader, str):
                arguments.append(drawn[reader])
                continue
            column = reader.column_of(x, p)
            arguments.append(column[fired] if reader.source is Source.MEMBER else column)
        values = self.function(times, *arguments)
        return [np.broadcast_to(np.asarray(value, dtype=float), fired.shape).copy() for value in values]


@dataclass(frozen=True)
class _Sampler:
    """Draw of a renewal law with the function of its arguments."""

    draw: Draw
    arguments: _MemberFunction

    def sample(
        self,
        times: Vector,
        x: Vector,
        p: Vector,
        fired: Indices,
        *,
        drawn: Drawn,
        rng: np.random.Generator,
    ) -> None:
        """Draw a value for every renewing member into the drawn values."""
        values = self.arguments.evaluate(times, x, p, fired, drawn) if self.draw.arguments else []
        drawn[self.draw.name] = DrawLowering._sample_draw(self.draw, values, fired.size, rng)


@dataclass(frozen=True)
class CompiledRenewal:
    """Renewal law compiled into the draws, resets, and waiting times of every member of its component.

    Args:
        label:
            The name of the renewal law.

        members:
            The number of members that renew on their own clocks.

        counts:
            The flat index of the hidden state that counts the renewals of every member.

        states:
            The flat indices of the states of every member that records keep, one row per member.

        names:
            The dotted state name and the component of every column of `states`.

        targets:
            The flat indices the reset writes, one row per reset component and one column per member.

        reset:
            The function of the renewing members that returns the value of every reset component, or `None`.

        waiting:
            The function of the renewing members that returns their next waiting times.

        before:
            The draws of the reset, taken before it.

        after:
            The draws of the waiting time alone, taken after the reset.
    """

    label: str
    members: int
    counts: Indices
    states: Indices
    names: tuple[str, ...]
    targets: Indices
    reset: _MemberFunction | None
    waiting: _MemberFunction
    before: tuple[_Sampler, ...]
    after: tuple[_Sampler, ...]

    def _wait(
        self,
        times: Vector,
        x: Vector,
        p: Vector,
        fired: Indices,
        *,
        drawn: Drawn,
        rng: np.random.Generator,
    ) -> Vector:
        """Return the next renewal times of members from their states, after drawing the waiting time's draws."""
        for sampler in self.after:
            sampler.sample(times, x, p, fired, drawn=drawn, rng=rng)
        waiting = self.waiting.evaluate(times, x, p, fired, drawn)[0]
        if not np.all(waiting >= 0):
            error_msg = (
                f'The renewal law {self.label} drew the waiting time {waiting[~(waiting >= 0)][0]}, but waiting '
                'times must be nonnegative. Draw them from a distribution with nonnegative support.'
            )
            raise SolverError(error_msg)
        return times + waiting

    def start(self, t: float, x: Vector, p: Vector, rng: np.random.Generator) -> Vector:
        """Return the first renewal time of every member from the states at a start time.

        Args:
            t:
                The start time.

            x:
                The state vector.

            p:
                The parameter vector.

            rng:
                The random number generator.

        Returns:
            times:
                The first renewal time of every member.
        """
        fired = np.arange(self.members, dtype=np.intp)
        times = np.full(self.members, t)
        drawn: Drawn = {}
        for sampler in self.before:
            sampler.sample(times, x, p, fired, drawn=drawn, rng=rng)
        return self._wait(times, x, p, fired, drawn=drawn, rng=rng)

    def fire(
        self,
        times: Vector,
        fired: Indices,
        x: Vector,
        p: Vector,
        rng: np.random.Generator,
    ) -> tuple[Vector, Vector, RenewalRecord]:
        """Renew members at their times: draw, reset their states, count the renewal, and draw the next times.

        Args:
            times:
                The renewal time of every renewing member.

            fired:
                The renewing members.

            x:
                The state vector.

            p:
                The parameter vector.

            rng:
                The random number generator.

        Returns:
            state:
                The state vector after the resets.

            following:
                The next renewal time of every renewing member.

            record:
                The record of the renewals.
        """
        drawn: Drawn = {}
        for sampler in self.before:
            sampler.sample(times, x, p, fired, drawn=drawn, rng=rng)
        state = x.copy()
        if self.reset is not None:
            for place, value in zip(self.targets, self.reset.evaluate(times, x, p, fired, drawn), strict=True):
                state[place[fired]] = value
        state[self.counts[fired]] += 1
        following = self._wait(times, state, p, fired, drawn=drawn, rng=rng)
        record = RenewalRecord(
            self.label,
            times.copy(),
            fired.copy(),
            x[self.states[fired]],
            state[self.states[fired]],
            self.names,
        )
        return state, following, record


@dataclass(frozen=True)
class BoundRenewal:
    """RenewalLaw of a component with its label, the group it renews, and the hidden state that counts it.

    Args:
        spec:
            The kernel spec of the component.

        law:
            The renewal law.

        label:
            The label of the renewal.

        group:
            The path of the element or group the renewal resets.

        entry:
            The hidden state that counts the renewals of every member.

        symbol:
            The symbol of the count.
    """

    spec: KernelSpec
    law: RenewalLaw
    label: str
    group: str
    entry: Entry
    symbol: Expression
