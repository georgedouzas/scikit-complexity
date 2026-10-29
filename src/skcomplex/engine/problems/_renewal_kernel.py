"""Renewal law compiled into the draws, resets, and waiting times of every member of its component."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

import numpy as np

from ...errors import SolverError
from ...specs import DrawSpec, RenewalRecordSpec, RunStateSpec
from ...types import Array, Indices
from ..types import Drawn, ReadKind
from ._base import BaseKernel
from ._draw_kernel import DrawKernel
from ._expression_printer import ExpressionPrinter
from ._reference_spec import ReferenceSpec

type MemberFunction = tuple[Callable[..., object], tuple[ReferenceSpec | str, ...]]
"""Function of the states and parameters of renewing members and of the values drawn for them, with its readers."""


type Sampler = tuple[DrawSpec, MemberFunction]
"""DrawSpec of a renewal law with the function of its arguments."""


_READ_SOURCES = frozenset({ReadKind.SINGLE, ReadKind.MEMBER})


class RenewalKernel(BaseKernel):
    """Renewal law compiled into the draws, resets, and waiting times of every member of its component.

    `run(state)` returns the next renewal time of every member, drawn from the state at its time, and the solver that
    keeps the renewal times renews the members that are due, at their own times.

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

    def __init__(
        self,
        label: str,
        members: int,
        counts: Indices,
        states: Indices,
        names: tuple[str, ...],
        *,
        targets: Indices,
        reset: MemberFunction | None,
        waiting: MemberFunction,
        before: tuple[Sampler, ...],
        after: tuple[Sampler, ...],
    ) -> None:
        self.label = label
        self.members = members
        self.counts = counts
        self.states = states
        self.names = names
        self.targets = targets
        self.reset = reset
        self.waiting = waiting
        self.before = before
        self.after = after

    @staticmethod
    def _evaluate(
        function: MemberFunction,
        times: Array,
        x: Array,
        p: Array,
        fired: Indices,
        *,
        drawn: Drawn,
    ) -> list[Array]:
        """Return the value of every expression of a member function for every renewing member."""
        compiled, readers = function
        arguments = []
        for reader in readers:
            if isinstance(reader, str):
                arguments.append(drawn[reader])
                continue
            column = reader.column_of(x, p)
            arguments.append(column[fired] if reader.source is ReadKind.MEMBER else column)
        values = ExpressionPrinter._values_of(compiled(times, *arguments))
        return [np.broadcast_to(np.asarray(value, dtype=float), fired.shape).copy() for value in values]

    def _sample(
        self,
        samplers: tuple[Sampler, ...],
        times: Array,
        x: Array,
        p: Array,
        fired: Indices,
        *,
        drawn: Drawn,
        rng: np.random.Generator,
    ) -> None:
        """DrawSpec a value of every sampler for every renewing member into the drawn values, in order."""
        for draw, arguments in samplers:
            values = self._evaluate(arguments, times, x, p, fired, drawn=drawn) if draw.arguments else []
            drawn[draw.name] = DrawKernel._sample_draw(draw, values, fired.size, rng)

    def _wait(
        self,
        times: Array,
        x: Array,
        p: Array,
        fired: Indices,
        *,
        drawn: Drawn,
        rng: np.random.Generator,
    ) -> Array:
        """Return the next renewal times of members from their states, after drawing the waiting time's draws."""
        self._sample(self.after, times, x, p, fired, drawn=drawn, rng=rng)
        waiting = self._evaluate(self.waiting, times, x, p, fired, drawn=drawn)[0]
        if not np.all(waiting >= 0):
            error_msg = (
                f'The renewal law {self.label} drew the waiting time {waiting[~(waiting >= 0)][0]}, but waiting '
                'times must be nonnegative. DrawSpec them from a distribution with nonnegative support.'
            )
            raise SolverError(error_msg)
        return times + waiting

    def _run(self, inputs: RunStateSpec | None) -> Array:
        """Return the next renewal time of every member, drawn from the state as if every member renewed at its time."""
        state = self._state_of(inputs)
        x, p, rng = state.x, state.p, self._generator_of(state)
        fired = np.arange(self.members, dtype=np.intp)
        times = np.full(self.members, state.t)
        drawn: Drawn = {}
        self._sample(self.before, times, x, p, fired, drawn=drawn, rng=rng)
        return self._wait(times, x, p, fired, drawn=drawn, rng=rng)

    def _fire(
        self,
        times: Array,
        fired: Indices,
        x: Array,
        p: Array,
        rng: np.random.Generator,
    ) -> tuple[Array, Array, RenewalRecordSpec]:
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
        self._sample(self.before, times, x, p, fired, drawn=drawn, rng=rng)
        state = x.copy()
        if self.reset is not None:
            for place, value in zip(
                self.targets,
                self._evaluate(self.reset, times, x, p, fired, drawn=drawn),
                strict=True,
            ):
                state[place[fired]] = value
        state[self.counts[fired]] += 1
        following = self._wait(times, state, p, fired, drawn=drawn, rng=rng)
        record = RenewalRecordSpec(
            self.label,
            times.copy(),
            fired.copy(),
            x[self.states[fired]],
            state[self.states[fired]],
            self.names,
        )
        return state, following, record
