"""Compile renewal laws into the waiting times and resets of every member, and record their firings."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import Expression, SolverError, SymbolicVector, ValidationError, components_of, symbol, symbols_of
from ..model import Draw, Renewal, draws_in, has_reductions, is_draw
from ._kernels import Reference, Source
from ._printing import numpy_function
from ._problem import RenewalRecord
from ._stochastic import distribution_of, sample_draw

type Vector = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.intp]
type Drawn = dict[str, Vector]
READ_SOURCES = frozenset({Source.SINGLE, Source.MEMBER})


@dataclass(frozen=True)
class _MemberFunction:
    """Expressions of the states and parameters of renewing members and of the values drawn for them."""

    function: Callable[..., list[Any]]
    readers: tuple[Reference | str, ...]

    def __call__(self, times: Vector, x: Vector, p: Vector, fired: Indices, drawn: Drawn) -> list[Vector]:
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


def _member_function(
    owner: str,
    expressions: Sequence[Expression],
    references: dict[Expression, Reference],
    slots: dict[Expression, str],
    time: Expression,
) -> _MemberFunction:
    """Return the function of renewing members that evaluates expressions, after checking the symbols they read."""
    arguments = sorted(symbols_of(list(expressions)) - {time}, key=str)
    readers: list[Reference | str] = []
    for item in arguments:
        if item in slots:
            readers.append(slots[item])
            continue
        reference = references.get(item)
        if reference is None or reference.source not in READ_SOURCES:
            error_msg = (
                f'A renewal law of {owner} reads {item}, which is not a state or a parameter of its component. Renewal '
                'laws read the states and parameters of the renewing member.'
            )
            raise ValidationError(error_msg)
        readers.append(reference)
    return _MemberFunction(numpy_function(time, arguments, list(expressions)), tuple(readers))


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
        values = self.arguments(times, x, p, fired, drawn) if self.draw.arguments else []
        drawn[self.draw.name] = sample_draw(self.draw, values, fired.size, rng)


def _ordered(found: list[Draw]) -> list[Draw]:
    """Return draws ordered so that every draw comes after the draws in its arguments."""
    ordered: list[Draw] = []
    pending = list(found)
    while pending:
        done = {item.name for item in ordered}
        ready = next(item for item in pending if {inner.name for inner in draws_in(list(item.arguments))} <= done)
        ordered.append(ready)
        pending.remove(ready)
    return ordered


def _declared(owner: str, found: list[Draw]) -> dict[str, Draw]:
    """Return the draws of a renewal law by name, after checking that each name has one declaration."""
    declared: dict[str, Draw] = {}
    for item in found:
        known = declared.setdefault(item.name, item)
        if (known.kind, known.distribution, known.parameters, known.arguments) != (
            item.kind,
            item.distribution,
            item.parameters,
            item.arguments,
        ):
            error_msg = (
                f'A renewal law of {owner} uses the draw name {item.name!r} for {known.call} and for {item.call}. Use '
                'one declaration per name, or give the draws different names.'
            )
            raise ValidationError(error_msg)
    return declared


def _lowest_value(owner: str, description: str, waiting: Expression) -> float:
    """Return the lowest value a waiting time that is a single draw can take, or 0 when it is not known."""
    if not is_draw(waiting):
        return 0.0
    found = Draw.of(waiting)
    numbers = [float(item) if SR(item).is_numeric() else None for item in found.arguments]
    if found.kind == 'normal':
        return -math.inf
    if found.kind == 'uniform':
        return numbers[0] if numbers[0] is not None else 0.0
    if found.kind != 'draw':
        return 0.0
    distribution = distribution_of(owner, description, found)
    values = dict(zip(found.keywords, numbers, strict=True))
    shapes = {key: value for key, value in values.items() if key not in {'loc', 'scale'}}
    lowest = (
        float(distribution.support(**shapes)[0])
        if all(value is not None for value in shapes.values())
        else float(distribution.a)
    )
    if lowest < 0:
        return lowest
    loc = values.get('loc', 0.0)
    return 0.0 if loc is None else loc


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
        waiting = self.waiting(times, x, p, fired, drawn)[0]
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
            for place, value in zip(self.targets, self.reset(times, x, p, fired, drawn), strict=True):
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


def _target_places(
    owner: str,
    targets: list[Expression],
    references: dict[Expression, Reference],
    rows: int,
) -> Indices:
    """Return the flat indices every reset component writes, one row per component and one column per member."""
    places = []
    for item in targets:
        reference = references.get(item)
        if reference is None or not reference.is_state or reference.source not in READ_SOURCES:
            error_msg = f'A renewal law of {owner} resets {item}, which is not a state of an element or a group member.'
            raise ValidationError(error_msg)
        if reference.source is Source.SINGLE and rows > 1:
            error_msg = (
                f'A renewal law of {owner} resets {item}, a state of a single element, while its members renew on '
                'their own clocks. Reset the states of the renewing members.'
            )
            raise ValidationError(error_msg)
        flat = reference.flat_indices if reference.source is Source.MEMBER else np.asarray([reference.index])
        places.append(np.asarray(flat, dtype=np.intp))
    return np.asarray(places, dtype=np.intp).reshape(len(places), rows)


def _recorded_states(group: str, references: dict[Expression, Reference], rows: int) -> tuple[Indices, tuple[str, ...]]:
    """Return the flat indices and the names of the states of every member of the group a renewal law renews."""
    columns: dict[int, tuple[Indices, str]] = {}
    for reference in references.values():
        owned = reference.entry.name.rpartition('.')[0] == group
        if not reference.is_state or reference.source not in READ_SOURCES or not owned:
            continue
        flat = reference.flat_indices if reference.source is Source.MEMBER else np.asarray([reference.index])
        if flat.size != rows:
            continue
        name = reference.entry.name if reference.entry.columns == 1 else f'{reference.entry.name}[{reference.column}]'
        columns[int(flat[0])] = (np.asarray(flat, dtype=np.intp), name)
    ordered = [columns[key] for key in sorted(columns)]
    indices = np.stack([flat for flat, _ in ordered], axis=-1) if ordered else np.zeros((rows, 0), dtype=np.intp)
    return indices, tuple(name for _, name in ordered)


def compile_renewal(
    owner: str,
    law: Renewal,
    references: dict[Expression, Reference],
    *,
    time: Expression,
    rows: int,
    label: str,
    group: str,
    counts: Indices,
) -> CompiledRenewal:
    """Return a renewal law compiled for the members of its component.

    Args:
        owner:
            The path of the component the law belongs to.

        law:
            The renewal law.

        references:
            The references of the symbols of the component.

        time:
            The time symbol.

        rows:
            The number of members that renew on their own clocks.

        label:
            The name of the law.

        group:
            The path of the element or group whose states the law renews.

        counts:
            The flat index of the hidden renewal count of every member.

    Returns:
        renewal:
            The compiled renewal law.

    Raises:
        ValidationError:
            The waiting time can be negative, the law reads a reduction or a symbol of another component, resets a
            symbol that is not a state of the renewing members, or gives one draw name two declarations.
    """
    description = f'renewal law {label}'
    reset = law.reset or {}
    targets = [item for target in reset for item in components_of(target)]
    values = [
        SR(item)
        for target, value in reset.items()
        for item in (
            components_of(value) if isinstance(value, SymbolicVector) else [value] * len(components_of(target))
        )
    ]
    waiting = SR(law.waiting)
    if any(has_reductions(item) for item in [*values, waiting]):
        error_msg = (
            f'The {description} of {owner} reads a reduction over a group. Renewal laws read the states and parameters '
            'of the renewing member.'
        )
        raise ValidationError(error_msg)
    if _lowest_value(owner, description, waiting) < 0:
        error_msg = (
            f'The {description} of {owner} draws its waiting time from {waiting}, which can be negative. Draw waiting '
            "times from a distribution with nonnegative support, such as 'expon', 'pareto', or 'lomax'."
        )
        raise ValidationError(error_msg)
    first = _declared(owner, draws_in(values))
    later = {
        name: item
        for name, item in _declared(owner, [*first.values(), *draws_in(waiting)]).items()
        if name not in first
    }
    slots = {name: symbol(f'_renewal_{name}') for name in [*first, *later]}
    calls = {item.call: slots[item.name] for item in [*first.values(), *later.values()]}
    names = {slot: name for name, slot in slots.items()}

    def samplers(found: dict[str, Draw]) -> tuple[_Sampler, ...]:
        return tuple(
            _Sampler(
                item,
                _member_function(owner, [SR(arg).subs(calls) for arg in item.arguments], references, names, time),
            )
            for item in _ordered(list(found.values()))
        )

    lowered = [item.subs(calls) for item in values]
    return CompiledRenewal(
        label=label,
        members=rows,
        counts=counts,
        states=_recorded_states(group, references, rows)[0],
        names=_recorded_states(group, references, rows)[1],
        targets=_target_places(owner, targets, references, rows),
        reset=_member_function(owner, lowered, references, names, time) if lowered else None,
        waiting=_member_function(owner, [waiting.subs(calls)], references, names, time),
        before=samplers(first),
        after=samplers(later),
    )
