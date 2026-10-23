"""Read delayed states through hidden parameters, take events aside, and compile both with the breakpoints."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ...base import Domain, Expression, SymbolicVector, ValidationError, components_of, symbol, symbols_of, vector
from ...laws import Boundary, Event, Jump, Law, Noise, Rate, Update, values_of
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._hybrid import (
    Breakpoints,
    CompiledEvent,
    Delay,
    Scalars,
    breakpoints_of,
    flat_sources,
    scalar_function,
    state_conditions_of,
)
from .._kernels import Reference, Source
from .._layout import Entry
from .._lowering import calls_of
from .._pipeline import Pass
from .._printing import numpy_function
from .._problem import Vector, VectorFunction
from .._reductions import ReductionPass
from .._specs import KernelSpec


def _delay_minimum(
    spec: KernelSpec,
    tau: Expression,
    minimum: Expression | None,
    declarations: Declarations,
) -> float:
    """Return the smallest value of a delay: the delay itself, its declared minimum, or its value at the parameters."""
    if SR(tau).is_numeric():
        return float(tau)
    if minimum is not None and SR(minimum).is_numeric():
        return float(minimum)
    symbols = sorted(SR(tau).variables(), key=str)
    references = [spec.references.get(item) for item in symbols]
    if symbols and all(reference is not None and not reference.is_state for reference in references):
        values = np.concatenate([np.zeros(0), *declarations.initial_parameters])
        columns = [reference.column_of(np.zeros(0), values) for reference in references if reference is not None]
        delays = numpy_function(declarations.time, symbols, [SR(tau)])(0.0, *columns)[0]
        return float(np.min(delays))
    error_msg = (
        f'A delay of {spec.owner} is the expression {tau}, whose smallest value is unknown. Give it as the third '
        'argument, `delayed(x, tau, minimum)`.'
    )
    raise ValidationError(error_msg)


def _substitute_delays(value: object, spec: KernelSpec, declarations: Declarations, compilation: Compilation) -> object:
    """Return a law value with every delayed state replaced by a hidden parameter that solvers fill from history."""
    if isinstance(value, SymbolicVector):
        return vector(_substitute_delays(item, spec, declarations, compilation) for item in value)
    if not isinstance(value, Expression):
        return value
    replacements = {}
    for call in calls_of(value, 'delayed'):
        arguments = call.operands()
        item, tau = arguments[0], arguments[1]
        reference = spec.references.get(item)
        if reference is None or not reference.is_state or reference.source not in {Source.SINGLE, Source.MEMBER}:
            error_msg = f'{spec.owner} delays {item}, which is not a state of an element or a group member.'
            raise ValidationError(error_msg)
        key = (id(spec), reference.entry.name, reference.column, str(tau))
        if key in compilation.placeholders:
            replacements[call] = compilation.placeholders[key]
            continue
        entry = reference.entry
        members = entry.members if reference.source is Source.MEMBER else None
        indices = (
            entry.start + np.arange(members) * entry.columns + reference.column
            if members is not None
            else np.asarray([entry.start + reference.column])
        )
        initial = np.concatenate([np.zeros(0), *declarations.initial_states])[indices]
        index = len(compilation.delays)
        hidden = Entry(
            f'__delayed_{index}',
            declarations.ends[False],
            (members,) if members is not None else (),
            Domain.REAL,
            members,
        )
        declarations.parameters.append(hidden)
        declarations.ends[False] += hidden.size
        declarations.initial_parameters.append(np.asarray(initial, dtype=float))
        placeholder = symbol(f'_delayed_{index}')
        spec.references[placeholder] = Reference(hidden, False, 0, reference.source)
        compilation.delays.append(
            {
                'entry': hidden,
                'states': indices,
                'tau': tau,
                'minimum': _delay_minimum(
                    spec,
                    tau,
                    next(iter(arguments[2:]), None),
                    declarations,
                ),
                'spec': spec,
                'symbol': placeholder,
                'state': item,
            },
        )
        compilation.placeholders[key] = placeholder
        replacements[call] = placeholder
    return value.subs(replacements) if replacements else value


def _with_delays(spec: KernelSpec, laws: list[Law], declarations: Declarations, compilation: Compilation) -> list[Law]:
    """Return laws with delayed states read through hidden parameters, keeping events aside in the compilation."""
    result: list[Law] = []
    for law in laws:
        if isinstance(law, Event):
            guard = _substitute_delays(SR(law.guard), spec, declarations, compilation)
            reset = {
                target: _substitute_delays(SR(value), spec, declarations, compilation)
                for target, value in (law.reset or {}).items()
            }
            compilation.events.append((spec, Event(guard, reset, law.direction, law.terminal, law.label)))
        elif isinstance(law, Jump):
            result.append(Jump(_substitute_delays(law.rate, spec, declarations, compilation), law.changes))
        elif isinstance(law, Rate | Noise):
            result.append(replace(law, value=_substitute_delays(law.value, spec, declarations, compilation)))
        elif isinstance(law, Update | Boundary):
            result.append(type(law)(law.target, _substitute_delays(law.value, spec, declarations, compilation)))
        else:
            result.append(law)
    return result


def _reset_indices(reference: Reference) -> npt.NDArray[np.intp]:
    """Return the flat indices a reset of a single element's state or of every group member's state writes."""
    if reference.entry.members is not None and reference.source in {Source.MEMBER, Source.REDUCED}:
        return reference.flat_indices.astype(np.intp)
    return np.asarray([reference.entry.start + reference.column], dtype=np.intp)


def _reset_function(
    spec: KernelSpec,
    event: Event,
    time: Expression,
    reductions: ReductionPass,
) -> VectorFunction | None:
    """Return a function that sets the states an event resets, all from the state before the event."""
    reset = event.reset or {}
    targets = [item for target in reset for item in components_of(target)]
    if not targets:
        return None
    values = [
        SR(item)
        for target, value in reset.items()
        for item in (
            components_of(value) if isinstance(value, SymbolicVector) else [value] * len(components_of(target))
        )
    ]
    references = [spec.references.get(item) for item in targets]
    if any(
        item is None or not item.is_state or item.source not in {Source.SINGLE, Source.MEMBER} for item in references
    ):
        error_msg = f'An event of {spec.owner} resets a symbol that is not a state of an element or a group member.'
        raise ValidationError(error_msg)
    arguments = sorted(symbols_of(values) - {time}, key=str)
    unknown = [item for item in arguments if spec.references.get(item) is None]
    if unknown:
        error_msg = f'An event of {spec.owner} uses {", ".join(map(str, unknown))}, which the namespace does not hold.'
        raise ValidationError(error_msg)
    reads = [(spec.references[item].is_state, _reset_indices(spec.references[item])) for item in arguments]
    places = [_reset_indices(reference) for reference in references]  # type: ignore[arg-type]
    function = numpy_function(time, arguments, values)

    def apply(t: float, x: Vector, p: Vector) -> Vector:
        q = reductions.extend(t, x, p)
        results = function(t, *((x if is_state else q)[index] for is_state, index in reads))
        state = x.copy()
        for place, value in zip(places, results, strict=True):
            state[place] = np.broadcast_to(np.asarray(value, dtype=float), place.shape)
        return state

    return apply


def _first_value(function: Scalars) -> Callable[[float, Vector, Vector], float]:
    """Return a function that gives the single value of a scalar function."""

    def first(t: float, x: Vector, p: Vector) -> float:
        return function(t, x, p)[0]

    return first


def compile_events(
    events: list[tuple[KernelSpec, Event]],
    time: Expression,
    reductions: ReductionPass,
) -> tuple[CompiledEvent, ...]:
    """Return the events of a system compiled into guard and reset functions of the state vector.

    Args:
        events:
            The events with the kernel specs of their components.

        time:
            The time symbol.

        reductions:
            The reductions the guards and resets read.

    Returns:
        events:
            The compiled events.
    """
    compiled = []
    counts = Counter(spec.owner for spec, _ in events)
    positions: Counter[str] = Counter()
    for spec, event in events:
        position = positions[spec.owner]
        positions[spec.owner] += 1
        sources = flat_sources(spec.references, spec.owner)
        guard = scalar_function([event.guard], sources, time, spec.owner)
        compiled.append(
            CompiledEvent(
                event.label or (spec.owner if counts[spec.owner] == 1 else f'{spec.owner}:{position}'),
                reductions.wrap(_first_value(guard)),  # type: ignore[arg-type]
                _reset_function(spec, event, time, reductions),
                int(event.direction),
                bool(event.terminal),
            ),
        )
    return tuple(compiled)


def hidden_events(
    specs: list[tuple[KernelSpec, list[Law]]],
    time: Expression,
    reductions: ReductionPass,
) -> tuple[CompiledEvent, ...]:
    """Return one hidden event per condition on the states of single components that switches a law of a flow.

    Args:
        specs:
            The laws of every kernel spec.

        time:
            The time symbol.

        reductions:
            The reductions the conditions read.

    Returns:
        events:
            The hidden events.
    """
    compiled = []
    for spec, laws in specs:
        sources = flat_sources(spec.references, spec.owner)
        values = [
            value
            for law in laws
            if isinstance(law, Law) and not isinstance(law, Jump)
            for value in values_of(law, include=('value', 'lhs', 'rhs'))
            if isinstance(value, Expression)
        ]
        for index, condition in enumerate(state_conditions_of(values, time)):
            if not set(condition.variables()) - {time} <= set(sources):
                continue
            guard = scalar_function([condition], sources, time, spec.owner)
            label = f'{spec.owner}:switch:{index}'
            compiled.append(CompiledEvent(label, reductions.wrap(_first_value(guard)), None, 0, False, hidden=True))  # type: ignore[arg-type]
    return tuple(compiled)


def compile_delays(delays: list[dict[str, Any]], time: Expression) -> tuple[Delay, ...]:
    """Return the delays of a system with their delay functions compiled.

    Args:
        delays:
            The delayed states with their hidden parameters.

        time:
            The time symbol.

    Returns:
        delays:
            The compiled delays.
    """
    compiled = []
    for item in delays:
        spec = item['spec']
        tau = scalar_function([item['tau']], flat_sources(spec.references, spec.owner), time, spec.owner)
        hidden = item['entry']
        compiled.append(
            Delay(
                parameters=hidden.start + np.arange(hidden.size),
                states=np.asarray(item['states'], dtype=np.intp),
                tau=_first_value(tau),
                minimum=float(item['minimum']),
                symbol=item['symbol'],
                state=item['state'],
                delay=SR(item['tau']),
            ),
        )
    return tuple(compiled)


def breakpoints_in(
    specs: list[tuple[KernelSpec, list[Law]]],
    events: list[tuple[KernelSpec, Event]],
    time: Expression,
) -> Breakpoints:
    """Return the breakpoints of the time-dependent inputs in every law and event guard.

    Args:
        specs:
            The laws of every kernel spec.

        events:
            The events with the kernel specs of their components.

        time:
            The time symbol.

    Returns:
        breakpoints:
            The breakpoints.
    """
    expressions: list[Any] = []
    for _, laws in specs:
        for law in laws:
            value = law.rate if isinstance(law, Jump) else getattr(law, 'value', None)
            expressions += (
                [item for item in components_of(value) if isinstance(item, Expression)] if value is not None else []
            )
    expressions += [event.guard for _, event in events]
    return breakpoints_of(expressions, time)


class DelaysPass(Pass):
    """Pass that reads delayed states through hidden parameters and takes the events of the laws aside."""

    reads = ('ordered', 'layout')
    writes = ('delayed', 'delays', 'placeholders', 'events')

    def run(self, compilation: Compilation) -> None:
        """Lower the delays of the ordered laws and take their events aside.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        compilation.delayed = [
            (spec, _with_delays(spec, laws, declarations, compilation))
            for spec, laws in cast('Laws', compilation.ordered)
        ]
