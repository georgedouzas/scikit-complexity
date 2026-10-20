"""Expand flows, auxiliaries, and delays into laws, and check that conserved quantities change only through flows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import (
    ConservationError,
    Domain,
    Expression,
    LabelError,
    ValidationError,
    VariableKind,
    declarations_of,
    symbol,
    symbols_of,
)
from ..model import Auxiliary, Boundary, Event, Flow, Jump, Law, Noise, Rate, Sink, Update
from ..model import Source as Environment
from ..model._flows import DELAY_ORDERS, MATERIAL_DELAYS
from ._kernels import Reference
from ._kernels import Source as Read
from ._layout import Entry
from ._quantities import CompiledFlow, LoopTerm, Recorded, StockFlow

BOUNDARY_PREFIX = '__boundary'

type Vector = npt.NDArray[np.float64]


class _Spec(Protocol):
    """Kernel spec of a component as the stock-flow pass reads it."""

    owner: str
    references: dict[Expression, Reference]
    rows: int


class _Declared(Protocol):
    """Declarations of a compilation as the stock-flow pass extends them with hidden stocks."""

    time: Expression
    states: list[Entry]
    parameters: list[Entry]
    initial_states: list[Vector]
    initial_parameters: list[Vector]


@dataclass
class _Record:
    """Recorded expression of a spec, rewritten as later passes substitute hidden stocks."""

    path: str
    expression: Expression
    spec: _Spec


@dataclass
class _Born:
    """Flow of a spec with the amount it moves and its compiled holders."""

    spec: _Spec
    flow: CompiledFlow
    amount: Expression
    sign: int


def _substituted(value: object, mapping: Mapping[Expression, Expression]) -> object:
    """Return a symbolic value with symbols replaced, and any other value unchanged."""
    if not mapping or not hasattr(value, 'subs'):
        return value
    return value.subs(mapping)


def _with_auxiliaries(law: Law, mapping: Mapping[Expression, Expression]) -> Law:
    """Return a law with the symbols of auxiliaries replaced by their expressions."""
    if not mapping:
        return law
    if isinstance(law, Rate | Noise | Update | Boundary):
        return dataclasses.replace(law, value=_substituted(law.value, mapping))
    if isinstance(law, Jump):
        return Jump(_substituted(law.rate, mapping), law.changes)
    if isinstance(law, Event):
        reset = None if law.reset is None else {key: _substituted(item, mapping) for key, item in law.reset.items()}
        return dataclasses.replace(law, guard=_substituted(law.guard, mapping), reset=reset)
    return law


def _delay_calls(expression: object) -> list[Expression]:
    """Return the calls of delay functions in an expression whose inputs hold no other delay."""
    if not isinstance(expression, Expression):
        return []
    found, stack = [], [expression]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        name = getattr(item.operator(), 'name', lambda: '')()
        if name in DELAY_ORDERS and not _delay_calls(item.operands()[0]):
            found.append(item)
        stack.extend(item.operands())
    return found


def _initial(item: Expression, spec: _Spec, declarations: _Declared) -> float:
    """Return the value of an expression of a spec's single symbols at the initial state and time zero."""
    states = np.concatenate([np.zeros(0), *declarations.initial_states])
    parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
    values: dict[Expression, float] = {declarations.time: 0.0}
    for name in symbols_of([item]) - {declarations.time}:
        reference = spec.references.get(name)
        if reference is None or reference.source not in {Read.SINGLE, Read.MEMBER} or reference.entry.members:
            error_msg = (
                f'A delay of {spec.owner} reads {name}, which is not a variable of a single element or interaction. '
                'Delays and smoothings start from the initial value of their input, read from single elements.'
            )
            raise ValidationError(error_msg)
        vector = states if reference.is_state else parameters
        values[name] = float(vector[reference.entry.start + reference.column])
    return float(SR(item).subs(values))


def _delay_laws(kind: str, stages: list[Expression], value: Expression, time: Expression) -> list[Law]:
    """Return the rate laws of the hidden stocks of a delay or a smoothing."""
    each = time / len(stages)
    if kind in MATERIAL_DELAYS:
        inflows = [value, *(stage / each for stage in stages[:-1])]
        return [Rate(stage, inflow - stage / each) for stage, inflow in zip(stages, inflows, strict=True)]
    inputs = [value, *stages[:-1]]
    return [Rate(stage, (item - stage) / each) for stage, item in zip(stages, inputs, strict=True)]


def _holder(
    holder: object,
    references: Mapping[Expression, Reference],
    owner: str,
) -> tuple[str | None, str]:
    """Return the state name of a holder, or `None` for the environment, and its name in flow matrices."""
    if isinstance(holder, Environment | Sink):
        return None, holder.label
    reference = references.get(holder)
    if reference is None or not reference.is_state:
        error_msg = (
            f'A flow of {owner} moves a quantity to or from {holder}, which is not a state of its namespace `s`.'
        )
        raise ValidationError(error_msg)
    return reference.entry.name, reference.entry.name


class StockFlowPass:
    """Pass of a compilation that expands flows, auxiliaries, and delays, and checks conserved quantities.

    Args:
        quantities:
            The quantity of every tagged state by entry name.
    """

    def __init__(self, quantities: Mapping[str, str]) -> None:
        self.quantities = quantities
        self._born: dict[int, _Born] = {}
        self._flows: list[_Born] = []
        self._records: list[_Record] = []
        self._terms: list[tuple[_Spec, str, str, Expression, dict[Expression, str], str | None, str | None]] = []

    def _quantity_of(self, target: object, spec: _Spec) -> tuple[str, str | None]:
        """Return the state name and the quantity of a law target."""
        reference = spec.references.get(target)
        name = reference.entry.name if reference is not None else str(target)
        return name, self.quantities.get(name)

    def _expand_flow(
        self,
        spec: _Spec,
        flow: Flow,
        label: str,
        mapping: Mapping[Expression, Expression],
        names: dict[Expression, str],
    ) -> list[Law]:
        """Return the rate or jump laws of a flow, after recording it."""
        source, source_label = _holder(flow.source, spec.references, spec.owner)
        sink, sink_label = _holder(flow.sink, spec.references, spec.owner)
        tags = {self.quantities.get(name) for name in (source, sink) if name is not None}
        if len(tags) > 1:
            error_msg = (
                f'The flow {spec.owner}.{label} moves a quantity from {source_label} to {sink_label}, which hold '
                f'different quantities {sorted(map(str, tags))}. A flow connects holders of one quantity.'
            )
            raise ConservationError(error_msg)
        path = f'{spec.owner}.{label}'
        compiled = CompiledFlow(path, source, sink, source_label, sink_label, next(iter(tags)))
        amount = SR(_substituted(flow.amount, mapping))
        holders = [(item, sign) for item, sign in ((flow.source, -1), (flow.sink, 1)) if isinstance(item, Expression)]
        if flow.rate is not None:
            laws: list[Law] = [Jump(_substituted(flow.rate, mapping), {item: sign * amount for item, sign in holders})]
        else:
            laws = [Rate(item, sign * amount) for item, sign in holders]
        sign = 1 if source is None else -1 if sink is None else 0
        born = _Born(spec, compiled, amount, sign)
        self._flows.append(born)
        self._born.update({id(law): born for law in laws})
        self._records.append(_Record(path, amount, spec))
        self._terms.append((spec, 'flow', path, SR(flow.amount), names, source, sink))
        return laws

    def expand(self, spec: _Spec, laws: list[Law]) -> list[Law]:
        """Return the laws of a component with flows and auxiliaries expanded into laws of the other kinds.

        Args:
            spec:
                The kernel spec of the component.

            laws:
                The laws of the component.

        Returns:
            laws:
                The expanded laws.

        Raises:
            LabelError:
                Two flows or auxiliaries of the component share a label.

            ConservationError:
                A flow connects holders of different quantities.
        """
        mapping: dict[Expression, Expression] = {}
        names: dict[Expression, str] = {}
        labels: set[str] = set()
        expanded: list[Law] = []
        for index, law in enumerate(laws):
            if isinstance(law, Auxiliary | Flow):
                label = law.label or f'flow_{index}'
                if label in labels:
                    error_msg = f'{spec.owner} declares two flows or auxiliaries labelled {label!r}. Rename one.'
                    raise LabelError(error_msg)
                labels.add(label)
            if isinstance(law, Auxiliary):
                value = SR(_substituted(law.expression, mapping))
                path = f'{spec.owner}.{law.label}'
                self._records.append(_Record(path, value, spec))
                self._terms.append((spec, 'auxiliary', path, SR(law.expression), names, None, None))
                mapping[law.symbol], names[law.symbol] = value, path
            elif isinstance(law, Flow):
                expanded.extend(self._expand_flow(spec, law, label, mapping, names))
            else:
                if isinstance(law, Rate) and isinstance(law.value, Expression) and isinstance(law.target, Expression):
                    target, _ = self._quantity_of(law.target, spec)
                    self._terms.append((spec, 'rate', target, law.value, names, None, None))
                expanded.append(_with_auxiliaries(law, mapping))
        return expanded

    def _check_conserved(self, spec: _Spec, law: Law) -> None:
        """Raise an error for a law outside flows that changes a state tagged with a quantity."""
        if id(law) in self._born:
            return
        targets: list[tuple[object, object]] = []
        if isinstance(law, Rate | Noise) and isinstance(law.target, Expression):
            targets = [(law.target, law.value)]
        elif isinstance(law, Jump):
            targets = list(law.changes.items())
        elif isinstance(law, Update) and isinstance(law.target, Expression):
            value = law.value
            if isinstance(value, Expression) and bool((value - law.target).is_trivial_zero()):
                return
            targets = [(law.target, value)]
        for target, value in targets:
            name, quantity = self._quantity_of(target, spec)
            if quantity is not None:
                error_msg = (
                    f'A {type(law).__name__} law of {spec.owner} changes {name}, which holds the quantity '
                    f'{quantity!r}, by {value} outside any flow. Move the change into a Flow between holders of '
                    f'{quantity!r} or with Source() or Sink().'
                )
                raise ConservationError(error_msg)

    def _to_updates(self, specs: list[tuple[_Spec, list[Law]]]) -> None:
        """Rewrite the rate laws of flows of a discrete system as increments of the update laws of their states."""
        increments: dict[tuple[Any, ...], list[tuple[_Spec, Expression, Expression]]] = defaultdict(list)
        for spec, laws in specs:
            for law in [law for law in laws if id(law) in self._born and isinstance(law, Rate)]:
                laws.remove(law)
                target: Any = law.target
                reference = spec.references[target]
                key: tuple[Any, ...] = (reference.entry.name, reference.column)
                if reference.entry.members is not None:
                    key = (*key, id(spec))
                increments[key].append((spec, target, SR(law.value)))
        for key, items in increments.items():
            host, index = _update_of(key, specs)
            if host is None:
                host_spec, target = items[0][0], items[0][1]
                host = next(laws for spec, laws in specs if spec is host_spec)
                value: Expression = target
                index = len(host)
                host.append(Update(target, target))
            else:
                host_spec = next(spec for spec, laws in specs if laws is host)
                found: Any = host[index]
                target, value = found.target, SR(found.value)
            for spec, _, increment in items:
                _merge_references(host_spec, spec, increment)
                value += increment
            host[index] = Update(target, value)

    def _accumulate(
        self,
        specs: list[tuple[_Spec, list[Law]]],
        declarations: _Declared,
        discrete: bool,
    ) -> dict[str, str]:
        """Add a hidden state per quantity that accumulates the net inflow from the environment."""
        boundary: dict[str, list[_Born]] = defaultdict(list)
        for born in self._flows:
            if born.sign and born.flow.quantity is not None:
                boundary[born.flow.quantity].append(born)
        laws_of = {id(spec): laws for spec, laws in specs}
        names = {}
        for quantity, items in boundary.items():
            if discrete and any(born.spec.rows > 1 for born in items):
                continue
            name = f'{BOUNDARY_PREFIX}.{quantity}'
            entry = Entry(name, sum(item.size for item in declarations.states), (), Domain.REAL)
            declarations.states.append(entry)
            declarations.initial_states.append(np.zeros(1))
            accumulator = symbol(f'_boundary_{re.sub(r"[^0-9A-Za-z]", "_", quantity)}')
            names[quantity] = name
            for born in items:
                born.spec.references[accumulator] = Reference(entry, True, 0, Read.SINGLE)
            if discrete:
                total = sum((born.sign * born.amount for born in items), SR(0))
                for born in items[1:]:
                    _merge_references(items[0].spec, born.spec, born.amount)
                laws_of[id(items[0].spec)].append(Update(accumulator, accumulator + total))
                continue
            for born in items:
                self._add_boundary(laws_of[id(born.spec)], born, accumulator)
        return names

    def _add_boundary(self, laws: list[Law], born: _Born, accumulator: Expression) -> None:
        """Add the inflow of a boundary flow to the accumulator of its quantity, as a rate or a jump increment."""
        for index, law in enumerate(laws):
            if isinstance(law, Jump) and self._born.get(id(law)) is born:
                laws[index] = Jump(law.rate, {**law.changes, accumulator: born.sign * born.amount})
                return
        laws.append(Rate(accumulator, born.sign * born.amount))

    def _expand_delays(self, spec: _Spec, laws: list[Law], declarations: _Declared, discrete: bool) -> None:
        """Replace the delays and smoothings in a spec's rate laws and records by hidden stocks with their own laws."""
        records = [record for record in self._records if record.spec is spec]
        counter = 0
        while True:
            values = [law.value for law in laws if isinstance(law, Rate)] + [record.expression for record in records]
            calls = [call for value in values for call in _delay_calls(value)]
            if not calls:
                return
            if discrete or spec.rows > 1:
                error_msg = (
                    f'{spec.owner} uses {calls[0]}, but delays and smoothings add hidden stocks to systems of rate '
                    'laws of single elements and interactions only.'
                )
                raise ValidationError(error_msg)
            call = calls[0]
            kind = str(call.operator().name())
            value, time = call.operands()
            order = DELAY_ORDERS[kind]
            entry = Entry(
                f'{spec.owner}.{kind}_{counter}',
                sum(item.size for item in declarations.states),
                (order,),
                Domain.REAL,
            )
            owner = re.sub(r'[^0-9A-Za-z]', '_', spec.owner)
            stages = [symbol(f'_{kind}_{counter}_{owner}_{index}') for index in range(order)]
            start = _initial(value, spec, declarations)
            each = _initial(SR(time), spec, declarations) / order if kind in MATERIAL_DELAYS else 1.0
            declarations.states.append(entry)
            declarations.initial_states.append(np.full(order, start * each))
            for column, stage in enumerate(stages):
                spec.references[stage] = Reference(entry, True, column, Read.SINGLE)
            output = stages[-1] / (time / order) if kind in MATERIAL_DELAYS else stages[-1]
            mapping = {call: output}
            laws[:] = [_with_auxiliaries(law, mapping) for law in laws]
            laws.extend(_delay_laws(kind, stages, value, time))
            for record in records:
                record.expression = SR(record.expression).subs(mapping)
            counter += 1

    def lower(self, specs: list[tuple[_Spec, list[Law]]], declarations: _Declared) -> StockFlow:
        """Check the conserved quantities and lower flows and delays into the laws of every spec, in place.

        Args:
            specs:
                The kernel specs with their expanded laws, which the pass changes in place.

            declarations:
                The declarations, which the pass extends with hidden stocks.

        Returns:
            stock_flow:
                The quantities, flows, records, and loop terms of the system.

        Raises:
            ConservationError:
                A law outside flows changes a state tagged with a quantity.

            ValidationError:
                A delay reads a group or belongs to a discrete system.
        """
        for spec, laws in specs:
            for law in laws:
                self._check_conserved(spec, law)
        discrete = any(isinstance(law, Update) and id(law) not in self._born for _, laws in specs for law in laws)
        if discrete:
            self._to_updates(specs)
        boundary = self._accumulate(specs, declarations, discrete)
        for spec, laws in specs:
            self._expand_delays(spec, laws, declarations, discrete)
        quantities: dict[str, list[str]] = defaultdict(list)
        for name, quantity in self.quantities.items():
            quantities[quantity].append(name)
        return StockFlow(
            quantities={quantity: tuple(names) for quantity, names in quantities.items()},
            flows=tuple(born.flow for born in self._flows),
            recorded=tuple(
                Recorded(record.path, record.expression, dict(record.spec.references)) for record in self._records
            ),
            boundary=boundary,
            terms=tuple(
                LoopTerm(kind, path, expression, dict(names), dict(spec.references), source, sink)
                for spec, kind, path, expression, names, source, sink in self._terms
            ),
            discrete=discrete,
        )


def _update_of(key: tuple[Any, ...], specs: list[tuple[_Spec, list[Law]]]) -> tuple[list[Law] | None, int]:
    """Return the laws holding the update law of a state component and its index, or `None`."""
    for spec, laws in specs:
        for index, law in enumerate(laws):
            if not isinstance(law, Update) or not isinstance(law.target, Expression):
                continue
            reference = spec.references.get(law.target)
            if reference is None:
                continue
            found: tuple[Any, ...] = (reference.entry.name, reference.column)
            if reference.entry.members is not None:
                found = (*found, id(spec))
            if found == key:
                return laws, index
    return None, 0


def _merge_references(host: _Spec, spec: _Spec, value: Expression) -> None:
    """Add the references of an expression's symbols from one spec to another, which lowers it."""
    for item in symbols_of([value]):
        reference = spec.references.get(item)
        if reference is None:
            continue
        known = host.references.get(item)
        if known is not None and known != reference:
            error_msg = (
                f'A flow of {spec.owner} reads {item} differently from {host.owner}, which updates its state. Declare '
                'the flow in the component that updates the state.'
            )
            raise ValidationError(error_msg)
        host.references[item] = reference


def quantities_of(blocks: Mapping[str, Any]) -> dict[str, str]:
    """Return the quantity of every state tagged with one, by entry name.

    Args:
        blocks:
            The blocks of the declared components by path.

    Returns:
        quantities:
            The quantity of every tagged state.
    """
    quantities = {}
    for block in blocks.values():
        if block.node.is_agent:
            continue
        for variable, declaration in declarations_of(block.node.component_type, VariableKind.STATE).items():
            if declaration.quantity is not None:
                quantities[block.entries[variable].name] = declaration.quantity
    return quantities
