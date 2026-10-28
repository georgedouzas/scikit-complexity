"""Expand flows and delays into laws, and check that conserved quantities change only through flows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Self, cast

import numpy as np
from sage.all__sagemath_symbolics import RR, SR, Expression

from ...core.errors import ConservationError, LabelError, ValidationError
from ...core.laws import (
    BaseLaw,
    BoundaryLaw,
    CrossingLaw,
    JumpLaw,
    NoiseLaw,
    RateLaw,
    Sink,
    Source,
    TransferLaw,
    UpdateLaw,
)
from ...core.laws.utils import delay_parts
from .._utils import declarations_of, symbol, symbols_of
from ..problems import LoopTerm, TransferNetwork, VariableKind
from ..problems._read_kind import ReadKind as Read
from ..problems._reference import Reference
from ..problems._transfer_network import _CompiledDelay, _CompiledFlow, _Recorded
from ._base import BaseLowering, BasePass
from ._block import Block
from ._declarations import Declarations
from ._kernel_spec import KernelSpec

_BOUNDARY_PREFIX = '__boundary'


@dataclass(frozen=True)
class _Record:
    """Recorded expression of a spec, rewritten as later passes substitute hidden states."""

    path: str
    expression: Expression
    spec: KernelSpec


@dataclass(frozen=True)
class _Born:
    """Transfer of a spec with the amount it moves and its compiled holders."""

    spec: KernelSpec
    flow: _CompiledFlow
    amount: Expression
    sign: int


class TransferLowering(BaseLowering):
    """Lowering of flows and delays, which checks that conserved quantities change only through flows.

    `prepare(compilation)` reads the quantity of every tagged state, and `run(laws)` takes the laws of every kernel spec
    and returns them with their flows expanded into laws of other kinds. A later pass finishes the lowering on the laws
    of every spec after their complex states and Markov laws are lowered: it checks the conserved quantities and lowers
    the flows and the delays into the transfer network of the system.

    Attributes:
        quantities_:
            The quantity of every tagged state by entry name.

        born_:
            The expanded flow of every law a flow became, by the identity of the law.

        flows_:
            The expanded flows, in the order they were expanded.

        records_:
            The recorded expressions of the flows, rewritten as later steps substitute hidden states.

        delays_:
            The delays and smoothings lowered to hidden states.

        terms_:
            The flows and rates of every spec, with their sources and sinks, for the loop terms.
    """

    quantities_: dict[str, str]
    born_: dict[int, _Born]
    flows_: list[_Born]
    records_: list[_Record]
    delays_: list[_CompiledDelay]
    terms_: list[tuple[KernelSpec, str, str, Expression, str | None, str | None]]

    def _prepare(self, context: object) -> Self:
        """Read the quantity of every tagged state of the compilation, with no flow expanded yet.

        Args:
            context:
                The compilation.

        Returns:
            self:
                The prepared lowering.
        """
        super()._prepare(context)
        self.quantities_ = self._quantities_of(cast('Declarations', self.compilation_.layout).blocks)
        self.born_, self.flows_, self.records_, self.delays_, self.terms_ = {}, [], [], [], []
        return self

    def _run(self, inputs: object) -> list[tuple[KernelSpec, list[BaseLaw]]]:
        """Return the laws of every kernel spec with flows expanded into laws of the other kinds.

        Args:
            inputs:
                The laws of every kernel spec.

        Returns:
            laws:
                The expanded laws of every kernel spec.

        Raises:
            LabelError:
                Two flows of a component share a label.

            ConservationError:
                A flow connects holders of different quantities.
        """
        return [
            (spec, self._expand(spec, list(laws)))
            for spec, laws in cast('list[tuple[KernelSpec, list[BaseLaw]]]', inputs)
        ]

    @staticmethod
    def _quantities_of(blocks: Mapping[str, Block]) -> dict[str, str]:
        """Return the quantity of every state tagged with one, by entry name."""
        quantities = {}
        for block in blocks.values():
            if block.node.is_agent:
                continue
            for variable, declaration in declarations_of(block.node.component_type, VariableKind.STATE).items():
                if declaration.quantity is not None:
                    quantities[block.entries[variable].name] = declaration.quantity
        return quantities

    def _quantity_of(self, target: object, spec: KernelSpec) -> tuple[str, str | None]:
        """Return the state name and the quantity of a law target."""
        reference = spec.references.get(target)
        name = reference.entry.name if reference is not None else str(target)
        return name, self.quantities_.get(name)

    def _expand_flow(
        self,
        spec: KernelSpec,
        flow: TransferLaw,
        label: str,
    ) -> list[BaseLaw]:
        """Return the rate or jump laws of a flow, after recording it."""
        source, source_label = self._holder(flow.source, spec.references, spec.owner)
        sink, sink_label = self._holder(flow.sink, spec.references, spec.owner)
        tags = {self.quantities_.get(name) for name in (source, sink) if name is not None}
        if len(tags) > 1:
            error_msg = (
                f'The flow {spec.owner}.{label} moves a quantity from {source_label} to {sink_label}, which hold '
                f'different quantities {sorted(map(str, tags))}. A flow connects holders of one quantity.'
            )
            raise ConservationError(error_msg)
        path = f'{spec.owner}.{label}'
        compiled = _CompiledFlow(path, source, sink, source_label, sink_label, next(iter(tags)))
        amount = SR(flow.amount)
        holders = [(item, sign) for item, sign in ((flow.source, -1), (flow.sink, 1)) if isinstance(item, Expression)]
        if flow.rate is not None:
            laws: list[BaseLaw] = [
                JumpLaw(flow.rate, {item: sign * amount for item, sign in holders}),
            ]
        else:
            laws = [RateLaw(item, sign * amount) for item, sign in holders]
        sign = 1 if source is None else -1 if sink is None else 0
        born = _Born(spec, compiled, amount, sign)
        self.flows_.append(born)
        self.born_.update({id(law): born for law in laws})
        self.records_.append(_Record(path, amount, spec))
        self.terms_.append((spec, 'flow', path, amount, source, sink))
        return laws

    def _expand(self, spec: KernelSpec, laws: list[BaseLaw]) -> list[BaseLaw]:
        """Return the laws of a component with flows expanded into laws of the other kinds.

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
                Two flows of the component share a label.

            ConservationError:
                A flow connects holders of different quantities.
        """
        labels: set[str] = set()
        expanded: list[BaseLaw] = []
        for index, law in enumerate(laws):
            if isinstance(law, TransferLaw):
                label = law.label or f'flow_{index}'
                self._claim_label(labels, spec.owner, label)
                expanded.extend(self._expand_flow(spec, law, label))
                continue
            if (
                isinstance(law, RateLaw)
                and isinstance(law.value, Expression)
                and isinstance(law._get_targets()[0], Expression)
            ):
                target, _ = self._quantity_of(law._get_targets()[0], spec)
                self.terms_.append((spec, 'rate', target, law.value, None, None))
            expanded.append(law)
        return expanded

    def _check_conserved(self, spec: KernelSpec, law: BaseLaw) -> None:
        """Raise an error for a law outside flows that changes a state tagged with a quantity."""
        if id(law) in self.born_:
            return
        targets: list[tuple[object, object]] = []
        if isinstance(law, RateLaw | NoiseLaw) and isinstance(law._get_targets()[0], Expression):
            targets = [(law._get_targets()[0], law.value)]
        elif isinstance(law, JumpLaw):
            targets = list(law.changes.items())
        elif isinstance(law, UpdateLaw) and isinstance(law._get_targets()[0], Expression):
            value = law.value
            if isinstance(value, Expression) and bool((value - law._get_targets()[0]).is_trivial_zero()):
                return
            targets = [(law._get_targets()[0], value)]
        for target, value in targets:
            name, quantity = self._quantity_of(target, spec)
            if quantity is not None:
                error_msg = (
                    f'A {type(law).__name__} of {spec.owner} changes {name}, which holds the quantity '
                    f'{quantity!r}, by {value} outside any flow. Move the change into a Transfer between holders of '
                    f'{quantity!r} or with Source() or Sink().'
                )
                raise ConservationError(error_msg)

    def _to_updates(self, specs: list[tuple[KernelSpec, list[BaseLaw]]]) -> None:
        """Rewrite the rate laws of flows of a discrete system as increments of the update laws of their states."""
        increments: dict[tuple[Any, ...], list[tuple[KernelSpec, Expression, Expression]]] = defaultdict(list)
        for spec, laws in specs:
            for law in [law for law in laws if id(law) in self.born_ and isinstance(law, RateLaw)]:
                laws.remove(law)
                target: Any = law._get_targets()[0]
                reference = spec.references[target]
                key: tuple[Any, ...] = (reference.entry.name, reference.column)
                if reference.entry.members is not None:
                    key = (*key, id(spec))
                increments[key].append((spec, target, SR(law.value)))
        for key, items in increments.items():
            host, index = self._update_of(key, specs)
            if host is None:
                host_spec, target = items[0][0], items[0][1]
                host = next(laws for spec, laws in specs if spec is host_spec)
                value: Expression = target
                index = len(host)
                host.append(UpdateLaw(target, target))
            else:
                host_spec = next(spec for spec, laws in specs if laws is host)
                found: Any = host[index]
                target, value = found.targets[0], SR(found.value)
            for spec, _, increment in items:
                self._merge_references(host_spec, spec, increment)
                value += increment
            host[index] = UpdateLaw(target, value)

    def _accumulate(
        self,
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
        declarations: Declarations,
        discrete: bool,
    ) -> dict[str, str]:
        """Add a hidden state per quantity that accumulates the net inflow from the environment."""
        boundary: dict[str, list[_Born]] = defaultdict(list)
        for born in self.flows_:
            if born.sign and born.flow.quantity is not None:
                boundary[born.flow.quantity].append(born)
        laws_of = {id(spec): laws for spec, laws in specs}
        names = {}
        for quantity, items in boundary.items():
            if discrete and any(born.spec.rows > 1 for born in items):
                continue
            name = f'{_BOUNDARY_PREFIX}.{quantity}'
            entry = declarations.declare_state(name, np.zeros(1), RR)
            accumulator = symbol(f'_boundary_{re.sub(r"[^0-9A-Za-z]", "_", quantity)}')
            names[quantity] = name
            for born in items:
                born.spec.references[accumulator] = Reference(entry, True, 0, Read.SINGLE)
            if discrete:
                total = sum((born.sign * born.amount for born in items), SR(0))
                for born in items[1:]:
                    self._merge_references(items[0].spec, born.spec, born.amount)
                laws_of[id(items[0].spec)].append(UpdateLaw(accumulator, accumulator + total))
                continue
            for born in items:
                self._add_boundary(laws_of[id(born.spec)], born, accumulator)
        return names

    def _add_boundary(self, laws: list[BaseLaw], born: _Born, accumulator: Expression) -> None:
        """Add the inflow of a boundary flow to the accumulator of its quantity, as a rate or a jump increment."""
        for index, law in enumerate(laws):
            if isinstance(law, JumpLaw) and self.born_.get(id(law)) is born:
                laws[index] = JumpLaw(law.rate, {**law.changes, accumulator: born.sign * born.amount})
                return
        laws.append(RateLaw(accumulator, born.sign * born.amount))

    def _expand_delays(self, spec: KernelSpec, laws: list[BaseLaw], declarations: Declarations, discrete: bool) -> None:
        """Replace the delays and smoothings in a spec's rate laws and records by hidden states with their own laws."""
        records = [record for record in self.records_ if record.spec is spec]
        counter = 0
        while True:
            values = [law.value for law in laws if isinstance(law, RateLaw)] + [record.expression for record in records]
            calls = [call for value in values for call in self._delay_calls(value)]
            if not calls:
                return
            if discrete or spec.rows > 1:
                error_msg = (
                    f'{spec.owner} uses {calls[0]}, but delays and smoothings add hidden states to systems of rate '
                    'laws of single elements and interactions only.'
                )
                raise ValidationError(error_msg)
            call = calls[0]
            kind = str(call.operator().name())
            value, time = call.operands()
            material, order = self._delay_parts(kind)
            owner = re.sub(r'[^0-9A-Za-z]', '_', spec.owner)
            stages = [symbol(f'_{kind}_{counter}_{owner}_{index}') for index in range(order)]
            start = self._initial(value, spec, declarations)
            each = self._initial(SR(time), spec, declarations) / order if material else 1.0
            entry = declarations.declare_state(
                f'{spec.owner}.{kind}_{counter}',
                np.full(order, start * each),
                RR,
                (order,),
            )
            for column, stage in enumerate(stages):
                spec.references[stage] = Reference(entry, True, column, Read.SINGLE)
            output = stages[-1] / (time / order) if material else stages[-1]
            mapping = {call: output}
            self.delays_.append(_CompiledDelay(call, output, {stage: spec.references[stage] for stage in stages}))
            laws[:] = [self._substituted_law(law, mapping) for law in laws]
            laws.extend(self._delay_laws(kind, stages, value, time))
            self.records_ = [
                replace(record, expression=SR(record.expression).subs(mapping)) if record.spec is spec else record
                for record in self.records_
            ]
            records = [record for record in self.records_ if record.spec is spec]
            counter += 1

    def _lower_flows(self, specs: list[tuple[KernelSpec, list[BaseLaw]]]) -> TransferNetwork:
        """Check the conserved quantities and lower flows and delays into the laws of every spec, in place.

        Args:
            specs:
                The kernel specs with their expanded laws, which the lowering changes in place, extending the layout of
                the compilation with hidden states.

        Returns:
            transfer_network:
                The quantities, flows, records, and loop terms of the system.

        Raises:
            ConservationError:
                A law outside flows changes a state tagged with a quantity.

            ValidationError:
                A delay reads a group or belongs to a discrete system.
        """
        declarations = cast('Declarations', self.compilation_.layout)
        for spec, laws in specs:
            for law in laws:
                self._check_conserved(spec, law)
        discrete = any(isinstance(law, UpdateLaw) and id(law) not in self.born_ for _, laws in specs for law in laws)
        if discrete:
            self._to_updates(specs)
        boundary = self._accumulate(specs, declarations, discrete)
        for spec, laws in specs:
            self._expand_delays(spec, laws, declarations, discrete)
        quantities: dict[str, list[str]] = defaultdict(list)
        for name, quantity in self.quantities_.items():
            quantities[quantity].append(name)
        return TransferNetwork(
            quantities={quantity: tuple(names) for quantity, names in quantities.items()},
            flows=tuple(born.flow for born in self.flows_),
            recorded=tuple(
                _Recorded(record.path, record.expression, dict(record.spec.references)) for record in self.records_
            ),
            boundary=boundary,
            terms=tuple(
                LoopTerm(kind, path, expression, dict(spec.references), source, sink)
                for spec, kind, path, expression, source, sink in self.terms_
            ),
            discrete=discrete,
            delays=tuple(self.delays_),
        )

    @staticmethod
    def _substituted(value: object, mapping: Mapping[Expression, Expression]) -> object:
        """Return a symbolic value with symbols replaced, and any other value unchanged."""
        if not mapping or not hasattr(value, 'subs'):
            return value
        return value.subs(mapping)

    @staticmethod
    def _substituted_law(law: BaseLaw, mapping: Mapping[Expression, Expression]) -> BaseLaw:
        """Return a law with symbols replaced by expressions, such as the calls of delays by their hidden states."""
        if not mapping:
            return law
        if isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
            return BasePass._replaced(law, value=TransferLowering._substituted(law.value, mapping))
        if isinstance(law, JumpLaw):
            return JumpLaw(TransferLowering._substituted(law.rate, mapping), law.changes)
        if isinstance(law, CrossingLaw):
            reset = (
                None
                if law.reset is None
                else {key: TransferLowering._substituted(item, mapping) for key, item in law.reset.items()}
            )
            return BasePass._replaced(law, guard=TransferLowering._substituted(law.guard, mapping), reset=reset)
        return law

    @staticmethod
    def _delay_parts(kind: str) -> tuple[bool, int]:
        """Return whether a delay function conserves its input, and its order, for a name known to be a delay."""
        parts = delay_parts(kind)
        if parts is None:
            error_msg = f'{kind} is not a delay function.'
            raise ValidationError(error_msg)
        return parts

    @staticmethod
    def _delay_laws(kind: str, stages: list[Expression], value: Expression, time: Expression) -> list[BaseLaw]:
        """Return the rate laws of the hidden states of a delay or a smoothing."""
        each = time / len(stages)
        if TransferLowering._delay_parts(kind)[0]:
            inflows = [value, *(stage / each for stage in stages[:-1])]
            return [RateLaw(stage, inflow - stage / each) for stage, inflow in zip(stages, inflows, strict=True)]
        inputs = [value, *stages[:-1]]
        return [RateLaw(stage, (item - stage) / each) for stage, item in zip(stages, inputs, strict=True)]

    @staticmethod
    def _holder(
        holder: object,
        references: Mapping[Expression, Reference],
        owner: str,
    ) -> tuple[str | None, str]:
        """Return the state name of a holder, or `None` for the environment, and its name in flow matrices."""
        if isinstance(holder, Source | Sink):
            return None, holder.label
        reference = references.get(holder)
        if reference is None or not reference.is_state:
            error_msg = (
                f'A flow of {owner} moves a quantity to or from {holder}, which is not a state of its namespace `s`.'
            )
            raise ValidationError(error_msg)
        return reference.entry.name, reference.entry.name

    @staticmethod
    def _update_of(
        key: tuple[Any, ...],
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
    ) -> tuple[list[BaseLaw] | None, int]:
        """Return the laws holding the update law of a state component and its index, or `None`."""
        for spec, laws in specs:
            for index, law in enumerate(laws):
                if not isinstance(law, UpdateLaw) or not isinstance(law._get_targets()[0], Expression):
                    continue
                reference = spec.references.get(law._get_targets()[0])
                if reference is None:
                    continue
                found: tuple[Any, ...] = (reference.entry.name, reference.column)
                if reference.entry.members is not None:
                    found = (*found, id(spec))
                if found == key:
                    return laws, index
        return None, 0

    @staticmethod
    def _merge_references(host: KernelSpec, spec: KernelSpec, value: Expression) -> None:
        """Add the references of an expression's symbols from one spec to another, which lowers it."""
        for item in symbols_of([value]):
            reference = spec.references.get(item)
            if reference is None:
                continue
            known = host.references.get(item)
            if known is not None and known != reference:
                error_msg = (
                    f'A flow of {spec.owner} reads {item} differently from {host.owner}, which updates its state. '
                    'Declare the flow in the component that updates the state.'
                )
                raise ValidationError(error_msg)
            host.references[item] = reference

    @staticmethod
    def _claim_label(labels: set[str], owner: str, label: str) -> None:
        """Add the label of a flow of a component, raising an error for a label it already declares."""
        if label in labels:
            error_msg = f'{owner} declares two flows labelled {label!r}. Rename one.'
            raise LabelError(error_msg)
        labels.add(label)

    @staticmethod
    def _initial(item: Expression, spec: KernelSpec, declarations: Declarations) -> float:
        """Return the value of an expression of a spec's single symbols at the initial state and time zero."""
        states = np.concatenate([np.zeros(0), *declarations.initial_states])
        parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
        values: dict[Expression, float] = {declarations.time: 0.0}
        for name in symbols_of([item]) - {declarations.time}:
            reference = spec.references.get(name)
            if reference is None or reference.source not in {Read.SINGLE, Read.MEMBER} or reference.entry.members:
                error_msg = (
                    f'A delay of {spec.owner} reads {name}, which is not a variable of a single element or '
                    'interaction. Delays and smoothings start from the initial value of their input, read from single '
                    'elements.'
                )
                raise ValidationError(error_msg)
            vector = states if reference.is_state else parameters
            values[name] = float(vector[reference.entry.start + reference.column])
        return float(SR(item).subs(values))

    @staticmethod
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
            if delay_parts(name) is not None and not TransferLowering._delay_calls(item.operands()[0]):
                found.append(item)
            stack.extend(item.operands())
        return found
