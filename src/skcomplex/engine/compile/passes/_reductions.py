"""Replace the reductions and neighbourhood reductions of laws by placeholders, compiled once each."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import RR, SR, Expression, I
from sage.modules.free_module_element import FreeModuleElement

from ....core.errors import RoleError, ValidationError
from ....core.laws import BaseLaw, BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.laws.utils import Call, NeighborReduction, Reduction, both, neighborhoods_in, reductions_in
from ....core.spaces import AdjacencyNeighborhood, BaseNeighborhood
from ...problems import (
    Entry,
    ReductionSchedule,
    Reference,
    Source,
    VariableKind,
    declarations_of,
    symbol,
    symbols_of,
    vector,
)
from ...problems._compiled_neighborhood import CompiledNeighborhood
from ...problems._compiled_reduction import CompiledReduction
from ...problems._edge_rows import ACTIVE_THRESHOLD
from ...problems._pair_read import PairRead
from ...problems._reads import Read
from .._base import BasePass
from .._compilation import Laws
from .._complex import _modulus_form
from .._declarations import Declarations
from .._draw_lowering import DrawLowering
from .._kernel_spec import KernelSpec
from .._observables import _HeldElement, _HeldReduction
from ._mask_requests import MaskRequestsPass
from ._specs import SpecsPass


@dataclass
class _Reductions:
    """Reductions collected from the laws of a system, deduplicated, with their placeholders and hidden entries."""

    declarations: Declarations
    drawn: int = 0
    found: dict[tuple[Any, ...], CompiledReduction] = field(default_factory=dict)
    by_symbol: dict[Expression, CompiledReduction] = field(default_factory=dict)
    readers: dict[Expression, str | None] = field(default_factory=dict)
    neighbourhoods: dict[tuple[Any, ...], CompiledNeighborhood] = field(default_factory=dict)
    local: dict[Expression, CompiledNeighborhood] = field(default_factory=dict)
    sequence: list[CompiledReduction | CompiledNeighborhood] = field(default_factory=list)

    @property
    def parameters(self) -> int:
        """Size of the parameter vector and the drawn values, where the hidden entries of the reductions start."""
        return self.declarations.ends[False] + self.drawn

    @property
    def extent(self) -> int:
        """Size of the extended parameter vector."""
        return max((item.entry.start + item.entry.size for item in self.sequence), default=self.parameters)

    def lower(self, value: object, spec: KernelSpec) -> object:
        """Return a law value with every reduction replaced by the placeholder that kernels read."""
        if isinstance(value, FreeModuleElement):
            return vector(self.lower(item, spec) for item in value)
        if not isinstance(value, Expression):
            return value
        found = reductions_in(value)
        local = neighborhoods_in(value)
        if not found and not local:
            return value
        placeholders = {item.call: self._placeholder(item, spec) for item in found}
        placeholders.update({item.call: self._neighborhood(item, spec) for item in local})
        lowered: Expression = value
        for call in sorted(placeholders, key=lambda item: -len(str(item))):
            lowered = lowered.subs({call: placeholders[call]})
        return _modulus_form(lowered)

    def _group_of(self, item: Expression, spec: KernelSpec) -> str | None:
        """Return the group, element set, or element whose members a symbol of a reduction stands for."""
        if item in spec.sets:
            return spec.sets[item].label
        if item in self.by_symbol or item in self.local:
            return self.readers[item]
        reference = spec.references.get(item)
        if reference is None or reference.source not in {Source.MEMBER, Source.SINGLE}:
            return None
        owner = reference.entry.name.rpartition('.')[0]
        return owner if reference.source is Source.MEMBER and owner in self.declarations.blocks else None

    def _group(self, reduction: Reduction, spec: KernelSpec, symbols: set[Expression]) -> str | None:
        """Return the one group a reduction reduces over, after checking that it reads no second group."""
        groups = {group for item in symbols if (group := self._group_of(item, spec)) is not None}
        over = None
        if reduction.over is not None:
            over = self._group_of(reduction.over, spec)
            reference = spec.references.get(reduction.over)
            if over is None and reference is not None:
                over = reference.entry.name.rpartition('.')[0]
            if over is not None:
                groups.add(over)
        if len(groups) > 1:
            names = ' and '.join(sorted(groups))
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reads the groups {names}. A reduction reduces over '
                'one group, so write one reduction per group.'
            )
            raise RoleError(error_msg)
        if groups:
            return next(iter(groups))
        if spec.group is not None:
            return spec.group
        return next((item.label for item in spec.sets.values()), None)

    def _read(self, item: Expression, spec: KernelSpec, group: str | None, size: int) -> Read:
        """Return how the reduction pass reads one symbol of a reduction."""
        if item in spec.sets:
            references = spec.sets[item].references
            indices = np.asarray([reference.entry.start + reference.column for reference in references], dtype=np.intp)
            return Read(references[0].is_state, indices)
        if item in self.by_symbol:
            nested = self.by_symbol[item]
            if nested.weights is None:
                return Read(False, nested.entry.start, nested.entry)
            if self.readers[item] != group:
                error_msg = (
                    f'A reduction of {spec.owner} over {group} reads the weighted reduction {nested.symbol} of the '
                    f'members of {self.readers[item]}. Read a weighted reduction inside reductions over its readers.'
                )
                raise ValidationError(error_msg)
            return Read(False, (nested.entry.start + np.arange(nested.entry.size)).astype(np.intp), nested.entry)
        if item in self.local:
            local = self.local[item]
            if local.group != group:
                error_msg = (
                    f'A reduction of {spec.owner} over {group} reads the neighbourhood reduction {local.symbol} of the '
                    f'members of {local.group}. Read a neighbourhood reduction inside reductions over its members.'
                )
                raise ValidationError(error_msg)
            return Read(False, (local.entry.start + np.arange(local.entry.size)).astype(np.intp), local.entry)
        reference = spec.references.get(item)
        if reference is None or reference.source not in {Source.MEMBER, Source.SINGLE}:
            error_msg = (
                f'A reduction of {spec.owner} reads {item}, which is not a state or parameter of an element, a group '
                'member, or another reduction. Reductions read the namespace `s` of the law.'
            )
            raise ValidationError(error_msg)
        entry = reference.entry
        if reference.source is Source.MEMBER and entry.members is not None:
            indices = entry.start + np.arange(entry.rows) * entry.columns + reference.column
            return Read(reference.is_state, indices.astype(np.intp), entry, reference.column, entry.columns)
        return Read(reference.is_state, entry.start + reference.column, entry, reference.column)

    def _members(self, group: str | None, spec: KernelSpec) -> tuple[int, Entry | None, npt.NDArray[np.intp] | None]:
        """Return the size of a group, an entry that gives its size, and the state indices of an element set."""
        if group is None:
            return 1, None, None
        block = self.declarations.blocks.get(group)
        if block is not None:
            entry = next(iter(block.entries.values()), None)
            return block.rows, entry, None
        references = next(item.references for item in spec.sets.values() if item.label == group)
        indices = np.asarray([reference.entry.start + reference.column for reference in references], dtype=np.intp)
        return len(references), None, indices

    def _weights(self, reduction: Reduction, spec: KernelSpec, group: str | None, size: int) -> Any:  # noqa: ANN401
        """Return the weights of a reduction as a CSR array after checking their shape."""
        if reduction.weights is None:
            return None
        readers = len(next(iter(spec.sets.values())).references) if spec.sets else spec.rows
        weights = reduction.weights
        if np.ndim(weights) == 1 and len(weights) == size:
            weights = np.broadcast_to(weights, (readers, size))
        shape = tuple(weights.shape)
        if shape != (readers, size):
            error_msg = (
                f'The weights of the reduction {reduction.call} of {spec.owner} over {group or "no group"} have the '
                f'shape {shape}, but the reduction needs the shape {(readers, size)}, one row per reading member and '
                'one column per reduced member.'
            )
            raise ValidationError(error_msg)
        return scipy.sparse.csr_array(weights, dtype=float)

    def _held(self, reduction: Reduction, spec: KernelSpec) -> Expression | None:
        """Return the state of the element that holds a reduction of agents, or `None` for a reduction of elements."""
        held = _HeldReduction._held_reduction(reduction, self.declarations.agents, self.declarations.time, spec.owner)
        if held is None:
            return None
        block = next(
            (
                block
                for block in self.declarations.blocks.values()
                if isinstance(block.node.component, _HeldElement) and block.node.component.reduction.key == held.key
            ),
            None,
        )
        if block is None:
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reduces the agents of {held.group}, which only a '
                'co-simulation computes, at its synchronisation points. Compile the system with `compile_system`.'
            )
            raise ValidationError(error_msg)
        value = cast('Expression', block.symbols['value'])
        spec.references[value] = Reference(block.entries['value'], True, 0, Source.SINGLE)
        return value

    def _placeholder(self, reduction: Reduction, spec: KernelSpec) -> Expression:
        """Return the placeholder of a reduction, the state that holds a reduction of agents, or its complex parts."""
        held = self._held(reduction, spec)
        if held is not None:
            return held
        if not SR(reduction.argument).has(I):
            return self._compiled(reduction, spec)
        if reduction.kind not in {'sum', 'mean'}:
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reduces a complex expression, which only total and '
                'mean reduce. Reduce its real or imaginary part.'
            )
            raise ValidationError(error_msg)
        argument = SR(reduction.argument)
        real = self._compiled(replace(reduction, argument=argument.real_part()), spec)
        imaginary = self._compiled(replace(reduction, argument=argument.imag_part()), spec)
        return real + I * imaginary

    def _compiled(self, reduction: Reduction, spec: KernelSpec) -> Expression:
        """Return the placeholder of a reduction of a real expression, compiling it the first time it is found."""
        argument = SR(self.lower(reduction.argument, spec))
        condition = None if reduction.condition is None else SR(self.lower(reduction.condition, spec))
        symbols = symbols_of([argument, *([condition] if condition is not None else [])]) - {self.declarations.time}
        group = self._group(reduction, spec, symbols)
        activity = MaskRequestsPass._activity_symbol(spec, group) if group in self.declarations.capacity else None
        if activity is not None:
            present = SR(activity) > ACTIVE_THRESHOLD
            condition = present if condition is None else both(condition, present)
            symbols.add(activity)
        size, members, set_indices = self._members(group, spec)
        weights = self._weights(reduction, spec, group, size)
        canonical = {item: spec.sets[item].canonical for item in symbols if item in spec.sets}
        argument_key = argument.subs(canonical) if canonical else argument
        condition_key = condition.subs(canonical) if canonical and condition is not None else condition
        key = (reduction.kind, str(argument_key), str(condition_key), str(reduction.call.operands()[2]), group)
        if key not in self.found:
            index = len(self.found)
            readers = weights.shape[0] if weights is not None else None
            shape = (readers,) if readers is not None else ()
            members_of = readers if readers is not None and not spec.sets else None
            entry = Entry(f'__reduced_{index}', self.extent, shape, RR, members_of)
            placeholder = symbol(f'_reduced_{index}')
            reads = {
                canonical.get(item, item): self._read(item, spec, group, size) for item in sorted(symbols, key=str)
            }
            compiled = CompiledReduction(
                placeholder,
                reduction.kind,
                argument_key,
                condition_key,
                weights,
                entry,
                size,
                group,
                reads,
                self.declarations.time,
                members,
                set_indices,
            )
            self.found[key] = compiled
            self.sequence.append(compiled)
            self.by_symbol[placeholder] = compiled
            reading = spec.group or next((item.label for item in spec.sets.values()), None)
            self.readers[placeholder] = reading if weights is not None else None
        compiled = self.found[key]
        column = spec.set_index if compiled.weights is not None and spec.sets else 0
        spec.references[compiled.symbol] = Reference(compiled.entry, False, column, Source.REDUCED)
        return compiled.symbol

    def _neighbor_group(self, reduction: NeighborReduction, spec: KernelSpec, symbols: set[Expression]) -> str:
        """Return the group whose members read a neighbourhood reduction and are each other's neighbours."""
        groups = set()
        for item in symbols:
            reference = spec.references.get(item)
            if item in self.local:
                groups.add(self.local[item].group)
            elif item in self.by_symbol and self.readers[item] is not None:
                groups.add(str(self.readers[item]))
            elif reference is not None and reference.source in {Source.MEMBER, Source.OTHER}:
                groups.add(reference.entry.name.rpartition('.')[0])
        groups = {group for group in groups if group in self.declarations.groups and not self._on_edges(group)} or (
            {spec.group} if spec.group is not None else set()
        )
        if len(groups) != 1:
            names = ' and '.join(sorted(groups)) or 'no group'
            error_msg = (
                f'The neighbourhood reduction {reduction.call} of {spec.owner} reads {names}. A neighbourhood '
                "reduction runs over the members of one group, which are each other's neighbours."
            )
            raise RoleError(error_msg)
        return next(iter(groups))

    def _on_edges(self, group: str) -> bool:
        """Check whether a group lives on the edges of a network."""
        return getattr(self.declarations.blocks[group].node.component, 'on', None) == 'edges'

    def _edge_read(self, reference: Reference, spec: KernelSpec, group: str, over: object) -> PairRead:
        """Return how a neighbourhood reduction reads a state of the edge group of its layer, per pair of neighbours."""
        entry = reference.entry
        edges = self.declarations.blocks[entry.name.rpartition('.')[0]]
        nodes = self.declarations.blocks[group]
        layer = getattr(over, 'layer', None)
        same = isinstance(over, AdjacencyNeighborhood) and SpecsPass._group_space(edges) is SpecsPass._group_space(
            nodes,
        )
        if not same or getattr(edges.node.component, 'layer', None) != layer:
            error_msg = (
                f'A neighbourhood reduction of {spec.owner} over {over!r} reads {entry.name} of the edge group '
                f'{edges.node.path}, which lives on other edges. Reduce over the layer of the edge group.'
            )
            raise RoleError(error_msg)
        indices = (entry.start + np.arange(entry.rows) * entry.columns + reference.column).astype(np.intp)
        return PairRead(Read(reference.is_state, indices, entry, reference.column, entry.columns), 'edge')

    def _pair_read(
        self,
        item: Expression,
        spec: KernelSpec,
        group: str,
        search: Any,  # noqa: ANN401
        over: object = None,
    ) -> PairRead:
        """Return how a neighbourhood reduction reads one symbol: shared, of the reading member, or of the neighbour."""
        reference = spec.references.get(item)
        if item in self.local or (item in self.by_symbol and self.readers[item] is not None):
            return PairRead(self._read(item, spec, group, 0), 'reader')
        if item in self.by_symbol:
            return PairRead(self._read(item, spec, None, 1), 'shared')
        if reference is None or reference.source not in {Source.MEMBER, Source.SINGLE, Source.OTHER}:
            error_msg = (
                f'A neighbourhood reduction of {spec.owner} reads {item}, which is not a state or parameter of a '
                'member, of its neighbour, or of an element. Neighbourhood reductions read the namespaces `s` and '
                '`s.other` of the law.'
            )
            raise ValidationError(error_msg)
        if reference.source is Source.SINGLE:
            read = Read(reference.is_state, reference.index, reference.entry, reference.column)
            return PairRead(read, 'shared')
        entry = reference.entry
        owner = entry.name.rpartition('.')[0]
        if owner != group and owner in self.declarations.blocks and self._on_edges(owner):
            return self._edge_read(reference, spec, group, over)
        if owner != group:
            error_msg = (
                f'A neighbourhood reduction of {spec.owner} over the members of {group} reads {item} of another group. '
                'Read the other group through a reduction over its members.'
            )
            raise RoleError(error_msg)
        indices = (entry.start + np.arange(entry.rows) * entry.columns + reference.column).astype(np.intp)
        read = Read(reference.is_state, indices, entry, reference.column, entry.columns)
        if reference.source is Source.MEMBER:
            return PairRead(read, 'reader')
        image = getattr(search, 'periods', None) is not None and entry == getattr(search, 'positions', None)
        return PairRead(read, 'other', image)

    def _neighborhood(self, reduction: NeighborReduction, spec: KernelSpec) -> Expression:
        """Return the placeholder of a neighbourhood reduction, compiling it the first time it is found."""
        argument = SR(self.lower(reduction.argument, spec))
        weight = SR(1) if reduction.weight is None else SR(self.lower(reduction.weight, spec))
        if argument.has(I) or weight.has(I):
            error_msg = (
                f'The neighbourhood reduction {reduction.call} of {spec.owner} reduces a complex expression. Reduce '
                'its real and imaginary parts.'
            )
            raise ValidationError(error_msg)
        symbols = symbols_of([argument, weight]) - {self.declarations.time}
        group = self._neighbor_group(reduction, spec, symbols)
        block = self.declarations.blocks[group]
        over = (
            reduction.over
            if reduction.over is not None
            else CompiledNeighborhood._default_neighborhood(block.node.space)
        )
        if over is None:
            error_msg = (
                f'The neighbourhood reduction {reduction.call} of {group} names no neighbourhood, but the group lives '
                f'on {block.node.space!r}, which has none of its own. Reduce over Radius, Nearest, or Cone.'
            )
            raise ValidationError(error_msg)
        key = ('neighborhood', reduction.kind, str(argument), str(weight), repr(over), group)
        if key not in self.neighbourhoods:
            states = declarations_of(block.node.component_type, VariableKind.STATE)
            changing = (
                self.declarations.changing_entries(SpecsPass._group_space(block))
                if block.node.active is not None
                else None
            )
            over = over.check() if isinstance(over, BaseNeighborhood) else over
            search = CompiledNeighborhood._neighbor_search(
                over, block.node.space, block.rows, block.entries, states, changing=changing,
            )
            reads = {item: self._pair_read(item, spec, group, search, over) for item in sorted(symbols, key=str)}
            index = len(self.neighbourhoods)
            entry = Entry(f'__neighbours_{index}', self.extent, (block.rows,), RR, block.rows)
            compiled = CompiledNeighborhood(
                symbol(f'_neighbours_{index}'),
                reduction.kind,
                argument,
                weight,
                entry,
                group,
                search,
                reads,
                self.declarations.time,
            )
            self.neighbourhoods[key] = compiled
            self.sequence.append(compiled)
            self.local[compiled.symbol] = compiled
            self.readers[compiled.symbol] = group
        compiled = self.neighbourhoods[key]
        spec.references[compiled.symbol] = Reference(compiled.entry, False, 0, Source.REDUCED)
        return compiled.symbol


class ReductionsPass(BasePass):
    """Pass that lowers the reductions of the laws, of the laws set apart, and of the events, in that order."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn', 'drawn_apart', 'layout', 'drawing', 'events')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('reduced', 'reduced_apart', 'reductions', 'events')

    def _run(self, inputs: object = None) -> None:
        """Lower the reductions of the drawn laws, the laws set apart, and the events."""
        compilation = self.compilation_
        declarations = cast('Declarations', compilation.layout)
        drawing = cast('DrawLowering', compilation.drawing)
        reductions = _Reductions(declarations, drawing.draws.size if drawing.draws else 0)
        specs, apart = cast('Laws', compilation.drawn), cast('Laws', compilation.drawn_apart)
        lowered = [(spec, [self._lower_law(law, spec, reductions) for law in laws]) for spec, laws in [*specs, *apart]]
        events = []
        for spec, event in compilation.events:
            reset = {target: reductions.lower(SR(value), spec) for target, value in (event.reset or {}).items()}
            guard = reductions.lower(SR(event.guard), spec)
            events.append(
                (spec, BasePass._replaced(event, guard=guard, reset=reset if event.reset is not None else None)),
            )
        compilation.events = events
        for spec, _ in lowered:
            self._prune_foreign(spec, declarations)
        compilation.reduced = lowered[: len(specs)]
        compilation.reduced_apart = lowered[len(specs) :]
        compilation.reductions = ReductionSchedule(
            tuple(reductions.found.values()),
            reductions.parameters,
            tuple(reductions.neighbourhoods.values()),
            tuple(reductions.sequence),
        )

    @staticmethod
    def _lower_law(law: BaseLaw, spec: KernelSpec, reductions: _Reductions) -> BaseLaw:
        """Return a law with every reduction in its values replaced by its placeholder."""
        if isinstance(law, JumpLaw):
            return JumpLaw(reductions.lower(law.rate, spec), law.changes)
        if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
            return law
        value = law.value
        if isinstance(value, Call):
            value = Call(value.function, tuple(reductions.lower(argument, spec) for argument in value.arguments))
        else:
            value = reductions.lower(value, spec)
        if isinstance(law, RateLaw | NoiseLaw):
            return BasePass._replaced(law, value=value)
        return BasePass._replaced(law, value=value)

    @staticmethod
    def _prune_foreign(spec: KernelSpec, declarations: Declarations) -> None:
        """Remove the references of groups that a spec reads only through reductions, so kernels loop over its group."""
        if spec.pairs is not None:
            return
        owners = {
            reference.entry.name.rpartition('.')[0]
            for reference in spec.references.values()
            if reference.source is Source.MEMBER
        }
        foreign = (owners & declarations.groups) - {spec.group, *spec.aligned}
        if not foreign:
            return
        spec.references = {
            item: reference
            for item, reference in spec.references.items()
            if reference.source is not Source.MEMBER or reference.entry.name.rpartition('.')[0] not in foreign
        }
