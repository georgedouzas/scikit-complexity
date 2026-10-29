"""Lower the reductions and neighbourhood reductions of laws to placeholders, compiled once each."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Self

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import RR, SR, Expression, I
from sage.modules.free_module_element import FreeModuleElement

from ....core.laws import BaseLaw, BoundaryLaw, CrossingLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.laws.utils import both, neighborhoods_in, reductions_in
from ....core.spaces import (
    AdjacencyNeighborhood,
    ConeNeighborhood,
    LatticeNeighborhood,
    MemoryNeighborhood,
    NearestNeighborhood,
    RadiusNeighborhood,
)
from ....errors import PipelineError, RoleError, ValidationError
from ....specs import CallSpec, EntrySpec, NeighborReductionSpec, PairReadSpec, ReadSpec, ReductionSpec
from ....specs._capacity_rows_spec import ACTIVE_THRESHOLD
from ....specs._held_reduction_spec import _HeldReductionSpec
from ....types import VariableKind
from ..._utils import declarations_of, symbol, symbols_of, vector
from ...problems import DeclarationsSpec, KernelSpec, ReferenceSpec
from ...problems._neighborhood_kernel import NeighborhoodKernel
from ...problems._reduction_kernel import ReductionKernel
from ...types import ReadKind
from .._base import BaseLowering, BasePass
from .._compilation import Compilation
from .._complex import _modulus_form
from .._observables import _HeldElement
from ._mask_requests import MaskRequestsPass
from ._specs import SpecsPass


class ReductionLowering(BaseLowering[list[tuple[KernelSpec, list[BaseLaw]]]]):
    """Lowering of the reductions and neighbourhood reductions of laws to placeholders, compiled once each.

    `prepare(compilation)` reads the layout and the drawn values of the compilation, and `run(laws)` takes the laws of
    every kernel spec and returns them with every reduction replaced by the placeholder that kernels read, collecting
    the reductions, deduplicated, with their hidden entries. `_apply_references()` then gives every spec the references
    of its placeholders.

    Attributes:
        declarations_:
            The layout of the compilation, which the hidden entries extend.

        drawn_:
            The number of drawn values after the parameters.

        found_:
            The compiled reductions by key.

        by_symbol_:
            The compiled reductions by placeholder.

        readers_:
            The group or element that reads every placeholder, or `None`.

        neighbourhoods_:
            The compiled neighbourhood reductions by key.

        local_:
            The compiled neighbourhood reductions by placeholder.

        sequence_:
            The compiled reductions and neighbourhood reductions in the order they are evaluated.
    """

    declarations_: DeclarationsSpec
    drawn_: int
    found_: dict[tuple[object, ...], ReductionKernel]
    by_symbol_: dict[Expression, ReductionKernel]
    readers_: dict[Expression, str | None]
    neighbourhoods_: dict[tuple[object, ...], NeighborhoodKernel]
    local_: dict[Expression, NeighborhoodKernel]
    sequence_: list[ReductionKernel | NeighborhoodKernel]

    def _prepare(self, context: Compilation | None) -> Self:
        """Read the layout and the drawn values of the compilation, with no reduction collected yet.

        Args:
            context:
                The compilation.

        Returns:
            self:
                The prepared lowering.
        """
        super()._prepare(context)
        draws = self.compilation_.draws
        self.declarations_ = Compilation._required(self.compilation_.layout, 'layout')
        self.drawn_ = draws.size if draws else 0
        self.found_, self.by_symbol_, self.readers_ = {}, {}, {}
        self.neighbourhoods_, self.local_, self.sequence_ = {}, {}, []
        return self

    def _run(self, inputs: list[tuple[KernelSpec, list[BaseLaw]]] | None) -> list[tuple[KernelSpec, list[BaseLaw]]]:
        """Return the laws of every kernel spec with every reduction replaced by its placeholder.

        Args:
            inputs:
                The laws of every kernel spec, crossing laws included.

        Returns:
            laws:
                The lowered laws of every kernel spec.

        Raises:
            PipelineError:
                No laws are passed.
        """
        if inputs is None:
            error_msg = 'The reduction lowering runs on the laws of every kernel spec. Pass them to `run`.'
            raise PipelineError(error_msg)
        return [(spec, [self._lower_law(law, spec) for law in laws]) for spec, laws in inputs]

    @property
    def parameters_(self) -> int:
        """Size of the parameter vector and the drawn values, where the hidden entries of the reductions start."""
        return self.declarations_.ends[False] + self.drawn_

    @property
    def extent_(self) -> int:
        """Size of the extended parameter vector."""
        return max((item.entry.start + item.entry.size for item in self.sequence_), default=self.parameters_)

    def _lower_law(self, law: BaseLaw, spec: KernelSpec) -> BaseLaw:
        """Return a law with every reduction in its values, its guard, and its reset replaced by its placeholder."""
        if isinstance(law, CrossingLaw):
            reset = {target: self._lower(SR(value), spec) for target, value in (law.reset or {}).items()}
            guard = self._lower(SR(law.guard), spec)
            return BasePass._replaced(law, guard=guard, reset=reset if law.reset is not None else None)
        if isinstance(law, JumpLaw):
            return JumpLaw(self._lower(law.rate, spec), law.changes)
        if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
            return law
        value = law.value
        if isinstance(value, CallSpec):
            value = CallSpec(value.function, tuple(self._lower(argument, spec) for argument in value.arguments))
        else:
            value = self._lower(value, spec)
        return BasePass._replaced(law, value=value)

    def _lower(self, value: object, spec: KernelSpec) -> object:
        """Return a law value with every reduction replaced by the placeholder that kernels read."""
        if isinstance(value, FreeModuleElement):
            return vector(self._lower(item, spec) for item in value)
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
        if item in self.by_symbol_ or item in self.local_:
            return self.readers_[item]
        reference = self._reference_of(item, spec)
        if reference is None or reference.source not in {ReadKind.MEMBER, ReadKind.SINGLE}:
            return None
        owner = reference.entry.name.rpartition('.')[0]
        return owner if reference.source is ReadKind.MEMBER and owner in self.declarations_.blocks else None

    def _group(self, reduction: ReductionSpec, spec: KernelSpec, symbols: set[Expression]) -> str | None:
        """Return the one group a reduction reduces over, after checking that it reads no second group."""
        groups = {group for item in symbols if (group := self._group_of(item, spec)) is not None}
        over = None
        if reduction.over is not None:
            over = self._group_of(reduction.over, spec)
            reference = self._reference_of(reduction.over, spec)
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

    def _read(self, item: Expression, spec: KernelSpec, group: str | None, size: int) -> ReadSpec:
        """Return how the reduction pass reads one symbol of a reduction."""
        if item in spec.sets:
            references = spec.sets[item].references
            indices = np.asarray([reference.entry.start + reference.column for reference in references], dtype=np.intp)
            return ReadSpec(references[0].is_state, indices)
        if item in self.by_symbol_:
            nested = self.by_symbol_[item]
            if nested.weights is None:
                return ReadSpec(False, nested.entry.start, nested.entry)
            if self.readers_[item] != group:
                error_msg = (
                    f'A reduction of {spec.owner} over {group} reads the weighted reduction {nested.symbol} of the '
                    f'members of {self.readers_[item]}. Read a weighted reduction inside reductions over its readers.'
                )
                raise ValidationError(error_msg)
            return ReadSpec(False, (nested.entry.start + np.arange(nested.entry.size)).astype(np.intp), nested.entry)
        if item in self.local_:
            local = self.local_[item]
            if local.group != group:
                error_msg = (
                    f'A reduction of {spec.owner} over {group} reads the neighbourhood reduction {local.symbol} of the '
                    f'members of {local.group}. Read a neighbourhood reduction inside reductions over its members.'
                )
                raise ValidationError(error_msg)
            return ReadSpec(False, (local.entry.start + np.arange(local.entry.size)).astype(np.intp), local.entry)
        reference = self._reference_of(item, spec)
        if reference is None or reference.source not in {ReadKind.MEMBER, ReadKind.SINGLE}:
            error_msg = (
                f'A reduction of {spec.owner} reads {item}, which is not a state or parameter of an element, a group '
                'member, or another reduction. Reductions read the namespace `s` of the law.'
            )
            raise ValidationError(error_msg)
        entry = reference.entry
        if reference.source is ReadKind.MEMBER and entry.members is not None:
            indices = entry.start + np.arange(entry.rows) * entry.columns + reference.column
            return ReadSpec(reference.is_state, indices.astype(np.intp), entry, reference.column, entry.columns)
        return ReadSpec(reference.is_state, entry.start + reference.column, entry, reference.column)

    def _members(self, group: str | None, spec: KernelSpec) -> tuple[int, EntrySpec | None]:
        """Return the size of a group and an entry that gives its size, or `None` for an element set."""
        if group is None:
            return 1, None
        block = self.declarations_.blocks.get(group)
        if block is not None:
            entry = next(iter(block.entries.values()), None)
            return block.rows, entry
        references = next(item.references for item in spec.sets.values() if item.label == group)
        return len(references), None

    def _weights(
        self,
        reduction: ReductionSpec,
        spec: KernelSpec,
        group: str | None,
        size: int,
    ) -> scipy.sparse.csr_array | None:
        """Return the weights of a reduction as a CSR array after checking their shape."""
        if reduction.weights is None:
            return None
        readers = len(next(iter(spec.sets.values())).references) if spec.sets else spec.rows
        weights = reduction.weights
        if not isinstance(weights, scipy.sparse.sparray | scipy.sparse.spmatrix):
            weights = np.asarray(weights, dtype=float)
            if weights.ndim == 1 and len(weights) == size:
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

    @staticmethod
    def _held_reduction(
        reduction: ReductionSpec,
        agents: Mapping[Expression, tuple[str, str]],
        time: Expression,
        owner: str,
    ) -> _HeldReductionSpec | None:
        """Return the held reduction of a reduction of agents, or `None` for a reduction of elements.

        Args:
            reduction:
                The reduction a law reads.

            agents:
                The path of the agent group and the variable of every symbol of agents.

            time:
                The time symbol.

            owner:
                The path of the component whose law reads the reduction.

        Returns:
            held:
                The held reduction, or `None`.

        Raises:
            RoleError:
                The reduction reads the agents of more than one group.

            ValidationError:
                The reduction has weights, or reads a symbol that is not a variable of the agents.
        """
        expressions = [SR(reduction.argument), *([SR(reduction.condition)] if reduction.condition is not None else [])]
        symbols = symbols_of(expressions) - {time}
        read = symbols | ({reduction.over} if reduction.over is not None else set())
        groups = {agents[item][0] for item in read if item in agents}
        if not groups:
            return None
        if len(groups) > 1:
            error_msg = (
                f'The reduction {reduction.call} of {owner} reads the agents of {" and ".join(sorted(groups))}. A '
                'reduction reduces over one group, so write one reduction per group.'
            )
            raise RoleError(error_msg)
        group = next(iter(groups))
        others = sorted(str(item) for item in symbols if item not in agents or agents[item][0] != group)
        if reduction.weights is not None or others:
            error_msg = (
                f'The reduction {reduction.call} of {owner} reduces the agents of {group}, which a co-simulation '
                'computes at its synchronisation points from the scalar states and parameters of the agents alone, '
                f'without weights{", but it reads " + ", ".join(others) if others else ""}. Reduce an expression of '
                'the agents.'
            )
            raise ValidationError(error_msg)
        ordered = tuple(sorted((item for item in symbols if agents[item][1]), key=str))
        return _HeldReductionSpec(
            group,
            reduction.kind,
            tuple(agents[item][1] for item in ordered),
            ordered,
            SR(reduction.argument),
            None if reduction.condition is None else SR(reduction.condition),
        )

    def _held(self, reduction: ReductionSpec, spec: KernelSpec) -> Expression | None:
        """Return the state of the element that holds a reduction of agents, or `None` for a reduction of elements."""
        held = self._held_reduction(reduction, self.declarations_.agents, self.declarations_.time, spec.owner)
        if held is None:
            return None
        block = next(
            (
                block
                for block in self.declarations_.blocks.values()
                if isinstance(block.node.component, _HeldElement)
                and block.node.component.reduction is not None
                and block.node.component.reduction.key == held.key
            ),
            None,
        )
        if block is None:
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reduces the agents of {held.group}, which only a '
                'co-simulation computes, at its synchronisation points. Compile the system with `compile_system`.'
            )
            raise ValidationError(error_msg)
        value = block.symbols['value']
        if not isinstance(value, Expression):
            error_msg = f'The reduction {reduction.call} of {spec.owner} reads a vector of agents, not one value.'
            raise ValidationError(error_msg)
        self._add_reference(spec, value, ReferenceSpec(block.entries['value'], True, 0, ReadKind.SINGLE))
        return value

    def _placeholder(self, reduction: ReductionSpec, spec: KernelSpec) -> Expression:
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

    def _compiled(self, reduction: ReductionSpec, spec: KernelSpec) -> Expression:
        """Return the placeholder of a reduction of a real expression, compiling it the first time it is found."""
        argument = SR(self._lower(reduction.argument, spec))
        condition = None if reduction.condition is None else SR(self._lower(reduction.condition, spec))
        symbols = symbols_of([argument, *([condition] if condition is not None else [])]) - {self.declarations_.time}
        group = self._group(reduction, spec, symbols)
        activity = MaskRequestsPass._activity_symbol(spec, group) if group in self.declarations_.capacity else None
        if activity is not None:
            present = SR(activity) > ACTIVE_THRESHOLD
            condition = present if condition is None else both(condition, present)
            symbols.add(activity)
        size, members = self._members(group, spec)
        weights = self._weights(reduction, spec, group, size)
        canonical = {item: spec.sets[item].canonical for item in symbols if item in spec.sets}
        argument_key = argument.subs(canonical) if canonical else argument
        condition_key = condition.subs(canonical) if canonical and condition is not None else condition
        key = (reduction.kind, str(argument_key), str(condition_key), str(reduction.call.operands()[2]), group)
        if key not in self.found_:
            index = len(self.found_)
            readers = weights.shape[0] if weights is not None else None
            shape = (readers,) if readers is not None else ()
            members_of = readers if readers is not None and not spec.sets else None
            entry = EntrySpec(f'__reduced_{index}', self.extent_, shape, RR, members_of)
            placeholder = symbol(f'_reduced_{index}')
            reads = {
                canonical.get(item, item): self._read(item, spec, group, size) for item in sorted(symbols, key=str)
            }
            compiled = ReductionKernel(
                placeholder,
                reduction.kind,
                argument_key,
                condition_key,
                weights,
                entry=entry,
                size=size,
                group=group,
                reads=reads,
                time=self.declarations_.time,
                members=members,
            ).check()
            self.found_[key] = compiled
            self.sequence_.append(compiled)
            self.by_symbol_[placeholder] = compiled
            reading = spec.group or next((item.label for item in spec.sets.values()), None)
            self.readers_[placeholder] = reading if weights is not None else None
        compiled = self.found_[key]
        column = spec.set_index if compiled.weights is not None and spec.sets else 0
        self._add_reference(spec, compiled.symbol, ReferenceSpec(compiled.entry, False, column, ReadKind.REDUCED))
        return compiled.symbol

    def _neighbor_group(self, reduction: NeighborReductionSpec, spec: KernelSpec, symbols: set[Expression]) -> str:
        """Return the group whose members read a neighbourhood reduction and are each other's neighbours."""
        groups = set()
        for item in symbols:
            reference = self._reference_of(item, spec)
            if item in self.local_:
                groups.add(self.local_[item].group)
            elif item in self.by_symbol_ and self.readers_[item] is not None:
                groups.add(str(self.readers_[item]))
            elif reference is not None and reference.source in {ReadKind.MEMBER, ReadKind.OTHER}:
                groups.add(reference.entry.name.rpartition('.')[0])
        groups = {group for group in groups if group in self.declarations_.groups and not self._on_edges(group)} or (
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
        return getattr(self.declarations_.blocks[group].node.component, 'on', None) == 'edges'

    def _edge_read(self, reference: ReferenceSpec, spec: KernelSpec, group: str, over: object) -> PairReadSpec:
        """Return how a neighbourhood reduction reads a state of the edge group of its layer, per pair of neighbours."""
        entry = reference.entry
        edges = self.declarations_.blocks[entry.name.rpartition('.')[0]]
        nodes = self.declarations_.blocks[group]
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
        return PairReadSpec(ReadSpec(reference.is_state, indices, entry, reference.column, entry.columns), 'edge')

    def _pair_read(
        self,
        item: Expression,
        spec: KernelSpec,
        group: str,
        search: object,
        over: object = None,
    ) -> PairReadSpec:
        """Return how a neighbourhood reduction reads one symbol: shared, of the reading member, or of the neighbour."""
        reference = self._reference_of(item, spec)
        if item in self.local_ or (item in self.by_symbol_ and self.readers_[item] is not None):
            return PairReadSpec(self._read(item, spec, group, 0), 'reader')
        if item in self.by_symbol_:
            return PairReadSpec(self._read(item, spec, None, 1), 'shared')
        if reference is None or reference.source not in {ReadKind.MEMBER, ReadKind.SINGLE, ReadKind.OTHER}:
            error_msg = (
                f'A neighbourhood reduction of {spec.owner} reads {item}, which is not a state or parameter of a '
                'member, of its neighbour, or of an element. Neighbourhood reductions read the namespaces `s` and '
                '`s.other` of the law.'
            )
            raise ValidationError(error_msg)
        if reference.source is ReadKind.SINGLE:
            read = ReadSpec(reference.is_state, reference.index, reference.entry, reference.column)
            return PairReadSpec(read, 'shared')
        entry = reference.entry
        owner = entry.name.rpartition('.')[0]
        if owner != group and owner in self.declarations_.blocks and self._on_edges(owner):
            return self._edge_read(reference, spec, group, over)
        if owner != group:
            error_msg = (
                f'A neighbourhood reduction of {spec.owner} over the members of {group} reads {item} of another group. '
                'ReadSpec the other group through a reduction over its members.'
            )
            raise RoleError(error_msg)
        indices = (entry.start + np.arange(entry.rows) * entry.columns + reference.column).astype(np.intp)
        read = ReadSpec(reference.is_state, indices, entry, reference.column, entry.columns)
        if reference.source is ReadKind.MEMBER:
            return PairReadSpec(read, 'reader')
        image = getattr(search, 'periods', None) is not None and entry == getattr(search, 'positions', None)
        return PairReadSpec(read, 'other', image)

    def _neighborhood(self, reduction: NeighborReductionSpec, spec: KernelSpec) -> Expression:
        """Return the placeholder of a neighbourhood reduction, compiling it the first time it is found."""
        argument = SR(self._lower(reduction.argument, spec))
        weight = SR(1) if reduction.weight is None else SR(self._lower(reduction.weight, spec))
        if argument.has(I) or weight.has(I):
            error_msg = (
                f'The neighbourhood reduction {reduction.call} of {spec.owner} reduces a complex expression. Reduce '
                'its real and imaginary parts.'
            )
            raise ValidationError(error_msg)
        symbols = symbols_of([argument, weight]) - {self.declarations_.time}
        group = self._neighbor_group(reduction, spec, symbols)
        block = self.declarations_.blocks[group]
        over = (
            reduction.over if reduction.over is not None else NeighborhoodKernel._default_neighborhood(block.node.space)
        )
        if over is None:
            error_msg = (
                f'The neighbourhood reduction {reduction.call} of {group} names no neighbourhood, but the group lives '
                f'on {block.node.space!r}, which has none of its own. Reduce over Radius, Nearest, or Cone.'
            )
            raise ValidationError(error_msg)
        key = ('neighborhood', reduction.kind, str(argument), str(weight), repr(over), group)
        if key not in self.neighbourhoods_:
            states = declarations_of(block.node.component_type, VariableKind.STATE)
            changing = (
                self.declarations_.changing_entries(SpecsPass._group_space(block))
                if block.node.active is not None
                else None
            )
            if not isinstance(
                over,
                AdjacencyNeighborhood
                | LatticeNeighborhood
                | MemoryNeighborhood
                | RadiusNeighborhood
                | NearestNeighborhood
                | ConeNeighborhood,
            ):
                error_msg = (
                    f'The neighbourhood reduction {reduction.call} of {group} reduces over {over!r}, which is not a '
                    'neighbourhood. Reduce over a lattice, network, memory, Radius, Nearest, or Cone neighbourhood.'
                )
                raise ValidationError(error_msg)
            over = over.check()
            search = NeighborhoodKernel._neighbor_search(
                over,
                block.node.space,
                block.rows,
                block.entries,
                states,
                changing=changing,
            )
            reads = {item: self._pair_read(item, spec, group, search, over) for item in sorted(symbols, key=str)}
            index = len(self.neighbourhoods_)
            entry = EntrySpec(f'__neighbours_{index}', self.extent_, (block.rows,), RR, block.rows)
            compiled = NeighborhoodKernel(
                symbol(f'_neighbours_{index}'),
                reduction.kind,
                argument,
                weight,
                entry,
                group=group,
                search=search,
                reads=reads,
                time=self.declarations_.time,
            ).check()
            self.neighbourhoods_[key] = compiled
            self.sequence_.append(compiled)
            self.local_[compiled.symbol] = compiled
            self.readers_[compiled.symbol] = group
        compiled = self.neighbourhoods_[key]
        self._add_reference(spec, compiled.symbol, ReferenceSpec(compiled.entry, False, 0, ReadKind.REDUCED))
        return compiled.symbol
