"""Lower the reductions of laws to placeholders compiled once each.

Neighbourhood reductions are lowered too.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import RR, SR, Expression, I
from sage.modules.free_module_element import FreeModuleElement
from sage.symbolic.operators import add_vararg, mul_vararg

from ....components.laws import BaseLaw, BoundaryLaw, CrossingLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....components.laws.utils import (
    both,
    cell_operations_in,
    contractions_in,
    hyperedges_in,
    maximum,
    neighborhoods_in,
    reductions_in,
)
from ....components.spaces import (
    AdjacencyNeighborhood,
    CellComplexSpace,
    CofaceNeighborhood,
    ConeNeighborhood,
    LatticeNeighborhood,
    MemoryNeighborhood,
    NearestNeighborhood,
    NetworkSpace,
    RadiusNeighborhood,
)
from ....errors import PipelineError, RoleError, ValidationError
from ....specs import (
    CallSpec,
    EntrySpec,
    HyperedgeReductionSpec,
    NeighborReductionSpec,
    PairReadSpec,
    ReadSpec,
    ReductionSpec,
)
from ....specs._capacity_rows_spec import ACTIVE_THRESHOLD
from ....specs._held_reduction_spec import _HeldReductionSpec
from ....types import Indices, VariableKind
from ..._utils import declarations_of, domain_of, symbol, symbols_of, vector
from ...problems._declarations_spec import DeclarationsSpec
from ...problems._kernel_spec import KernelSpec
from ...problems._neighborhood_kernel import NeighborhoodKernel
from ...problems._reduction_kernel import ReductionKernel
from ...problems._reference_spec import ReferenceSpec
from ...problems._static_search import StaticSearch
from ...types import ReadKind
from .._base import BaseLowering, BasePass
from .._compilation import Compilation
from .._complex import _modulus_form
from .._observables import _HeldElement
from ._cells import DEGREES, CellTables
from ._hyperedges import HyperedgeTables
from ._mask_requests import MaskRequestsPass
from ._specs import SpecsPass

PAIRWISE_KINDS = ('sum', 'mean', 'count', 'max', 'min')


class ReductionLowering(BaseLowering[list[tuple[KernelSpec, list[BaseLaw]]], list[tuple[KernelSpec, list[BaseLaw]]]]):
    """Lowering of the reductions of laws to placeholders compiled once each.

    Neighbourhood reductions are lowered too. `prepare(compilation)` reads the layout and the drawn values of the
    compilation, and `run(laws)` takes the laws of every kernel spec and sets them in `lowered_` with every reduction
    replaced by the placeholder that kernels read, collecting the reductions, deduplicated, with their hidden entries.
    `_apply_references()` then gives every spec the references of its placeholders.

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

        cells_:
            The degree and the cell complex of the value of every placeholder of the exterior calculus.
    """

    declarations_: DeclarationsSpec
    drawn_: int
    found_: dict[tuple[object, ...], ReductionKernel]
    by_symbol_: dict[Expression, ReductionKernel]
    readers_: dict[Expression, str | None]
    neighbourhoods_: dict[tuple[object, ...], NeighborhoodKernel]
    local_: dict[Expression, NeighborhoodKernel]
    sequence_: list[ReductionKernel | NeighborhoodKernel]
    cells_: dict[Expression, tuple[int, CellComplexSpace]]

    def _prepare(self, context: Compilation | None) -> None:
        """Read the layout and the drawn values of the compilation.

        No reduction is collected yet.

        Args:
            context:
                The compilation.
        """
        super()._prepare(context)
        draws = self.compilation_.draws
        self.declarations_ = Compilation._required(self.compilation_.layout, 'layout')
        self.drawn_ = draws.size if draws else 0
        self.found_, self.by_symbol_, self.readers_ = {}, {}, {}
        self.neighbourhoods_, self.local_, self.sequence_ = {}, {}, []
        self.cells_ = {}

    def _run(self, inputs: list[tuple[KernelSpec, list[BaseLaw]]] | None) -> None:
        """Set the laws of every kernel spec with reductions replaced by placeholders.

        Every reduction is replaced by its placeholder.

        Args:
            inputs:
                The laws of every kernel spec, crossing laws included.

        Raises:
            PipelineError:
                No laws are passed.
        """
        if inputs is None:
            error_msg = 'The reduction lowering runs on the laws of every kernel spec. Pass them to `run`.'
            raise PipelineError(error_msg)
        self.lowered_ = [(spec, [self._lower_law(law, spec) for law in laws]) for spec, laws in inputs]

    @property
    def parameters_(self) -> int:
        """Size of the parameter vector and the drawn values.

        It is where the hidden entries of the reductions start.
        """
        return self.declarations_.ends[False] + self.drawn_

    @property
    def extent_(self) -> int:
        """Size of the extended parameter vector."""
        return max((item.entry.start + item.entry.size for item in self.sequence_), default=self.parameters_)

    def _lower_law(self, law: BaseLaw, spec: KernelSpec) -> BaseLaw:
        """Return a law with every reduction replaced by its placeholder.

        The reductions are those in its values, its guard, and its reset.
        """
        if isinstance(law, CrossingLaw):
            reset = {
                target: value if BaseLaw._is_function(value) else self._lower(SR(value), spec)
                for target, value in (law.reset or {}).items()
            }
            guard = law.guard if BaseLaw._is_function(law.guard) else self._lower(SR(law.guard), spec)
            return BasePass._replaced(law, guard=guard, reset=reset if law.reset is not None else None)
        if isinstance(law, JumpLaw):
            return JumpLaw(self._lower(law.rate, spec), law.changes)
        if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
            return law
        value = law.value
        if isinstance(value, CallSpec):
            arguments = None if value.arguments is None else tuple(self._lower(item, spec) for item in value.arguments)
            value = replace(value, arguments=arguments)
        else:
            value = self._lower(value, spec)
        return BasePass._replaced(law, value=value)

    def _lower(self, value: object, spec: KernelSpec) -> object:
        """Return a law value with every reduction replaced by its placeholder.

        Kernels read the placeholder.
        """
        if isinstance(value, FreeModuleElement):
            return vector(self._lower(item, spec) for item in value)
        if not isinstance(value, Expression):
            return value
        found = reductions_in(value)
        local = neighborhoods_in(value)
        around = hyperedges_in(value)
        contracted = contractions_in(value)
        cells = cell_operations_in(value)
        if not found and not local and not around and not contracted and not cells:
            return value
        placeholders = {call: self._contraction(call, spec) for call in contracted}
        placeholders.update({call: self._cell_operation(call, spec) for call in cells})
        placeholders.update({item.call: self._placeholder(item, spec) for item in found})
        placeholders.update({item.call: self._neighborhood(item, spec) for item in local})
        placeholders.update({item.call: self._hyperedges(item, spec) for item in around})
        lowered: Expression = value
        for call in sorted(placeholders, key=lambda item: -len(str(item))):
            lowered = lowered.subs({call: placeholders[call]})
        return _modulus_form(lowered)

    def _group_of(self, item: Expression, spec: KernelSpec) -> str | None:
        """Return the group, element set, or element a reduction symbol stands for.

        The symbol stands for its members.
        """
        if item in spec.sets:
            return spec.sets[item].label
        if item in self.by_symbol_ or item in self.local_:
            return self.readers_[item]
        reference = self._reference_of(item, spec)
        if reference is None or reference.source not in {ReadKind.MEMBER, ReadKind.OTHER}:
            return None
        owner = reference.entry.name.rpartition('.')[0]
        return owner if owner in self.declarations_.blocks else None

    def _group(self, reduction: ReductionSpec, spec: KernelSpec, symbols: set[Expression]) -> str | None:
        """Return the one group a reduction reduces over.

        It first checks that it reads no second group.
        """
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
        if reference is None or reference.source not in {ReadKind.MEMBER, ReadKind.SINGLE, ReadKind.OTHER}:
            error_msg = (
                f'A reduction of {spec.owner} reads {item}, which is not a state or parameter of an element, a group '
                'member, or another reduction. Reductions read the namespace `s` of the law.'
            )
            raise ValidationError(error_msg)
        entry = reference.entry
        if reference.source in {ReadKind.MEMBER, ReadKind.OTHER} and entry.members is not None:
            indices = entry.start + np.arange(entry.rows) * entry.columns + reference.column
            return ReadSpec(reference.is_state, indices.astype(np.intp), entry, reference.column, entry.columns)
        return ReadSpec(reference.is_state, entry.start + reference.column, entry, reference.column)

    def _members(self, group: str | None, spec: KernelSpec) -> tuple[int, EntrySpec | None]:
        """Return the size of a group and an entry that gives its size.

        It is `None` for an element set.
        """
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
    ) -> scipy.sparse.csr_array | ReadSpec | None:
        """Return the weights of a reduction as a CSR array or a shaped parameter read.

        The shape is checked.
        """
        if reduction.weights is None:
            return None
        readers = len(next(iter(spec.sets.values())).references) if spec.sets else spec.rows
        weights = reduction.weights
        if isinstance(weights, Expression):
            return self._weights_read(reduction, spec, weights, (readers, size))
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

    def _weights_read(
        self,
        reduction: ReductionSpec,
        spec: KernelSpec,
        weights: Expression,
        shape: tuple[int, int],
    ) -> ReadSpec:
        """Return the read of the shaped parameter a reduction takes as weights.

        It first checks its shape.
        """
        reference = self._reference_of(weights, spec)
        entry = reference.entry if reference is not None else None
        if entry is None or entry.shape != shape:
            given = 'not a shaped parameter' if entry is None else f'of the shape {entry.shape}'
            error_msg = (
                f'The weights {weights} of the reduction {reduction.call} of {spec.owner} are {given}, but the '
                f'reduction needs a shaped parameter of the shape {shape}, one row per reading member and one column '
                'per reduced member.'
            )
            raise ValidationError(error_msg)
        indices = (entry.start + np.arange(entry.size)).astype(np.intp)
        return ReadSpec(reference.is_state if reference is not None else False, indices, entry)

    @staticmethod
    def _held_reduction(
        reduction: ReductionSpec,
        agents: Mapping[Expression, tuple[str, str]],
        time: Expression,
        owner: str,
    ) -> _HeldReductionSpec | None:
        """Return the held reduction of a reduction of agents.

        It is `None` for a reduction of elements.

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
        """Return the state of the element that holds a reduction of agents.

        It is `None` for a reduction of elements.
        """
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
                'co-simulation computes, at its synchronisation points. Compile the system with `PassCompiler`.'
            )
            raise ValidationError(error_msg)
        value = block.symbols['value']
        if not isinstance(value, Expression):
            error_msg = f'The reduction {reduction.call} of {spec.owner} reads a vector of agents, not one value.'
            raise ValidationError(error_msg)
        self._add_reference(spec, value, ReferenceSpec(block.entries['value'], True, 0, ReadKind.SINGLE))
        return value

    def _placeholder(self, reduction: ReductionSpec, spec: KernelSpec) -> Expression:
        """Return the placeholder of a reduction.

        It is the state that holds a reduction of agents, or its complex parts.
        """
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

    def _reduced_group(self, reduction: ReductionSpec, spec: KernelSpec, others: set[Expression]) -> str:
        """Return the one group whose members the `other` symbols of a reduction stand for."""
        groups = {self._owner_of(item, spec) for item in others}
        if reduction.over is not None:
            groups.add(self._group_of(reduction.over, spec) or self._owner_of(reduction.over, spec))
        if len(groups) != 1 or None in groups:
            names = ' and '.join(sorted(group or 'no group' for group in groups))
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reads the other members of {names}. A reduction reads '
                'the other members of the one group it reduces over.'
            )
            raise RoleError(error_msg)
        return str(next(iter(groups)))

    def _owner_of(self, item: Expression, spec: KernelSpec) -> str | None:
        """Return the path of the element or group a symbol of a spec reads.

        It is `None` for an unknown symbol.
        """
        reference = self._reference_of(item, spec)
        return None if reference is None else reference.entry.name.rpartition('.')[0]

    def _reads_reader(self, item: Expression, spec: KernelSpec) -> bool:
        """Check whether a symbol of a reduction holds one value per reading member.

        The reading member is a member of a group.
        """
        if item in self.local_ or item in self.by_symbol_:
            return self.readers_[item] is not None
        reference = self._reference_of(item, spec)
        return reference is not None and reference.source is ReadKind.MEMBER

    def _pairwise_read(self, item: Expression, spec: KernelSpec, readers: str | None) -> PairReadSpec:
        """Return how a reduction that reads `other` reads one symbol.

        It reads it shared, of the reader, or of the other member.
        """
        if item in self.local_ or (item in self.by_symbol_ and self.readers_[item] is not None):
            return PairReadSpec(self._read(item, spec, readers, 0), 'reader')
        if item in self.by_symbol_:
            return PairReadSpec(self._read(item, spec, None, 1), 'shared')
        reference = self._reference_of(item, spec)
        if reference is not None and reference.source is ReadKind.SINGLE:
            return PairReadSpec(
                ReadSpec(reference.is_state, reference.index, reference.entry, reference.column),
                'shared',
            )
        owner = self._owner_of(item, spec)
        sides = {ReadKind.OTHER: 'other', ReadKind.MEMBER: 'reader'}
        side = sides.get(reference.source) if reference is not None else None
        if reference is None or side is None or (side == 'reader' and owner != readers):
            error_msg = (
                f'A reduction of {spec.owner} reads {item}, which is not a state or parameter of an element, of the '
                f'reading member of {readers or "no group"}, or of the other member. Reductions read the namespaces '
                '`s` and `s.other` of the law.'
            )
            raise ValidationError(error_msg)
        entry = reference.entry
        indices = (entry.start + np.arange(entry.rows) * entry.columns + reference.column).astype(np.intp)
        return PairReadSpec(ReadSpec(reference.is_state, indices, entry, reference.column, entry.columns), side)

    def _weighted_pairwise_form(
        self,
        reduction: ReductionSpec,
        spec: KernelSpec,
        argument: Expression,
        selected: Expression,
        condition: Expression | None,
    ) -> tuple[str, Expression, Expression]:
        """Return the kind, argument, and weight of every pair of a weighted reduction.

        The reduction is weighted by a shaped parameter.
        """
        weights = reduction.weights
        if (
            not isinstance(weights, Expression)
            or reduction.kind not in {'sum', 'mean'}
            or (reduction.kind == 'mean' and condition is not None)
        ):
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reads `other` with weights, which only a sum, or a '
                'mean that selects no members, reads from a shaped parameter. Give the weights as a shaped parameter, '
                'or write the term so that it factors into a part of the member and a part of the other member.'
            )
            raise ValidationError(error_msg)
        return reduction.kind, argument, selected * weights

    def _pair_weights_read(
        self,
        reduction: ReductionSpec,
        spec: KernelSpec,
        group: str,
    ) -> dict[Expression, PairReadSpec]:
        """Return the read of the shaped parameter that weights the pairs of a reduction.

        It is no read without one.
        """
        weights = reduction.weights
        if not isinstance(weights, Expression):
            return {}
        readers = self.declarations_.blocks[spec.group].rows if spec.group is not None else 1
        read = self._weights_read(reduction, spec, weights, (readers, self.declarations_.blocks[group].rows))
        return {weights: PairReadSpec(read, 'pair')}

    def _pairwise_form(
        self,
        reduction: ReductionSpec,
        spec: KernelSpec,
        argument: Expression,
        condition: Expression | None,
    ) -> tuple[str, Expression, Expression]:
        """Return the kind, argument, and weight of every pair of a reduction.

        The reduction reads `other`.
        """
        if reduction.kind not in PAIRWISE_KINDS:
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reads `other`, which the reductions '
                f'{", ".join(PAIRWISE_KINDS)} read. Write it as one of them.'
            )
            raise ValidationError(error_msg)
        selected = SR(1) if condition is None else SR(condition)
        if reduction.weights is not None:
            return self._weighted_pairwise_form(reduction, spec, argument, selected, condition)
        if reduction.kind == 'count':
            return 'sum', selected, SR(1)
        if reduction.kind == 'mean' and condition is not None:
            return 'normalized-mean', argument, selected
        if reduction.kind in {'max', 'min'} and condition is not None:
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reads `other` and selects members, which the largest '
                'and smallest values of pairs do not. Select the members inside the argument instead.'
            )
            raise ValidationError(error_msg)
        return reduction.kind, argument, selected

    def _pairwise(
        self,
        reduction: ReductionSpec,
        spec: KernelSpec,
        argument: Expression,
        condition: Expression | None,
        others: set[Expression],
    ) -> Expression:
        """Return the placeholder of a reduction that reads `other`.

        It holds one value per reading member over every member. The symbols of `other` stand for the reduced member and
        the other symbols of the reading group for the reading member, so the reduction runs over every pair of a
        reading member and a member of the reduced group.
        """
        group = self._reduced_group(reduction, spec, others)
        if group in self.declarations_.capacity:
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} reads `other` over the variable group {group}, whose '
                'members come and go. Reduce over a group of fixed members.'
            )
            raise ValidationError(error_msg)
        kind, argument, weight = self._pairwise_form(reduction, spec, argument, condition)
        weighted = self._pair_weights_read(reduction, spec, group)
        symbols = symbols_of([argument, weight]) - {self.declarations_.time, *weighted}
        readers = spec.group
        rows = self.declarations_.blocks[readers].rows if readers is not None else 1
        size = self.declarations_.blocks[group].rows
        key = ('pairwise', kind, str(argument), str(weight), group, readers)
        if key not in self.neighbourhoods_:
            indptr = (np.arange(rows + 1) * size).astype(np.intp)
            search = StaticSearch(indptr, np.tile(np.arange(size, dtype=np.intp), rows)).prepare()
            reads = {item: self._pairwise_read(item, spec, readers) for item in sorted(symbols, key=str)} | weighted
            index = len(self.neighbourhoods_)
            shape, members = ((rows,), rows) if readers is not None else ((), None)
            entry = EntrySpec(f'__neighbours_{index}', self.extent_, shape, RR, members)
            compiled = NeighborhoodKernel(
                symbol(f'_neighbours_{index}'),
                kind,
                argument,
                weight,
                entry,
                group=readers or group,
                search=search,
                reads=reads,
                time=self.declarations_.time,
            ).prepare()
            self.neighbourhoods_[key] = compiled
            self.sequence_.append(compiled)
            self.local_[compiled.symbol] = compiled
            self.readers_[compiled.symbol] = readers
        compiled = self.neighbourhoods_[key]
        self._add_reference(spec, compiled.symbol, ReferenceSpec(compiled.entry, False, 0, ReadKind.REDUCED))
        return compiled.symbol

    def _separated(
        self,
        reduction: ReductionSpec,
        spec: KernelSpec,
        argument: Expression,
        others: set[Expression],
    ) -> Expression | None:
        """Return a sum or mean over pairs whose terms factor, else `None`.

        The terms factor into a reader part and an other part. A sum or mean over the other members `j` of `sum_k a_k(i)
        b_k(j)`, weighted by `W_ij` or not, is `sum_k a_k(i)` times the weighted sum or mean of `b_k(j)`, so it runs as
        one reduction per `b_k` over the members, at a cost linear in their number instead of one value per pair. With
        at least as many parts `b_k` as members, the pairs cost less, and it returns `None`.
        """
        if reduction.kind not in {'sum', 'mean'}:
            return None
        expanded = argument.expand()
        terms = expanded.operands() if expanded.operator() is add_vararg else [expanded]
        parts: dict[str, tuple[Expression, Expression]] = {}
        for term in terms:
            factors = term.operands() if term.operator() is mul_vararg else [term]
            reader, other = SR(1), SR(1)
            for factor in factors:
                symbols = symbols_of([factor]) - {self.declarations_.time}
                if not symbols & others:
                    reader = reader * factor
                elif any(self._reads_reader(item, spec) for item in symbols - others):
                    return None
                else:
                    other = other * factor
            key = str(other)
            total = parts.get(key, (other, SR(0)))[1]
            parts[key] = (other, total + reader)
        if len(parts) >= self.declarations_.blocks[self._reduced_group(reduction, spec, others)].rows:
            return None
        reduced = [
            reader * self._compiled(replace(reduction, argument=other), spec) for other, reader in parts.values()
        ]
        return SR(sum(reduced, SR(0)))

    def _compiled(self, reduction: ReductionSpec, spec: KernelSpec) -> Expression:
        """Return the placeholder of a reduction of a real expression.

        It is compiled the first time it is found.
        """
        argument = SR(self._lower(reduction.argument, spec))
        condition = None if reduction.condition is None else SR(self._lower(reduction.condition, spec))
        symbols = symbols_of([argument, *([condition] if condition is not None else [])]) - {self.declarations_.time}
        others = {
            item
            for item in symbols
            if (reference := self._reference_of(item, spec)) is not None and reference.source is ReadKind.OTHER
        }
        if any(self._reads_reader(item, spec) for item in symbols - others) and others:
            separated = self._separated(reduction, spec, argument, others) if condition is None else None
            if separated is not None:
                return separated
            return self._pairwise(reduction, spec, argument, condition, others)
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
                function=reduction.function,
            ).prepare()
            self.found_[key] = compiled
            self.sequence_.append(compiled)
            self.by_symbol_[placeholder] = compiled
            reading = spec.group or next((item.label for item in spec.sets.values()), None)
            self.readers_[placeholder] = reading if weights is not None else None
        compiled = self.found_[key]
        column = spec.set_index if compiled.weights is not None and spec.sets else 0
        self._add_reference(spec, compiled.symbol, ReferenceSpec(compiled.entry, False, column, ReadKind.REDUCED))
        return compiled.symbol

    def _tensor_entry(self, call: Expression, spec: KernelSpec) -> tuple[ReferenceSpec, EntrySpec]:
        """Return the reference and the entry of the tensor of a contraction.

        It first checks its order and readers.
        """
        tensor, *factors = call.operands()
        reference = self._reference_of(tensor, spec)
        block = self.declarations_.blocks.get(spec.group) if spec.group is not None else None
        rows = block.rows if block is not None else None
        entry = reference.entry if reference is not None else None
        if entry is None or len(entry.shape) != len(factors) + 1 or entry.shape[0] != rows:
            given = 'not a shaped parameter' if entry is None else f'of the shape {entry.shape}'
            error_msg = (
                f'The tensor {tensor} of the contraction {call} of {spec.owner} is {given}, but the contraction needs '
                f'a shaped parameter of order {len(factors) + 1} whose first index runs over the {rows or "no"} '
                'members of the reading group. Declare it `ParameterSpec(shape=(role,) * k)` on a role bound to a '
                'group.'
            )
            raise ValidationError(error_msg)
        return cast(ReferenceSpec, reference), entry

    def _axis_reads(
        self,
        factor: Expression,
        axis: int,
        sizes: tuple[int, ...],
        spec: KernelSpec,
    ) -> tuple[Expression, dict[Expression, ReadSpec]]:
        """Return the expression of one index of a contraction, with its reads.

        The contraction runs over the product of its indices. Every symbol of one member reads its value at the member
        of the index `axis` of every combination of members.
        """
        symbols = sorted(symbols_of([factor]) - {self.declarations_.time}, key=str)
        group = self._group(ReductionSpec(factor, 'sum', factor, None, None, None), spec, set(symbols))
        if group is not None and group in self.declarations_.capacity:
            error_msg = (
                f'A contraction of {spec.owner} reads the variable group {group}, whose members come and go. '
                'Contract over a group of fixed members.'
            )
            raise ValidationError(error_msg)
        renamed: dict[Expression, Expression] = {}
        reads: dict[Expression, ReadSpec] = {}
        for item in symbols:
            read = self._read(item, spec, group, sizes[axis])
            if read.per_member and len(cast(Indices, read.index)) != sizes[axis]:
                error_msg = (
                    f'The expression {factor} of index {axis + 1} of a contraction of {spec.owner} reads '
                    f'{len(cast(Indices, read.index))} members, but the tensor has {sizes[axis]} along that index.'
                )
                raise ValidationError(error_msg)
            name = symbol(f'_contracted_{axis}_{item}')
            renamed[item] = name
            if not read.per_member:
                reads[name] = read
                continue
            shape = [1] * len(sizes)
            shape[axis] = sizes[axis]
            index = np.broadcast_to(np.asarray(read.index, dtype=np.intp).reshape(shape), sizes).ravel()
            reads[name] = ReadSpec(read.is_state, np.ascontiguousarray(index))
        return SR(factor).subs(renamed), reads

    def _contraction(self, call: Expression, spec: KernelSpec) -> Expression:
        """Return the placeholder of a contraction.

        It is a sum over every combination of members weighted by the tensor.
        """
        reference, entry = self._tensor_entry(call, spec)
        key = ('contraction', str(call), spec.group)
        if key not in self.found_:
            factors = [SR(self._lower(item, spec)) for item in call.operands()[1:]]
            rows, sizes = entry.shape[0], tuple(entry.shape[1:])
            argument, reads = SR(1), dict[Expression, ReadSpec]()
            for axis, factor in enumerate(factors):
                expression, axis_reads = self._axis_reads(factor, axis, sizes, spec)
                argument, reads = argument * expression, reads | axis_reads
            size = math.prod(sizes)
            matrix = replace(entry, shape=(rows, size))
            weights = ReadSpec(reference.is_state, (entry.start + np.arange(entry.size)).astype(np.intp), matrix)
            index = len(self.found_)
            placeholder = symbol(f'_reduced_{index}')
            compiled = ReductionKernel(
                placeholder,
                'sum',
                argument,
                None,
                weights,
                entry=EntrySpec(f'__reduced_{index}', self.extent_, (rows,), RR, rows),
                size=size,
                group=spec.group,
                reads=reads,
                time=self.declarations_.time,
            ).prepare()
            self.found_[key] = compiled
            self.sequence_.append(compiled)
            self.by_symbol_[placeholder] = compiled
            self.readers_[placeholder] = spec.group
        compiled = self.found_[key]
        self._add_reference(spec, compiled.symbol, ReferenceSpec(compiled.entry, False, 0, ReadKind.REDUCED))
        return compiled.symbol

    def _neighbor_group(
        self,
        reduction: NeighborReductionSpec | HyperedgeReductionSpec,
        spec: KernelSpec,
        symbols: set[Expression],
    ) -> str:
        """Return the group whose members read a neighbourhood reduction.

        Its members are each other's neighbours.
        """
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
        """Return how a neighbourhood reduction reads a state of its edge group.

        The edge group is that of its layer, and it reads it per pair of neighbours.
        """
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
        """Return how a neighbourhood reduction reads one symbol.

        It reads it shared, of the reading member, or of the neighbour.
        """
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

    def _cell_place(self, item: Expression, spec: KernelSpec) -> tuple[int, CellComplexSpace] | None:
        """Return the degree and the cell complex of a symbol.

        It is `None` for a value shared by every cell.
        """
        if item in self.cells_:
            return self.cells_[item]
        reference = self._reference_of(item, spec)
        if reference is None or reference.source is ReadKind.SINGLE:
            return None
        block = self.declarations_.blocks.get(reference.entry.name.rpartition('.')[0])
        space = block.node.space if block is not None else None
        part = getattr(block.node.component, 'on', None) if block is not None else None
        if not isinstance(space, CellComplexSpace) or part not in DEGREES:
            error_msg = (
                f'A law of {spec.owner} applies the exterior calculus to {item}, which lives on no part of a cell '
                'complex. Place its group on the sites, edges, or hyperedges of a `CellComplexSpace`.'
            )
            raise ValidationError(error_msg)
        return DEGREES[part], space

    def _cell_argument(self, argument: Expression, spec: KernelSpec) -> tuple[int, CellComplexSpace]:
        """Return the one degree and cell complex of an exterior calculus argument.

        They are those of the symbols of the argument.
        """
        symbols = symbols_of([argument]) - {self.declarations_.time}
        places = {place for place in (self._cell_place(item, spec) for item in symbols) if place is not None}
        if len({(degree, id(space)) for degree, space in places}) != 1:
            error_msg = (
                f'A law of {spec.owner} applies the exterior calculus to {argument}, which needs the states of cells '
                f'of one degree of one complex, but it reads {len(places)}.'
            )
            raise ValidationError(error_msg)
        return next(iter(places))

    def _cell_read(self, item: Expression, spec: KernelSpec, reader: str | None = None) -> PairReadSpec:
        """Return how a kernel over the cells of a matrix reads one symbol.

        It reads it shared, of the cell, or of the reader.
        """
        if item in self.local_:
            local = self.local_[item]
            indices = (local.entry.start + np.arange(local.entry.size)).astype(np.intp)
            return PairReadSpec(ReadSpec(False, indices, local.entry), 'reader' if local.group == reader else 'other')
        if item in self.by_symbol_:
            return PairReadSpec(self._read(item, spec, None, 1), 'shared')
        reference = self._reference_of(item, spec)
        if reference is None or reference.source not in {ReadKind.MEMBER, ReadKind.SINGLE, ReadKind.OTHER}:
            error_msg = (
                f'The exterior calculus of {spec.owner} reads {item}, which is not a state or parameter of a member of '
                'a group on a cell complex or of an element.'
            )
            raise ValidationError(error_msg)
        if reference.source is ReadKind.SINGLE:
            read = ReadSpec(reference.is_state, reference.index, reference.entry, reference.column)
            return PairReadSpec(read, 'shared')
        entry = reference.entry
        indices = (entry.start + np.arange(entry.rows) * entry.columns + reference.column).astype(np.intp)
        side = 'reader' if entry.name.rpartition('.')[0] == reader else 'other'
        return PairReadSpec(ReadSpec(reference.is_state, indices, entry, reference.column, entry.columns), side)

    def _cell_kernel(
        self,
        key: tuple[object, ...],
        spec: KernelSpec,
        terms: tuple[str, Expression, Expression, str],
        matrix: scipy.sparse.csr_array,
        *,
        signed: bool = True,
    ) -> Expression:
        """Return the placeholder of a kernel of one value per row of a matrix.

        It is compiled the first time it is found.

        Args:
            key:
                The key of the kernel, unique among the neighbourhood reductions.

            spec:
                The kernel spec that reads it.

            terms:
                The kind, the argument, and the weight of the reduction, and the path of the group of its rows.

            matrix:
                The pairs of every row, with the factors of the kernel as its data.

            signed:
                Whether the data of the matrix are factors of the pairs.

        Returns:
            placeholder:
                The placeholder of the kernel.
        """
        kind, argument, weight, group = terms
        if key not in self.neighbourhoods_:
            symbols = symbols_of([argument, weight]) - {self.declarations_.time}
            reads = {
                item: self._cell_read(item, spec, group if not signed else None) for item in sorted(symbols, key=str)
            }
            indptr = matrix.indptr.astype(np.intp)
            search = StaticSearch(indptr, matrix.indices.astype(np.intp)).prepare()
            index = len(self.neighbourhoods_)
            entry = EntrySpec(f'__neighbours_{index}', self.extent_, (matrix.shape[0],), RR, matrix.shape[0])
            compiled = NeighborhoodKernel(
                symbol(f'_neighbours_{index}'),
                kind,
                argument,
                weight,
                entry,
                group=group,
                search=search,
                reads=reads,
                time=self.declarations_.time,
                coefficients=np.asarray(matrix.data, dtype=float) if signed else None,
            ).prepare()
            self.neighbourhoods_[key] = compiled
            self.sequence_.append(compiled)
            self.local_[compiled.symbol] = compiled
            self.readers_[compiled.symbol] = group
        compiled = self.neighbourhoods_[key]
        self._add_reference(spec, compiled.symbol, ReferenceSpec(compiled.entry, False, 0, ReadKind.REDUCED))
        return compiled.symbol

    def _cell_operation(self, call: Expression, spec: KernelSpec) -> Expression:
        """Return the placeholder of a call of the exterior calculus.

        It is 0 for an exterior derivative applied twice.
        """
        name = call.operator().name()
        inner = SR(call.operands()[0])
        twice = name == 'exterior_derivative' and getattr(inner.operator(), 'name', lambda: '')() == name
        argument = SR(self._lower(inner.operands()[0] if twice else inner, spec))
        degree, space = self._cell_argument(argument, spec)
        if twice:
            CellTables.exterior_derivative_of(space, degree + 1)
            return SR(0)
        if name == 'exterior_derivative':
            matrix, reached = CellTables.exterior_derivative_of(space, degree), degree + 1
        elif name == 'codifferential':
            matrix, reached = CellTables.codifferential_of(space, degree, self.declarations_), degree - 1
        else:
            matrix, reached = CellTables.hodge_star_of(space, degree, self.declarations_), degree
        group = CellTables.group_at(space, reached, self.declarations_)
        key = ('cell', name, str(argument), degree, id(space))
        placeholder = self._cell_kernel(key, spec, ('sum', argument, SR(1), group), matrix)
        self.cells_[placeholder] = (reached, space)
        return placeholder

    def _coface(self, reduction: NeighborReductionSpec, spec: KernelSpec) -> Expression:
        """Return the placeholder of a reduction over the cofaces of every member.

        The members are those of a group on a cell complex.
        """
        group = spec.group
        block = self.declarations_.blocks.get(group) if group is not None else None
        space = block.node.space if block is not None else None
        part = getattr(block.node.component, 'on', None) if block is not None else None
        if group is None or not isinstance(space, CellComplexSpace) or part not in DEGREES:
            error_msg = (
                f'The reduction {reduction.call} of {spec.owner} runs over cofaces, which needs a group on the sites, '
                'edges, or hyperedges of a `CellComplexSpace`.'
            )
            raise ValidationError(error_msg)
        argument = SR(self._lower(reduction.argument, spec))
        weight = SR(1) if reduction.weight is None else SR(self._lower(reduction.weight, spec))
        matrix = abs(CellTables.exterior_derivative_of(space, DEGREES[part])).T.tocsr()
        key = ('coface', reduction.kind, str(argument), str(weight), group)
        return self._cell_kernel(key, spec, (reduction.kind, argument, weight, group), matrix, signed=False)

    def _neighborhood(self, reduction: NeighborReductionSpec, spec: KernelSpec) -> Expression:
        """Return the placeholder of a neighbourhood reduction.

        It is compiled the first time it is found.
        """
        if isinstance(reduction.over, CofaceNeighborhood):
            return self._coface(reduction, spec)
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
            over = over.prepare()
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
            ).prepare()
            self.neighbourhoods_[key] = compiled
            self.sequence_.append(compiled)
            self.local_[compiled.symbol] = compiled
            self.readers_[compiled.symbol] = group
        compiled = self.neighbourhoods_[key]
        self._add_reference(spec, compiled.symbol, ReferenceSpec(compiled.entry, False, 0, ReadKind.REDUCED))
        return compiled.symbol

    def _hyperedges(self, reduction: HyperedgeReductionSpec, spec: KernelSpec) -> Expression:
        """Return the placeholder of a reduction over the hyperedges around every member.

        It is the sum of one reduction per size. Without a size, the reduction runs over the hyperedges of every size of
        the network: the sum and the count add the reductions of every size, and the mean divides their sum by the
        number of hyperedges around the member.

        Args:
            reduction:
                The reduction over the hyperedges around every member.

            spec:
                The kernel the reduction is read in.

        Returns:
            placeholder:
                The placeholder of the reduction.

        Raises:
            RoleError:
                The network has no hyperedge of the size of the reduction.

            ValidationError:
                The members do not live on a network with hyperedges, or a reduction other than the sum, the mean, or
                the count runs over hyperedges of several sizes.
        """
        argument = SR(reduction.argument)
        weight = SR(1) if reduction.weight is None else SR(reduction.weight)
        group = self._neighbor_group(reduction, spec, symbols_of([argument, weight]) - {self.declarations_.time})
        space = self.declarations_.blocks[group].node.space
        if not isinstance(space, NetworkSpace) or not space.hyperedges_:
            error_msg = (
                f'The hyperedge reduction {reduction.call} of {spec.owner} reads {group}, which lives on {space!r}. '
                'Give the group a network space with hyperedges.'
            )
            raise ValidationError(error_msg)
        if group in self.declarations_.capacity:
            error_msg = (
                f'The hyperedge reduction {reduction.call} of {spec.owner} reads the variable group {group}, whose '
                'members come and go. Reduce over the hyperedges of a group of fixed members.'
            )
            raise ValidationError(error_msg)
        sizes = sorted({len(item) for item in space.hyperedges_})
        if reduction.size is not None and reduction.size not in sizes:
            error_msg = (
                f'The hyperedge reduction {reduction.call} of {spec.owner} runs on hyperedges of size '
                f'{reduction.size}, but the hyperedges of the network of {group} have the sizes '
                f'{", ".join(map(str, sizes))}.'
            )
            raise RoleError(error_msg)
        if reduction.size is not None:
            sizes = [reduction.size]
        if len(sizes) == 1:
            return self._hyperedge_kernel(reduction, spec, group, (reduction.kind, argument, weight), sizes[0])
        if reduction.kind not in {'sum', 'mean', 'count'}:
            error_msg = (
                f'The hyperedge reduction {reduction.call} of {spec.owner} runs on the hyperedges of the sizes '
                f'{", ".join(map(str, sizes))}, which only the sum, the mean, and the count reduce together. Give a '
                'size.'
            )
            raise ValidationError(error_msg)
        kind = 'sum' if reduction.kind == 'mean' else reduction.kind
        parts = [self._hyperedge_kernel(reduction, spec, group, (kind, argument, weight), size) for size in sizes]
        combined = SR(sum(parts, SR(0)))
        if reduction.kind != 'mean':
            return combined
        numbers = [self._hyperedge_kernel(reduction, spec, group, ('sum', SR(1), SR(1)), size) for size in sizes]
        return combined / maximum(SR(sum(numbers, SR(0))), 1)

    def _hyperedge_kernel(
        self,
        reduction: HyperedgeReductionSpec,
        spec: KernelSpec,
        group: str,
        terms: tuple[str, Expression, Expression],
        size: int,
    ) -> Expression:
        """Return the placeholder of a reduction over the hyperedges of one size.

        It is compiled the first time it is found. Every row of the reduction is a hyperedge around a member: the
        symbols of `others` read the other members of the hyperedge through the tables of their positions, and the
        reductions over `members` are written out over the member and the other members.

        Args:
            reduction:
                The reduction over the hyperedges around every member.

            spec:
                The kernel the reduction is read in.

            group:
                The path of the group of the members.

            terms:
                The kind, the argument, and the weight of the reduction.

            size:
                The size of the hyperedges.

        Returns:
            placeholder:
                The placeholder of the reduction over the hyperedges of the size.

        Raises:
            ValidationError:
                The reduction reads more other members than a hyperedge of the size has, the members outside a
                reduction over them, the neighbour `other`, or a complex expression.
        """
        kind, argument, weight = terms
        columns, positions = HyperedgeTables._around(spec, ReadKind.OTHER)
        focal = {item: symbol(f'{item}__focal', domain_of(item)) for item in columns}
        slots = {item: [focal[item], *others[: size - 1]] for item, others in columns.items()}
        expansions = HyperedgeTables._expansions_over(slots, [argument, weight])
        argument, weight = SR(argument).subs(expansions), SR(weight).subs(expansions)
        argument, weight = SR(self._lower(argument, spec)), SR(self._lower(weight, spec))
        symbols = symbols_of([argument, weight]) - {self.declarations_.time}
        beyond = sorted(str(item) for item in symbols if positions.get(item, -1) >= size - 1)
        outside = sorted(str(item) for item in symbols if item in columns)
        if beyond or outside or argument.has(I) or weight.has(I):
            error_msg = (
                f'The hyperedge reduction {reduction.call} of {spec.owner} reads '
                f'{", ".join([*beyond, *outside]) or "a complex expression"}. A reduction over hyperedges of size '
                f'{size} reads the {size - 1} other members `others`, and `members` inside reductions over them, of '
                'real expressions.'
            )
            raise ValidationError(error_msg)
        block = self.declarations_.blocks[group]
        key = ('hyperedges', kind, str(argument), str(weight), size, group)
        if key not in self.neighbourhoods_:
            indptr, tables = HyperedgeTables._incidences(cast(NetworkSpace, block.node.space), size)
            search = StaticSearch(indptr, np.arange(indptr[-1], dtype=np.intp)).prepare()
            sources = {value: item for item, value in focal.items()}
            reads = {
                item: self._hyperedge_read(item, spec, group, search, (sources, positions, tables))
                for item in sorted(symbols, key=str)
            }
            index = len(self.neighbourhoods_)
            entry = EntrySpec(f'__neighbours_{index}', self.extent_, (block.rows,), RR, block.rows)
            compiled = NeighborhoodKernel(
                symbol(f'_neighbours_{index}'),
                kind,
                argument,
                weight,
                entry,
                group=group,
                search=search,
                reads=reads,
                time=self.declarations_.time,
            ).prepare()
            self.neighbourhoods_[key] = compiled
            self.sequence_.append(compiled)
            self.local_[compiled.symbol] = compiled
            self.readers_[compiled.symbol] = group
        compiled = self.neighbourhoods_[key]
        self._add_reference(spec, compiled.symbol, ReferenceSpec(compiled.entry, False, 0, ReadKind.REDUCED))
        return compiled.symbol

    def _hyperedge_read(
        self,
        item: Expression,
        spec: KernelSpec,
        group: str,
        search: StaticSearch,
        slots: tuple[dict[Expression, Expression], dict[Expression, int], list[Indices]],
    ) -> PairReadSpec:
        """Return how a reduction over hyperedges reads one symbol.

        It reads it of the member, of another member, or shared.

        Args:
            item:
                The symbol the reduction reads.

            spec:
                The kernel the reduction is read in.

            group:
                The path of the group of the members.

            search:
                The search of the symbols of the system.

            slots:
                The sources of the slots, their positions, and the tables of the positions.

        Returns:
            read:
                How the reduction reads the symbol.

        Raises:
            ValidationError:
                The symbol is the neighbour `other` of the member, which no hyperedge has.
        """
        sources, positions, tables = slots
        if item not in sources and item not in positions:
            reference = self._reference_of(item, spec)
            if reference is not None and reference.source is ReadKind.OTHER:
                error_msg = (
                    f'The hyperedge reduction of {spec.owner} reads {item}, the neighbour `other` of the member. Read '
                    'the other members of a hyperedge with `others`.'
                )
                raise ValidationError(error_msg)
            return self._pair_read(item, spec, group, search)
        reference = spec.references[sources.get(item, item)]
        entry = reference.entry
        indices = (entry.start + np.arange(entry.rows) * entry.columns + reference.column).astype(np.intp)
        read = ReadSpec(reference.is_state, indices, entry, reference.column, entry.columns)
        if item in sources:
            return PairReadSpec(read, 'reader')
        return PairReadSpec(read, 'other', table=tables[positions[item]])
