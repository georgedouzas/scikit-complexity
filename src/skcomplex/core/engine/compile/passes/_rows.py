"""Settle the rows every kernel spec is evaluated over."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....components.laws import BaseLaw, ConstraintLaw, EventLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....components.laws._boundary import _BoundaryLaw
from ....components.laws._flow_term import FlowTermLaw
from ....components.laws._linear_program import _LinearProgramLaw
from ....components.laws.utils._cells import cell_operations_in
from ....components.laws.utils._draws import draws_in
from ....components.laws.utils._phase import axis_reductions_in
from ....components.laws.utils._reductions import hyperedges_in, neighborhoods_in, reductions_in
from ....components.spaces import PhaseSpace
from ....components.spaces._discrete_time import _DiscreteTime
from ....errors import IncompatibleLawsError, RoleError
from ..._utils import components_of
from ...enums import ReadKind
from ...problems._declarations_spec import DeclarationsSpec
from ...problems._kernel_spec import KernelSpec
from ...problems._neighbor_read_spec import _NeighborReadSpec
from .._base import BasePass
from .._compilation import Compilation
from .._draw_lowering import DrawLowering
from .._matrix_law_spec import RANDOM_FIELDS
from .._renewal_law import RenewalLaw


class RowsPass(BasePass):
    """A pass that checks the draws of the laws and settles the rows of every spec.

    The specs are kernel specs, with their laws set apart.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'apart', 'programs', 'transfer', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered',)

    def _run(self, inputs: object = None) -> None:
        """Check the draws and settle the rows of every kernel spec.

        A spec whose rows change is replaced by a copy with its rows settled in every field of the compilation, such as
        the laws, the laws set apart, and the flows, which later passes read. The linear programs of a spec settle its
        rows with its other laws. A field over position that laws of a phase-space field read is read at every class
        of its position first.
        """
        compilation = self.compilation_
        declarations = Compilation._get_required(compilation.layout, 'layout')
        apart = [
            *Compilation._get_required(compilation.apart, 'apart'),
            *Compilation._get_required(compilation.programs, 'programs'),
        ]
        lowered = Compilation._get_required(compilation.lowered, 'lowered')
        stepped = self._get_stepped_flows(compilation, [laws for _, laws, _ in lowered])
        settled: dict[int, KernelSpec] = {}
        for spec, laws, _ in lowered:
            self._check_draw_placement(spec.owner, [law for law in laws if id(law) not in stepped])
            every = [*laws, *(law for other, items in apart if other is spec for law in items)]
            new = self._get_settled(self._get_broadcast(spec, every, declarations), every)
            if new is not spec:
                settled[id(spec)] = new
        compilation._replace_specs(settled)

    @staticmethod
    def _get_stepped_flows(compilation: Compilation, laws: list[list[BaseLaw]]) -> set[int]:
        """Return the identities of the rate laws of flows that move once per step.

        They move their amounts once per step and may draw. The flows of a system that declares discrete time, or that
        holds update laws or flows of increments, become updates.
        """
        born = compilation.transfer.born if compilation.transfer is not None else {}
        declared = isinstance(compilation.system.time, _DiscreteTime)
        if not declared and not any(
            (isinstance(law, UpdateLaw) and id(law) not in born) or (isinstance(law, FlowTermLaw) and law.increment)
            for items in laws
            for law in items
        ):
            return set()
        return {id(law) for items in laws for law in items if id(law) in born and isinstance(law, RateLaw)}

    @staticmethod
    def _get_broadcast(spec: KernelSpec, laws: list[BaseLaw], declarations: DeclarationsSpec) -> KernelSpec:
        """Return a spec that reads a field over position at every class of a phase space.

        The phase space is that of the field its laws change.

        A drift of a phase-space field by a force over its position lattice, such as `s.force.x * s.target.x`, reads
        the force of the position of every class. A spec without such a field is returned itself.
        """
        targets = {
            reference.entry.name.rpartition('.')[0]
            for law in laws
            for target in law._get_targets()
            for item in components_of(target)
            if (reference := spec.references.get(item)) is not None
        }
        phases = [
            space
            for path in targets
            if path in declarations.blocks
            and isinstance(space := getattr(declarations.blocks[path].node.component, 'space', None), PhaseSpace)
        ]
        if not phases:
            return spec
        phase = phases[0]
        references = dict(spec.references)
        for item, reference in spec.references.items():
            owner = reference.entry.name.rpartition('.')[0]
            block = declarations.blocks.get(owner)
            if reference.source is not ReadKind.MEMBER or owner in targets or block is None:
                continue
            if (
                isinstance(getattr(block.node.component, 'space', None), PhaseSpace)
                or block.rows != phase.position_size_
            ):
                continue
            table = np.repeat(np.arange(block.rows), phase.size_ // phase.position_size_)
            read = _NeighborReadSpec(table, None, None, None, 0)
            references[item] = replace(reference, source=ReadKind.NEIGHBOR, neighbor=read)
        return replace(spec, references=references) if references != spec.references else spec

    @staticmethod
    def _check_aligned(spec: KernelSpec, groups: list[str]) -> None:
        """Raise an error for groups of unequal size an interaction reads member-wise.

        The groups an interaction reads member by member must have one size.
        """
        sizes = {group: spec.groups[group] for group in groups}
        if len(set(sizes.values())) <= 1:
            return
        described = ', '.join(f'{group} of {size} members' for group, size in sizes.items())
        error_msg = (
            f'Interaction {spec.owner} reads the groups {described} member by member, which needs groups of one size. '
            'Read the other groups through reductions, or couple every pair of members with an interaction over '
            "'pairs'."
        )
        raise RoleError(error_msg)

    @staticmethod
    def _get_outside_reductions(value: object) -> set[Expression]:
        """Return the symbols of a law value that occur outside its reductions.

        A call of the exterior calculus, such as `exterior_derivative(s.source.u)`, reads the cells of another degree
        through the incidences of the complex, and an integral over class axes, such as the moment of a phase-space
        field, reads every class of a position, so both count as reductions.
        """
        expressions = [item for item in components_of(value) if isinstance(item, Expression)]
        found = [reduction.call for item in expressions for reduction in reductions_in(item)]
        found += [call for item in expressions for call in cell_operations_in(item)]
        found += [call for item in expressions for call in axis_reductions_in(item)]
        found += [reduction.call for item in expressions for reduction in neighborhoods_in(item)]
        found += [reduction.call for item in expressions for reduction in hyperedges_in(item)]
        calls, operators = {str(call) for call in found}, {call.operator() for call in found}
        symbols: set[Expression] = set()
        stack = list(expressions)
        while stack:
            item = stack.pop()
            if item.operator() is None:
                symbols |= set(item.variables())
            elif item.operator() not in operators or str(item) not in calls:
                stack.extend(item.operands())
        return symbols

    @staticmethod
    def _get_read_values(law: BaseLaw) -> list[object]:
        """Return the values of a law whose symbols settle the rows of its spec.

        A jump law gives its rate, and a linear program its constraints, its objectives, and the condition under which
        it is solved.
        """
        if isinstance(law, JumpLaw):
            return [law.rate]
        if isinstance(law, _LinearProgramLaw):
            return [*law.constraints, *(objective.expression for objective in law.objectives), law.active]
        return [getattr(law, 'value', None)]

    @staticmethod
    def _get_settled(spec: KernelSpec, laws: list[BaseLaw]) -> KernelSpec:
        """Return a spec with the groups it is evaluated over.

        They are set from the member symbols its laws read. Only the symbols outside reductions count.

        Laws without reductions keep the group their one group role binds to, so a single element bound next to it
        receives the sum over the members. Groups of one size that laws read outside reductions are read member by
        member, row `i` of each together.
        """
        values = [value for law in laws for value in RowsPass._get_read_values(law)]
        if not spec.groups or (len(spec.groups) == 1 and not any(reductions_in(value) for value in values)):
            return spec
        read: set[Expression] = set()
        for value in values:
            read |= RowsPass._get_outside_reductions(value)
        for law in laws:
            read |= {item for target in law._get_targets() for item in components_of(target)}
        owners = {
            reference.entry.name.rpartition('.')[0]
            for item in read
            if (reference := spec.references.get(item)) is not None and reference.source is ReadKind.MEMBER
        }
        groups = sorted(owners & set(spec.groups))
        RowsPass._check_aligned(spec, groups)
        group = next(iter(groups), None)
        rows = spec.groups[group] if group is not None else 1
        return replace(spec, group=group, aligned=tuple(groups), rows=rows)

    @staticmethod
    def _check_draw_placement(owner: str, laws: list[BaseLaw]) -> None:
        """Raise an error for a misplaced or unknown random draw.

        A draw is misplaced in a law other than an update or a renewal law.

        Args:
            owner:
                The path of the component the laws belong to.

            laws:
                The laws of the component.

        Raises:
            IncompatibleLawsError:
                A rate, noise, jump, boundary, equation, or event law holds a draw.

            ValidationError:
                A draw names a distribution that `scipy.stats` does not provide, or keywords it does not take.
        """
        for law in laws:
            if not isinstance(
                law,
                RateLaw | NoiseLaw | _BoundaryLaw | UpdateLaw | JumpLaw | ConstraintLaw | EventLaw | RenewalLaw,
            ):
                continue
            values = [value for value in law._get_values(include=RANDOM_FIELDS) if DrawLowering._is_symbolic(value)]
            found = draws_in(values)
            if found and not isinstance(law, UpdateLaw | RenewalLaw):
                error_msg = (
                    f'The {law._get_name()} of {owner} holds the random draw {found[0].call}, but only update and '
                    'renewal laws take random draws. Express randomness in continuous time with Noise laws for Wiener '
                    'processes, Jump laws for random events, or Renewal laws for events after drawn waiting times.'
                )
                raise IncompatibleLawsError(error_msg)
            for item in found:
                if item.kind == 'draw':
                    DrawLowering._get_distribution_of(owner, f'{law._get_name()}', item)
