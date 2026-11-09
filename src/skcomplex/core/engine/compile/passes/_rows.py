"""Settle the rows every kernel spec is evaluated over.

It first checks where its laws draw random values.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from sage.all__sagemath_symbolics import Expression

from ....components.laws import BaseLaw, BoundaryLaw, CrossingLaw, EquationLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....components.laws.utils._draws import draws_in
from ....components.laws.utils._reductions import hyperedges_in, neighborhoods_in, reductions_in
from ....components.spaces import DiscreteTime
from ....errors import IncompatibleLawsError, RoleError
from ....specs._matrix_law_spec import _RANDOM_FIELDS
from ..._utils import components_of
from ...problems._kernel_spec import KernelSpec
from ...types import ReadKind
from .._base import BasePass
from .._compilation import Compilation
from .._draw_lowering import DrawLowering
from .._renewal_law import RenewalLaw


class RowsPass(BasePass):
    """Pass that checks the draws of the laws and settles the rows of every spec.

    The specs are kernel specs, with their laws set apart.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'apart', 'transfer')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered',)

    def _run(self, inputs: object = None) -> None:
        """Check the draws and settle the rows of every kernel spec.

        A spec whose rows change is replaced by a copy with its rows settled in every field of the compilation, such as
        the laws, the laws set apart, and the flows, which later passes read.
        """
        compilation = self.compilation_
        apart = Compilation._required(compilation.apart, 'apart')
        lowered = Compilation._required(compilation.lowered, 'lowered')
        stepped = self._stepped_flows(compilation, [laws for _, laws, _ in lowered])
        settled: dict[int, KernelSpec] = {}
        for spec, laws, _ in lowered:
            self._check_draw_placement(spec.owner, [law for law in laws if id(law) not in stepped])
            new = self._settled(spec, [*laws, *(law for other, items in apart if other is spec for law in items)])
            if new is not spec:
                settled[id(spec)] = new
        compilation._replace_specs(settled)

    @staticmethod
    def _stepped_flows(compilation: Compilation, laws: list[list[BaseLaw]]) -> set[int]:
        """Return the identities of the rate laws of flows that move once per step.

        They move their amounts once per step and may draw. The flows of a system that declares discrete time, or that
        holds update laws, become updates.
        """
        born = compilation.transfer.born if compilation.transfer is not None else {}
        declared = isinstance(compilation.system.time, DiscreteTime)
        if not declared and not any(
            isinstance(law, UpdateLaw) and id(law) not in born for items in laws for law in items
        ):
            return set()
        return {id(law) for items in laws for law in items if id(law) in born and isinstance(law, RateLaw)}

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
    def _outside_reductions(value: object) -> set[Expression]:
        """Return the symbols of a law value that occur outside its reductions."""
        expressions = [item for item in components_of(value) if isinstance(item, Expression)]
        found = [reduction.call for item in expressions for reduction in reductions_in(item)]
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
    def _settled(spec: KernelSpec, laws: list[BaseLaw]) -> KernelSpec:
        """Return a spec with the groups it is evaluated over.

        They are set from the member symbols its laws read. Only the symbols outside reductions count.

        Laws without reductions keep the group their one group role binds to, so a single element bound next to it
        receives the sum over the members. Groups of one size that laws read outside reductions are read member by
        member, row `i` of each together.
        """
        values = [law.rate if isinstance(law, JumpLaw) else getattr(law, 'value', None) for law in laws]
        if not spec.groups or (len(spec.groups) == 1 and not any(reductions_in(value) for value in values)):
            return spec
        read: set[Expression] = set()
        for law, value in zip(laws, values, strict=True):
            read |= RowsPass._outside_reductions(value)
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
                RateLaw | NoiseLaw | BoundaryLaw | UpdateLaw | JumpLaw | EquationLaw | CrossingLaw | RenewalLaw,
            ):
                continue
            values = [value for value in law._get_values(include=_RANDOM_FIELDS) if DrawLowering._is_symbolic(value)]
            found = draws_in(values)
            if found and not isinstance(law, UpdateLaw | RenewalLaw):
                error_msg = (
                    f'The {type(law).__name__} of {owner} holds the random draw {found[0].call}, but only update and '
                    'renewal laws take random draws. Express randomness in continuous time with Noise laws for Wiener '
                    'processes, Jump laws for random events, or Renewal laws for events after drawn waiting times.'
                )
                raise IncompatibleLawsError(error_msg)
            for item in found:
                if item.kind == 'draw':
                    DrawLowering._distribution_of(owner, f'{type(law).__name__}', item)
