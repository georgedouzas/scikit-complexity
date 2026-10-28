"""Settle the rows every kernel spec is evaluated over, after checking where its laws draw random values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from sage.all__sagemath_symbolics import Expression

from ....core.laws import BaseLaw, BoundaryLaw, CrossingLaw, EquationLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.laws.utils import draws_in, neighborhoods_in, reductions_in
from ....errors import IncompatibleLawsError, RoleError
from ..._utils import components_of
from ...problems._read_kind import ReadKind
from .._base import BasePass
from .._compilation import BlockLaws, Laws
from .._draw_lowering import DrawLowering
from .._kernel_spec import KernelSpec
from .._matrix_law import _RANDOM_FIELDS
from .._renewal_law import RenewalLaw


class RowsPass(BasePass):
    """Pass that checks the draws of the laws and settles the rows of every kernel spec, with its laws set apart."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'apart')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered',)

    def _run(self, inputs: object = None) -> None:
        """Check the draws and settle the rows of every kernel spec."""
        compilation = self.compilation_
        apart = cast('Laws', compilation.apart)
        for spec, laws, _ in cast('BlockLaws', compilation.lowered):
            self._check_draw_placement(spec.owner, laws)
            self._settle_rows(spec, [*laws, *(law for other, items in apart if other is spec for law in items)])

    @staticmethod
    def _check_aligned(spec: KernelSpec, groups: list[str]) -> None:
        """Raise an error when the groups an interaction reads member by member differ in size."""
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
    def _settle_rows(spec: KernelSpec, laws: list[BaseLaw]) -> None:
        """Set the groups an interaction is evaluated over from the member symbols its laws read outside reductions.

        Laws without reductions keep the group their one group role binds to, so a single element bound next to it
        receives the sum over the members. Groups of one size that laws read outside reductions are read member by
        member, row `i` of each together.
        """
        values = [law.rate if isinstance(law, JumpLaw) else getattr(law, 'value', None) for law in laws]
        if not spec.groups or (len(spec.groups) == 1 and not any(reductions_in(value) for value in values)):
            return
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
        spec.group = next(iter(groups), None)
        spec.aligned = tuple(groups)
        spec.rows = spec.groups[spec.group] if spec.group is not None else 1

    @staticmethod
    def _check_draw_placement(owner: str, laws: list[BaseLaw]) -> None:
        """Raise an error when a law other than an update or a renewal law holds a random draw, or a draw is unknown.

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
            values = [
                value for value in law._get_values(include=_RANDOM_FIELDS) if DrawLowering._is_symbolic(value)
            ]
            found = draws_in(values)
            if found and not isinstance(law, UpdateLaw | RenewalLaw):
                error_msg = (
                    f'The {type(law).__name__} of {owner} holds the random draw {found[0].call}, but only update and '
                    'renewal laws take random draws. Model randomness in continuous time with Noise laws for Wiener '
                    'processes, Jump laws for random events, or Renewal laws for events after drawn waiting times.'
                )
                raise IncompatibleLawsError(error_msg)
            for item in found:
                if item.kind == 'draw':
                    DrawLowering._distribution_of(owner, f'{type(law).__name__}', item)
