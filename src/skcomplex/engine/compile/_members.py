"""Lower reads of group members at an offset along the group order to neighbour reads of an index table."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from typing import Self

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ...core.laws import BaseLaw, BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ...core.spaces import LatticeNeighborhood, LatticeSpace
from ...core.spaces._lattice import BOUNDARIES
from ...errors import ValidationError
from ...specs import NeighborReadSpec
from .._utils import domain_of, symbol, vector
from ..problems import KernelSpec, ReferenceSpec
from ..problems._law_kernel import LATTICE_BOUNDARIES, MODES
from ..types import ReadKind
from ._base import BaseLowering, BasePass
from ._compilation import Compilation

_SHIFT = 'shift_member'


class MemberLowering(BaseLowering[list[BaseLaw]]):
    """Lowering of the member offset reads of the laws of one component to neighbour reads of its group's index table.

    `prepare(compilation)` reads the time of the compilation, and `run(laws)` returns the laws with every member offset
    read replaced by a symbol that reads the offset member, whose reference it notes for the spec.

    Args:
        spec:
            The kernel spec of the component, which `_apply_references` replaces by a copy with the offset reads.

    Attributes:
        time_:
            The time symbol.

        shifted_:
            The member symbol, offset, boundary, and fill that every offset read reads.
    """

    time_: Expression
    shifted_: dict[Expression, tuple[Expression, int, str, float]]

    def __init__(self, spec: KernelSpec) -> None:
        self.spec = spec

    def _prepare(self, context: Compilation | None) -> Self:
        """Read the time of the compilation and start with no offset reads.

        Args:
            context:
                The compilation.

        Returns:
            self:
                The prepared lowering.

        Raises:
            ValidationError:
                The compilation has no layout yet.
        """
        super()._prepare(context)
        layout = self.compilation_.layout
        if layout is None:
            error_msg = 'The compilation has no layout yet. Lower member offsets after the layout pass.'
            raise ValidationError(error_msg)
        self.time_ = layout.time
        self.shifted_ = {}
        return self

    def _run(self, inputs: list[BaseLaw] | None) -> list[BaseLaw]:
        """Return laws whose member offset reads are lowered to neighbour reads of the group's index table.

        A jump keeps a change whose target is not a symbolic expression, such as a request of another formalism.

        Args:
            inputs:
                The laws of the component.

        Returns:
            laws:
                The lowered laws.
        """
        result: list[BaseLaw] = []
        for law in inputs or []:
            if isinstance(law, JumpLaw):
                changes: dict[Expression, float] = defaultdict(float)
                for target, increment in (law.changes or {}).items():
                    changes[SR(self._lower(target)) if isinstance(target, Expression) else target] += increment
                result.append(JumpLaw(self._lower(law.rate), dict(changes)))
            elif isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
                result.append(BasePass._replaced(law, value=self._lower(law.value)))
            else:
                result.append(law)
        return result

    def _member(self, item: Expression) -> tuple[Expression, int, str | None, float]:
        """Return the member symbol a symbol reads, with the offset, boundary, and fill it is already read at."""
        if item in self.shifted_:
            return self.shifted_[item]
        reference = self.spec.references.get(item)
        if reference is None or reference.source is not ReadKind.MEMBER:
            name = reference.entry.name if reference is not None else str(item)
            error_msg = (
                f'A law of {self.spec.owner} reads {name} at a member offset, but shift_member reads only states and '
                'parameters of group members, bound through a role.'
            )
            raise ValidationError(error_msg)
        return item, 0, None, 0.0

    def _read(self, base: Expression, offset: int, boundary: str, fill: float) -> Expression:
        """Return the symbol that reads a member symbol at an offset, registering its reference the first time."""
        if not offset:
            return base
        key = (base, offset, boundary, fill)
        for item, value in self.shifted_.items():
            if value == key:
                return item
        reference = self.spec.references[base]
        target = symbol(f'{base}__shift{len(self.shifted_)}', domain_of(base))
        read = self._neighbor_read(reference.entry.rows, offset, boundary, fill)
        self._add_reference(
            self.spec,
            target,
            ReferenceSpec(reference.entry, reference.is_state, reference.column, ReadKind.NEIGHBOR, read),
        )
        self.shifted_[target] = key
        return target

    def _shift(self, expression: Expression, offset: int, boundary: str, fill: float) -> Expression:
        """Return an expression read at the member an offset along the group order of every member.

        Args:
            expression:
                The expression of member symbols, possibly already read at an offset.

            offset:
                The number of members along the group order.

            boundary:
                `'periodic'`, `'reflecting'`, or `'fixed'`.

            fill:
                The value read past the ends of a fixed boundary.

        Returns:
            shifted:
                The expression in symbols that read the offset members.

        Raises:
            ValidationError:
                The expression reads a single element, two groups, or nests reads with different boundaries.
        """
        expression = SR(expression)
        members = {item: self._member(item) for item in expression.variables() if item != self.time_}
        groups = {self.spec.references[base].entry.name.rpartition('.')[0] for base, *_ in members.values()}
        if len(groups) > 1:
            error_msg = (
                f'A law of {self.spec.owner} reads the groups {", ".join(sorted(groups))} at one member offset. '
                'shift_member reads the members of one group.'
            )
            raise ValidationError(error_msg)
        replacements = {}
        for item, (base, current, inner, inner_fill) in members.items():
            if inner is not None and (inner, inner_fill) != (boundary, fill):
                error_msg = (
                    f'A law of {self.spec.owner} nests member offsets with the boundaries {inner!r} and {boundary!r}. '
                    'Nested offsets add and take one boundary.'
                )
                raise ValidationError(error_msg)
            replacements[item] = self._read(base, current + offset, boundary, fill)
        return expression.subs(replacements)

    def _lower(self, value: object) -> object:
        """Return a law value with every member offset read replaced by symbols that read the offset members.

        Args:
            value:
                The value of a law.

        Returns:
            lowered:
                The value with neighbour reads.
        """
        if isinstance(value, FreeModuleElement):
            return vector(self._lower(item) for item in value)
        if not isinstance(value, Expression) or not BasePass._calls_of(value, _SHIFT):
            return value
        operator = value.operator()
        if getattr(operator, 'name', lambda: '')() != _SHIFT:
            return operator(*(self._lower(item) for item in value.operands()))
        expression, offset, boundary, fill = self._shift_arguments(value)
        return self._shift(SR(self._lower(expression)), offset, boundary, fill)

    @staticmethod
    def _neighbor_read(rows: int, offset: int, boundary: str, fill: float) -> NeighborReadSpec:
        """Return the read of the member at an offset of every member of a group of `rows` members."""
        lattice = LatticeSpace((rows,), boundary=boundary, fill=fill).check().prepare()
        table = lattice._index_table(LatticeNeighborhood(((offset,),)))[:, 0]
        ghost = fill if boundary == 'fixed' else None
        kind = next(kind for kind, name in LATTICE_BOUNDARIES.items() if name == boundary)
        return NeighborReadSpec(table, ghost, (rows,), (offset,), MODES[kind])

    @staticmethod
    def _shift_arguments(call: Expression) -> tuple[Expression, int, str, float]:
        """Return the expression, the offset, the boundary, and the fill of a member offset read.

        Args:
            call:
                A call returned by `shift_member`.

        Returns:
            arguments:
                The expression of member symbols, the number of members along the group order, the boundary, and the
                value read past the ends of a fixed boundary.

        Examples:
            >>> from skcomplex.engine._utils import symbol
            >>> from skcomplex.engine.compile._members import MemberLowering
            >>> from skcomplex.core.laws.utils import shift_member
            >>> MemberLowering._shift_arguments(shift_member(symbol('x'), -2, boundary='fixed', fill=1.0))
            (x, -2, 'fixed', 1.0)
        """
        expression, offset, boundary, fill = call.operands()
        return expression, int(offset), BOUNDARIES[int(boundary)], float(fill)
