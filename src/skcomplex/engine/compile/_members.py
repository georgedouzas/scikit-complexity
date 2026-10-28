"""Lower reads of group members at an offset along the group order to neighbour reads of an index table."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ...core.errors import ValidationError
from ...core.spaces import LatticeNeighborhood, LatticeSpace
from ...core.spaces._lattice import BOUNDARIES
from ..problems import LATTICE_BOUNDARIES, MODES, NeighborRead, Reference, Source, domain_of, symbol, vector
from ._base import BasePass

_SHIFT = 'shift_member'


@dataclass
class MemberShifts:
    """Member offset reads of the laws of one component, with the symbols that read the offset members.

    Args:
        owner:
            The path of the component.

        references:
            The references of the component's symbols, extended with the offset reads.

        time:
            The time symbol.
    """

    owner: str
    references: dict[Expression, Reference]
    time: Expression
    shifted: dict[Expression, tuple[Expression, int, str, float]] = field(default_factory=dict)

    def _member(self, item: Expression) -> tuple[Expression, int, str | None, float]:
        """Return the member symbol a symbol reads, with the offset, boundary, and fill it is already read at."""
        if item in self.shifted:
            return self.shifted[item]
        reference = self.references.get(item)
        if reference is None or reference.source is not Source.MEMBER:
            name = reference.entry.name if reference is not None else str(item)
            error_msg = (
                f'A law of {self.owner} reads {name} at a member offset, but shift_member reads only states and '
                'parameters of group members, bound through a role.'
            )
            raise ValidationError(error_msg)
        return item, 0, None, 0.0

    def _read(self, base: Expression, offset: int, boundary: str, fill: float) -> Expression:
        """Return the symbol that reads a member symbol at an offset, registering its reference the first time."""
        if not offset:
            return base
        key = (base, offset, boundary, fill)
        for item, value in self.shifted.items():
            if value == key:
                return item
        reference = self.references[base]
        target = symbol(f'{base}__shift{len(self.shifted)}', domain_of(base))
        read = self._neighbor_read(reference.entry.rows, offset, boundary, fill)
        self.references[target] = Reference(
            reference.entry,
            reference.is_state,
            reference.column,
            Source.NEIGHBOR,
            read,
        )
        self.shifted[target] = key
        return target

    def shift(self, expression: Expression, offset: int, boundary: str, fill: float) -> Expression:
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
        members = {item: self._member(item) for item in expression.variables() if item != self.time}
        groups = {self.references[base].entry.name.rpartition('.')[0] for base, *_ in members.values()}
        if len(groups) > 1:
            error_msg = (
                f'A law of {self.owner} reads the groups {", ".join(sorted(groups))} at one member offset. '
                'shift_member reads the members of one group.'
            )
            raise ValidationError(error_msg)
        replacements = {}
        for item, (base, current, inner, inner_fill) in members.items():
            if inner is not None and (inner, inner_fill) != (boundary, fill):
                error_msg = (
                    f'A law of {self.owner} nests member offsets with the boundaries {inner!r} and {boundary!r}. '
                    'Nested offsets add and take one boundary.'
                )
                raise ValidationError(error_msg)
            replacements[item] = self._read(base, current + offset, boundary, fill)
        return expression.subs(replacements)

    def lower(self, value: object) -> object:
        """Return a law value with every member offset read replaced by symbols that read the offset members.

        Args:
            value:
                The value of a law.

        Returns:
            lowered:
                The value with neighbour reads.
        """
        if isinstance(value, FreeModuleElement):
            return vector(self.lower(item) for item in value)
        if not isinstance(value, Expression) or not BasePass._calls_of(value, _SHIFT):
            return value
        operator = value.operator()
        if getattr(operator, 'name', lambda: '')() != _SHIFT:
            return operator(*(self.lower(item) for item in value.operands()))
        expression, offset, boundary, fill = self._shift_arguments(value)
        return self.shift(SR(self.lower(expression)), offset, boundary, fill)

    @staticmethod
    def _neighbor_read(rows: int, offset: int, boundary: str, fill: float) -> NeighborRead:
        """Return the read of the member at an offset of every member of a group of `rows` members."""
        lattice = LatticeSpace((rows,), boundary=boundary, fill=fill).check().prepare()
        table = lattice._index_table(LatticeNeighborhood(((offset,),)))[:, 0]
        ghost = fill if boundary == 'fixed' else None
        kind = next(kind for kind, name in LATTICE_BOUNDARIES.items() if name == boundary)
        return NeighborRead(table, ghost, (rows,), (offset,), MODES[kind])

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
            >>> from skcomplex.engine.problems import symbol
            >>> from skcomplex.engine.compile._members import MemberShifts
            >>> from skcomplex.core.laws.utils import shift_member
            >>> MemberShifts._shift_arguments(shift_member(symbol('x'), -2, boundary='fixed', fill=1.0))
            (x, -2, 'fixed', 1.0)
        """
        expression, offset, boundary, fill = call.operands()
        return expression, int(offset), BOUNDARIES[int(boundary)], float(fill)
