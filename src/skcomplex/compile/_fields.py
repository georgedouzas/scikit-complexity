"""Lower differential operators on fields to finite-difference stencils and graph Laplacians."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..core import Expression, ValidationError, domain_of, symbol
from ..model import Field, LatticeSpace, Neighborhood, NetworkSpace, where
from ._kernels import Reference, Source

type Indices = npt.NDArray[np.intp]
type Offset = tuple[int, ...]

MODES = {'periodic': 0, 'zero-flux': 1, 'fixed': 2}
LATTICE_BOUNDARIES = {'periodic': 'periodic', 'zero-flux': 'reflecting', 'fixed': 'fixed'}


@dataclass(frozen=True, eq=False)
class NeighborRead:
    """How a symbol reads a field at a neighbouring site of every member.

    Args:
        table:
            The member index of the neighbour of every member, with the number of members for the fixed value.

        ghost:
            The value read outside fixed edges, or `None` when no neighbour is outside.

        sizes:
            The lattice shape, or `None` on a network.

        steps:
            The offset of the neighbour on a lattice, or `None` on a network.

        mode:
            0 for periodic, 1 for zero-flux, and 2 for fixed edges.
    """

    table: Indices
    ghost: float | None
    sizes: tuple[int, ...] | None
    steps: Offset | None
    mode: int

    def read(self, column: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the neighbour's value of every member from the column of values.

        Args:
            column:
                One value per member.

        Returns:
            values:
                The value at the neighbour of every member.
        """
        if self.ghost is None:
            return column[self.table]
        return np.append(column, self.ghost)[self.table]


def _tag(offset: Offset | int) -> str:
    """Return a name fragment for a lattice offset or a network neighbour slot."""
    if isinstance(offset, int):
        return f'k{offset}'
    return '_'.join(f'{"m" if step < 0 else "p"}{abs(step)}' for step in offset)


@dataclass
class FieldLowering:
    """Stencils of the operators in the laws of one component, with the symbols they read.

    Args:
        owner:
            The path of the component.

        references:
            The references of the component's symbols, extended with the neighbour reads.

        fields:
            The field of every member entry by entry name.

        initial:
            The initial value of every symbol that can be evaluated, for stability limits.
    """

    owner: str
    references: dict[Expression, Any]
    fields: Mapping[str, Field]
    initial: Mapping[Expression, float] = field(default_factory=dict)
    shifted: dict[Expression, tuple[Expression, Offset | int]] = field(default_factory=dict)
    limits: list[float] = field(default_factory=list)
    _tables: dict[tuple[Any, ...], tuple[Indices, float | None]] = field(default_factory=dict)

    def _field_of(self, item: Expression) -> Field | None:
        """Return the field whose member a symbol reads, or `None` for a single value."""
        reference = self.references[item]
        if reference.source is not Source.MEMBER:
            return None
        found = self.fields.get(reference.entry.name)
        if found is None:
            error_msg = (
                f'A law of {self.owner} applies a differential operator to {item}, which is not a state or parameter '
                'of a field. Create the group with `Field`.'
            )
            raise ValidationError(error_msg)
        return found

    def _table(self, item: Expression, place: Field, offset: Offset | int) -> tuple[Indices, float | None, int]:
        """Return the neighbour table, the fixed value, and the mode of a symbol at an offset."""
        reference = self.references[item]
        variable = reference.entry.name.rpartition('.')[2]
        if reference.is_state:
            kind, value = place.boundary_of(variable)
        else:
            kind, value = (
                'periodic' if getattr(place.space, 'boundary', 'periodic') == 'periodic' else 'zero-flux',
                0.0,
            )
        key = (id(place), kind, value, offset)
        if key not in self._tables:
            space = place.space
            if isinstance(space, NetworkSpace):
                nodes = list(space.graph.nodes)
                index = {node: position for position, node in enumerate(nodes)}
                neighbours = [[index[other] for other in space.graph.neighbors(node)] for node in nodes]
                width = max((len(items) for items in neighbours), default=0)
                table = np.asarray(
                    [items + [position] * (width - len(items)) for position, items in enumerate(neighbours)],
                    dtype=np.intp,
                ).reshape(len(nodes), width)
                self._tables[key] = (table[:, offset], None)
            else:
                dimensions = cast('LatticeSpace', space).dimensions
                lattice = LatticeSpace(dimensions, boundary=LATTICE_BOUNDARIES[kind], fill=value)
                table = lattice.index_table(Neighborhood((offset,)))[:, 0]  # type: ignore[arg-type]
                self._tables[key] = (table, value if kind == 'fixed' else None)
        table, ghost = self._tables[key]
        return table, ghost, MODES[kind]

    def shift(self, expression: Expression, offset: Offset | int) -> Expression:
        """Return an expression read at a neighbouring site of every member.

        Args:
            expression:
                The expression of member symbols, possibly already shifted.

            offset:
                The lattice offset, or the network neighbour slot.

        Returns:
            shifted:
                The expression in symbols that read the neighbour.
        """
        expression = SR(expression)
        replacements = {}
        for item in expression.variables():
            base, current = self.shifted.get(item, (item, None))
            if base not in self.references:
                continue
            place = self._field_of(base)
            if place is None:
                continue
            if isinstance(offset, int) or isinstance(current, int):
                if current is not None:
                    error_msg = f'A law of {self.owner} nests operators on a network, which only take one step.'
                    raise ValidationError(error_msg)
                total: Offset | int = offset
            else:
                total = tuple(a + b for a, b in zip(current or (0,) * len(offset), offset, strict=True))
            if not isinstance(total, int) and not any(total):
                replacements[item] = base
                continue
            name = f'{base}__{_tag(total)}'
            target = symbol(name, domain_of(base))
            if target not in self.references:
                reference = self.references[base]
                table, ghost, mode = self._table(base, place, total)
                lattice = isinstance(place.space, LatticeSpace)
                read = NeighborRead(
                    table,
                    ghost,
                    place.space.dimensions if lattice else None,  # type: ignore[union-attr]
                    total if lattice else None,  # type: ignore[arg-type]
                    mode,
                )
                self.references[target] = Reference(
                    reference.entry,
                    reference.is_state,
                    reference.column,
                    Source.NEIGHBOR,
                    read,
                )
                self.shifted[target] = (base, total)
            replacements[item] = target
        return expression.subs(replacements)

    def _space(self, expression: Expression) -> LatticeSpace | NetworkSpace:
        """Return the space of the fields an expression reads."""
        spaces = {
            id(place.space): place.space
            for item in SR(expression).variables()
            if (base := self.shifted.get(item, (item, None))[0]) in self.references
            and (place := self._field_of(base)) is not None
        }
        if len(spaces) != 1:
            error_msg = (
                f'A law of {self.owner} applies an operator to {expression}, which reads {len(spaces)} fields. An '
                'operator acts on the states of fields of one space.'
            )
            raise ValidationError(error_msg)
        return cast('LatticeSpace | NetworkSpace', next(iter(spaces.values())))

    def _laplacian(self, expression: Expression) -> Expression:
        """Return the stencil of the Laplacian of an expression."""
        space = self._space(expression)
        if isinstance(space, NetworkSpace):
            width = max((degree for _, degree in space.graph.degree), default=0)
            return sum((self.shift(expression, slot) - expression for slot in range(width)), SR(0))
        total = SR(0)
        for axis in range(space.n_dim):
            step = tuple(int(axis == other) for other in range(space.n_dim))
            back = tuple(-item for item in step)
            total += self.shift(expression, step) + self.shift(expression, back) - 2 * expression
        return total / space.spacing**2

    def _lattice(self, expression: Expression, axis: int, name: str) -> tuple[LatticeSpace, Offset, Offset]:
        """Return the lattice of an expression and the unit offsets along an axis."""
        space = self._space(expression)
        if not isinstance(space, LatticeSpace) or not 0 <= axis < space.n_dim:
            error_msg = (
                f'A law of {self.owner} takes the {name} of {expression} along axis {axis}, which needs a lattice.'
            )
            raise ValidationError(error_msg)
        step = tuple(int(axis == other) for other in range(space.n_dim))
        return space, step, tuple(-item for item in step)

    def lower(self, expression: object) -> object:
        """Return an expression with every operator replaced by its stencil, innermost first.

        Args:
            expression:
                The value of a law.

        Returns:
            lowered:
                The value with stencils.
        """
        if not has_operators(expression):
            return expression
        call = cast('Expression', expression)
        operator = call.operator()
        operands = [self.lower(item) for item in call.operands()]
        name = getattr(operator, 'name', lambda: '')()
        if name == 'laplacian':
            return self._laplacian(SR(operands[0]))
        if name == 'partial':
            space, step, back = self._lattice(SR(operands[0]), int(SR(operands[1])), 'derivative')
            return (self.shift(operands[0], step) - self.shift(operands[0], back)) / (2 * space.spacing)
        if name == 'upwind':
            value, velocity = SR(operands[0]), SR(operands[2])
            space, step, back = self._lattice(value, int(SR(operands[1])), 'advection')
            behind = (value - self.shift(value, back)) / space.spacing
            ahead = (self.shift(value, step) - value) / space.spacing
            return velocity * where(velocity > 0, behind, ahead)
        return operator(*operands)

    def note_limit(self, value: object) -> None:
        """Record the diffusive step limit of the Laplacians in a law value, when their coefficients are numbers.

        Args:
            value:
                The value of a rate law before lowering.
        """
        if not isinstance(value, Expression):
            return
        for call in _operator_calls(value, 'laplacian'):
            placeholder = symbol('_laplacian_')
            coefficient = SR(value.subs({call: placeholder})).diff(placeholder).subs(self.initial)
            if coefficient.variables():
                continue
            magnitude = abs(float(coefficient))
            if magnitude == 0:
                continue
            space = self._space(call.operands()[0])
            if isinstance(space, NetworkSpace):
                degree, spacing = max((item for _, item in space.graph.degree), default=1), 1.0
            else:
                degree, spacing = 2 * space.n_dim, float(space.spacing)
            self.limits.append(spacing**2 / (degree * magnitude))


def _operator_calls(expression: Expression, name: str) -> list[Expression]:
    """Return the calls of an operator by name inside an expression."""
    found, stack = [], [SR(expression)]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        if getattr(item.operator(), 'name', lambda: '')() == name:
            found.append(item)
        stack.extend(item.operands())
    return found


def has_operators(expression: object) -> bool:
    """Check whether an expression applies a differential operator.

    Args:
        expression:
            The expression.

    Returns:
        found:
            Whether it calls `laplacian`, `partial`, or `upwind`.
    """
    if not isinstance(expression, Expression):
        return False
    return bool(re.search(r'\b(laplacian|partial|upwind)\(', str(expression)))
