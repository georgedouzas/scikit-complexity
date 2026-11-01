"""Lower differential operators on fields to finite-difference stencils and graph Laplacians."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import re
from collections import defaultdict
from collections.abc import Callable, Mapping

import numpy as np
import numpy.typing as npt
import scipy.fft
from sage.all__sagemath_symbolics import SR, Expression
from scipy.ndimage import convolve, map_coordinates
from scipy.signal import oaconvolve
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import factorized
from skfem import Basis, ElementTetP1, ElementTriP1, asm
from skfem.models.poisson import laplace, mass

from ...core.elements import FieldElement
from ...core.laws import BaseLaw, BoundaryLaw, NoiseLaw, RateLaw, UpdateLaw
from ...core.laws.utils import INVERSION_BOUNDARIES, displacement, where
from ...core.spaces import AdjacencyNeighborhood, LatticeNeighborhood, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import ValidationError
from ...specs import NeighborReadSpec
from ...types import Indices, IntTuple
from .._utils import domain_of, symbol
from ..problems import DeclarationsSpec, DiffusiveLimitSpec, ExpressionPrinter, KernelSpec, ReferenceSpec
from ..problems._field_operator_kernel import FieldOperatorKernel
from ..problems._law_kernel import LATTICE_BOUNDARIES, MODES
from ..types import ReadKind
from ._base import BaseLowering, BasePass
from ._compilation import Compilation

OPERATOR_NAMES = ('laplacian', 'partial', 'upwind', 'inverse_helmholtz', 'convolve', 'shifted')
TIME = symbol('t')
PLANE = 2
SMALL_KERNEL = 7
WHOLE_SITE = 1e-9
SHIFTED_MODES = {'periodic': 'grid-wrap', 'zero-flux': 'reflect', 'fixed': 'grid-constant'}
FILTER_MODES = {'periodic': 'wrap', 'zero-flux': 'reflect', 'fixed': 'constant'}


class FieldLowering(BaseLowering[list[BaseLaw]]):
    """Lowering of the differential operators in the laws of one component to stencils, with the symbols they read.

    `prepare(compilation)` reads the fields and the initial values of the compilation, and `run(laws)` returns the laws
    with every operator replaced by its stencil, noting the stable steps of the Laplacians of rate laws.

    Args:
        spec:
            The kernel spec of the component, which `_apply_references` replaces by a copy with the neighbour reads
            and the derived values.

    Attributes:
        fields_:
            The field of every member entry by entry name.

        initial_:
            The initial value of every symbol that can be evaluated, for stability limits.

        shifted_:
            The symbol and offset every neighbour read reads.

        derived_:
            The field and the derived read of every spectral or finite element operator.

        limits_:
            The stable steps of the explicit Laplacians of finite-difference fields.

        volume_limits_:
            The stable steps of the Laplacians of finite-volume fields, as functions of the parameters.

        written_:
            The lowered value and the value as written of every law, by the entry name and column of its targets.

        tables_:
            The neighbour tables, by their axis, offset, boundary, and space.

        solution_boundaries_:
            The boundary condition given to an inversion, by the symbol of its solution.
    """

    fields_: Mapping[str, FieldElement]
    initial_: Mapping[Expression, float]
    shifted_: dict[Expression, tuple[Expression, IntTuple | int]]
    derived_: dict[Expression, tuple[FieldElement, FieldOperatorKernel]]
    limits_: list[float]
    volume_limits_: list[DiffusiveLimitSpec]
    written_: dict[tuple[str, int], list[tuple[Expression, Expression]]]
    tables_: dict[tuple[object, ...], tuple[Indices, float | None]]
    solution_boundaries_: dict[Expression, tuple[str, object]]

    def __init__(self, spec: KernelSpec) -> None:
        self.spec = spec

    def _prepare(self, context: Compilation | None) -> None:
        """Read the fields and the initial values of the compilation, with no stencil yet.

        Args:
            context:
                The compilation.

        Raises:
            ValidationError:
                The compilation has no layout yet.
        """
        super()._prepare(context)
        declarations = self.compilation_.layout
        if declarations is None:
            error_msg = 'The compilation has no layout yet. Lower differential operators after the layout pass.'
            raise ValidationError(error_msg)
        self.fields_ = self._field_entries(declarations)
        self.initial_ = self._initial_values(self.spec, declarations)
        self.shifted_, self.derived_, self.tables_, self.solution_boundaries_ = {}, {}, {}, {}
        self.limits_, self.volume_limits_ = [], []
        self.written_ = defaultdict(list)

    def _run(self, inputs: list[BaseLaw] | None) -> None:
        """Set laws whose differential operators are lowered, noting the stable steps of rate laws.

        Args:
            inputs:
                The laws of the component.
        """
        result: list[BaseLaw] = []
        for law in inputs or []:
            if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw) or not self._has_operators(law.value):
                result.append(law)
                continue
            if isinstance(law, RateLaw):
                self._note_limit(law.value)
            lowered = self._lower(law.value)
            if isinstance(law.value, Expression):
                for reference in self.spec._target_references(law._get_targets()[0]):
                    self.written_[(reference.entry.name, reference.column)].append((lowered, law.value))
            result.append(BasePass._replaced(law, value=lowered))
        self.lowered_ = result

    @staticmethod
    def _field_entries(declarations: DeclarationsSpec) -> dict[str, FieldElement]:
        """Return the fields by the name of each of their entries."""
        return {
            entry.name: block.node.component
            for block in declarations.blocks.values()
            if isinstance(block.node.component, FieldElement)
            for entry in block.entries.values()
        }

    @staticmethod
    def _initial_values(spec: KernelSpec, declarations: DeclarationsSpec) -> dict[Expression, float]:
        """Return the initial value of every symbol of a component, the first member's for groups."""
        states = np.concatenate([np.zeros(0), *declarations.initial_states])
        parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
        return {
            item: float((states if reference.is_state else parameters)[reference.entry.start + reference.column])
            for item, reference in spec.references.items()
            if reference.entry.start + reference.column < (states if reference.is_state else parameters).size
        }

    def _known_reference(self, item: Expression) -> ReferenceSpec:
        """Return the reference of a symbol of the spec, raising an error for a symbol the spec does not read.

        Args:
            item:
                The symbol.

        Returns:
            reference:
                The reference of the symbol.

        Raises:
            ValidationError:
                The spec does not read the symbol.
        """
        reference = self._reference_of(item, self.spec)
        if reference is None:
            error_msg = f'A law of {self.spec.owner} reads {item}, which is not a state or parameter of its namespace.'
            raise ValidationError(error_msg)
        return reference

    def _field_of(self, item: Expression) -> FieldElement | None:
        """Return the field whose member a symbol reads, or `None` for a single value."""
        reference = self._known_reference(item)
        if reference.source is not ReadKind.MEMBER:
            return None
        found = self.fields_.get(reference.entry.name)
        if found is None:
            error_msg = (
                f'A law of {self.spec.owner} applies a differential operator to {item}, which is not a state or '
                'parameter of a field. Create the group with `FieldElement`.'
            )
            raise ValidationError(error_msg)
        return found

    def _table(
        self,
        item: Expression,
        place: FieldElement,
        offset: IntTuple | int,
    ) -> tuple[Indices, float | None, int]:
        """Return the neighbour table, the fixed value, and the mode of a symbol at an offset."""
        reference = self._known_reference(item)
        variable = reference.entry.name.rpartition('.')[2]
        if reference.is_state:
            kind, value = place._get_boundary(variable)
        else:
            kind, value = (
                'periodic' if getattr(place.space, 'boundary', 'periodic') == 'periodic' else 'zero-flux',
                0.0,
            )
        key = (id(place), kind, value, offset)
        if key not in self.tables_:
            space = place.space
            if isinstance(space, NetworkSpace):
                table = space._index_table(AdjacencyNeighborhood())
                sites = np.arange(table.shape[0])[:, None]
                self.tables_[key] = (np.where(table == table.shape[0], sites, table)[:, offset], None)
            else:
                if not isinstance(space, LatticeSpace) or isinstance(offset, int) or not isinstance(value, float):
                    error_msg = (
                        f'A law of {self.spec.owner} reads {item} at a lattice offset, which needs a field on a '
                        'lattice with a number outside fixed edges.'
                    )
                    raise ValidationError(error_msg)
                dimensions = space.shape_
                lattice = LatticeSpace(dimensions, boundary=LATTICE_BOUNDARIES[kind], fill=value).check().prepare()
                table = lattice._index_table(LatticeNeighborhood((offset,)))[:, 0]
                self.tables_[key] = (table, value if kind == 'fixed' else None)
        table, ghost = self.tables_[key]
        return table, ghost, MODES[kind]

    def _shift(self, expression: Expression, offset: IntTuple | int) -> Expression:
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
            base, current = self.shifted_.get(item, (item, None))
            if base in self.derived_:
                replacements[item] = self._shifted_derived(base, self._total_offset(current, offset))
                continue
            if self._reference_of(base, self.spec) is None:
                continue
            place = self._field_of(base)
            if place is None:
                continue
            total = self._total_offset(current, offset)
            if not isinstance(total, int) and not any(total):
                replacements[item] = base
                continue
            name = f'{base}__{self._tag(total)}'
            target = symbol(name, domain_of(base))
            if self._reference_of(target, self.spec) is None:
                reference = self._known_reference(base)
                table, ghost, mode = self._table(base, place, total)
                space = place.space
                read = NeighborReadSpec(
                    table,
                    ghost,
                    space.shape_ if isinstance(space, LatticeSpace) else None,
                    total if isinstance(space, LatticeSpace) and not isinstance(total, int) else None,
                    mode,
                )
                self._add_reference(
                    self.spec,
                    target,
                    ReferenceSpec(reference.entry, reference.is_state, reference.column, ReadKind.NEIGHBOR, read),
                )
                self.shifted_[target] = (base, total)
            replacements[item] = target
        return expression.subs(replacements)

    def _total_offset(self, current: IntTuple | int | None, offset: IntTuple | int) -> IntTuple | int:
        """Return the offset of a read at an offset from a read that is already shifted."""
        if isinstance(offset, int) or isinstance(current, int):
            if current is not None:
                error_msg = f'A law of {self.spec.owner} nests operators on a network, which only take one step.'
                raise ValidationError(error_msg)
            return offset
        return tuple(a + b for a, b in zip(current or (0,) * len(offset), offset, strict=True))

    def _boundaries(self, expression: Expression) -> list[tuple[str, object]]:
        """Return the boundary conditions of the field states an expression reads, through derived values."""
        found = []
        for item in SR(expression).variables():
            if item in self.solution_boundaries_:
                found.append(self.solution_boundaries_[item])
                continue
            if item in self.derived_:
                found += self._boundaries(self.derived_[item][1].operand)
                continue
            base = self.shifted_.get(item, (item, None))[0]
            reference = self._reference_of(base, self.spec)
            if reference is None or not reference.is_state or reference.source is not ReadKind.MEMBER:
                continue
            place = self._field_of(base)
            if place is not None:
                found.append(place._get_boundary(reference.entry.name.rpartition('.')[2]))
        return found

    def _boundary(self, expression: Expression, place: FieldElement) -> tuple[str, object]:
        """Return the one boundary condition of the field states an expression reads, or the edges of the space."""
        found = set(self._boundaries(expression))
        if len(found) > 1:
            error_msg = (
                f'A law of {self.spec.owner} applies a nonlocal operator to {expression}, whose states have different '
                'boundary conditions. Give the states of the operand one boundary condition, or give an inversion '
                'the boundary of its solution.'
            )
            raise ValidationError(error_msg)
        return found.pop() if found else place._get_boundary('')

    def _neighbor_table(self, place: FieldElement, kind: str, value: object, offset: IntTuple | int) -> Indices:
        """Return the neighbour of every site at an offset, with the number of sites for reads beyond fixed edges."""
        space = place.space
        if isinstance(space, NetworkSpace) and isinstance(offset, int):
            table = space._index_table(AdjacencyNeighborhood())
            sites = np.arange(table.shape[0])[:, None]
            return np.asarray(np.where(table == table.shape[0], sites, table)[:, offset], dtype=np.intp)
        if not isinstance(space, LatticeSpace) or isinstance(offset, int) or not isinstance(value, float):
            error_msg = (
                f'A law of {self.spec.owner} reads a nonlocal value at a neighbour, which needs a field on a lattice '
                'or a network with a number outside fixed edges.'
            )
            raise ValidationError(error_msg)
        lattice = LatticeSpace(space.shape_, boundary=LATTICE_BOUNDARIES[kind], fill=value).check().prepare()
        return np.asarray(lattice._index_table(LatticeNeighborhood((offset,)))[:, 0], dtype=np.intp)

    def _shifted_derived(self, base: Expression, total: IntTuple | int) -> Expression:
        """Return the symbol of a derived value read at a neighbouring site, indexing its values by the neighbour table.

        Args:
            base:
                The symbol of the derived value.

            total:
                The lattice offset, or the network neighbour slot.

        Returns:
            shifted:
                The symbol of the derived value at the neighbour.
        """
        if not isinstance(total, int) and not any(total):
            return base
        target = symbol(f'{base}__{self._tag(total)}')
        if target in self.derived_:
            return target
        place = self.derived_[base][0]
        kind, value = self._boundary(base, place)
        table = self._neighbor_table(place, kind, value, total)
        size = table.shape[0]
        inside = table < size
        matrix = csr_matrix((np.ones(inside.sum()), (np.flatnonzero(inside), table[inside])), shape=(size, size))
        ghost = float(value) if kind == 'fixed' and isinstance(value, float) else 0.0

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.append(values, ghost)[table]

        read = FieldOperatorKernel(operand=base, symbols=[base], apply=apply, matrix=matrix).check().prepare()
        self.shifted_[target] = (base, total)
        return self._derived(place, read, target)

    def _places(self, expression: Expression) -> dict[int, FieldElement]:
        """Return the fields an expression reads, through member symbols, neighbour reads, and derived values."""
        places: dict[int, FieldElement] = {}
        place: FieldElement | None
        for item in SR(expression).variables():
            if item in self.derived_:
                place = self.derived_[item][0]
            else:
                base = self.shifted_.get(item, (item, None))[0]
                place = self._field_of(base) if self._reference_of(base, self.spec) is not None else None
            if place is not None:
                places[id(place)] = place
        return places

    def _place(self, expression: Expression) -> FieldElement:
        """Return the one field an expression reads, or raise an error."""
        places = self._places(expression)
        discretizations = {place.discretization for place in places.values()}
        spaces = {id(place.space) for place in places.values()}
        if len(spaces) != 1 or len(discretizations) != 1:
            error_msg = (
                f'A law of {self.spec.owner} applies an operator to {expression}, which reads {len(spaces)} fields. An '
                'operator acts on the states of fields of one space and one discretization.'
            )
            raise ValidationError(error_msg)
        return next(iter(places.values()))

    def _space(self, expression: Expression) -> LatticeSpace | NetworkSpace:
        """Return the space of the fields an expression reads."""
        spaces = {id(place.space): place.space for place in self._places(expression).values()}
        if len(spaces) != 1:
            error_msg = (
                f'A law of {self.spec.owner} applies an operator to {expression}, which reads {len(spaces)} fields. An '
                'operator acts on the states of fields of one space.'
            )
            raise ValidationError(error_msg)
        space = next(iter(spaces.values()))
        if not isinstance(space, LatticeSpace | NetworkSpace):
            error_msg = (
                f'A law of {self.spec.owner} applies a stencil to {expression}, which lives on a mesh. Use the '
                'finite-element discretization.'
            )
            raise ValidationError(error_msg)
        return space

    def _laplacian(self, expression: Expression) -> Expression:
        """Return the stencil of the Laplacian of an expression."""
        space = self._space(expression)
        if isinstance(space, NetworkSpace):
            width = space._index_table(AdjacencyNeighborhood()).shape[1]
            return sum((self._shift(expression, slot) - expression for slot in range(width)), SR(0))
        total = SR(0)
        for axis in range(space.n_dim_):
            step = tuple(int(axis == other) for other in range(space.n_dim_))
            back = tuple(-item for item in step)
            total += (self._shift(expression, step) + self._shift(expression, back) - 2 * expression) / (
                space.spacings_[axis] ** 2
            )
        return total

    def _lattice(self, expression: Expression, axis: int, name: str) -> tuple[LatticeSpace, IntTuple, IntTuple]:
        """Return the lattice of an expression and the unit offsets along an axis."""
        space = self._space(expression)
        if not isinstance(space, LatticeSpace) or not 0 <= axis < space.n_dim_:
            error_msg = (
                f'A law of {self.spec.owner} takes the {name} of {expression} along axis {axis}, which needs a lattice.'
            )
            raise ValidationError(error_msg)
        step = tuple(int(axis == other) for other in range(space.n_dim_))
        return space, step, tuple(-item for item in step)

    def _lower(self, expression: object) -> object:
        """Return an expression with every operator replaced by its stencil, innermost first.

        Args:
            expression:
                The value of a law.

        Returns:
            lowered:
                The value with stencils.
        """
        if not isinstance(expression, Expression) or not self._has_operators(expression):
            return expression
        call = expression
        operator = call.operator()
        operands = [self._lower(item) for item in call.operands()]
        name = getattr(operator, 'name', lambda: '')()
        if name not in OPERATOR_NAMES:
            return operator(*operands)
        if name == 'shifted':
            return self._shifted(SR(operands[0]), [SR(item) for item in operands[1:]])
        place = self._place(SR(operands[0]))
        if name == 'convolve':
            return self._convolve(SR(operands[0]), SR(operands[1]), SR(operands[2]), place)
        return self._discretized(name, operands, place)

    def _discretized(self, name: str, operands: list[object], place: FieldElement) -> Expression:
        """Return a local operator or an inversion lowered for the discretization of its field."""
        if place.discretization == 'spectral':
            return self._spectral(name, operands, place)
        if name == 'inverse_helmholtz':
            given = self._given_boundary(operands[2:])
            return self._inverse(SR(operands[0]), self._screening(SR(operands[1])), place, given)
        if place.discretization == 'finite-element':
            return self._finite_element(name, SR(operands[0]), place)
        return self._stencil(name, operands)

    def _stencil(self, name: str, operands: list[object]) -> Expression:
        """Return the finite-difference stencil of an operator on a lattice or a network."""
        if name == 'laplacian':
            return self._laplacian(SR(operands[0]))
        axis = int(SR(operands[1]))
        value = SR(operands[0])
        space, step, back = self._lattice(value, axis, 'derivative' if name == 'partial' else 'advection')
        if name == 'partial':
            return (self._shift(value, step) - self._shift(value, back)) / (2 * space.spacings_[axis])
        velocity = SR(operands[2])
        behind = (value - self._shift(value, back)) / space.spacings_[axis]
        ahead = (self._shift(value, step) - value) / space.spacings_[axis]
        return velocity * where(velocity > 0, behind, ahead)

    def _derived(self, place: FieldElement, read: FieldOperatorKernel, target: Expression | None = None) -> Expression:
        """Return a symbol that stands for a derived value of a field, a new one by default, with its read."""
        target = symbol(f'_derived_{place.label}_{len(self.derived_)}') if target is None else target
        references = [self._known_reference(item) for item in read.symbols]
        entry = next(
            reference.entry
            for reference in references
            if reference.source in {ReadKind.MEMBER, ReadKind.NEIGHBOR, ReadKind.DERIVED}
        )
        self._add_reference(self.spec, target, ReferenceSpec(entry, True, 0, ReadKind.DERIVED, read))
        self.derived_[target] = (place, read)
        return target

    def _spectral(self, name: str, operands: list[object], place: FieldElement) -> Expression:
        """Return the symbol of a spectral operator applied to an expression, composing chains of operators."""
        space = place.space
        if not isinstance(space, LatticeSpace) or space.boundary != 'periodic':
            error_msg = f'A law of {self.spec.owner} uses a spectral field, which needs a periodic lattice.'
            raise ValidationError(error_msg)
        if name == 'upwind':
            velocity = SR(operands[2])
            return velocity * self._spectral('partial', [operands[0], operands[1]], place)
        if name == 'inverse_helmholtz' and self._given_boundary(operands[2:]) not in {None, ('periodic', 0.0)}:
            error_msg = (
                f'A law of {self.spec.owner} inverts on a spectral field with a boundary that is not periodic. Give '
                'the inversion no boundary, or use the finite-difference discretization.'
            )
            raise ValidationError(error_msg)
        axis = int(SR(operands[1])) if name == 'partial' else 0
        if name == 'partial' and not 0 <= axis < space.n_dim_:
            error_msg = f'A law of {self.spec.owner} takes a derivative along axis {axis} of a {space.n_dim_}D lattice.'
            raise ValidationError(error_msg)
        screening = self._screening(SR(operands[1])) if name == 'inverse_helmholtz' else 0.0
        multiplier = self._spectral_multiplier(name, axis, space.shape_, space.spacings_, screening)
        return self._multiplied(SR(operands[0]), multiplier, place)

    def _multiplied(
        self,
        operand: Expression,
        multiplier: npt.NDArray[np.complex128],
        place: FieldElement,
    ) -> Expression:
        """Return the symbol of a Fourier multiplier applied to an expression, composing chains of multipliers."""
        dimensions = place.space.shape_ if isinstance(place.space, LatticeSpace) else ()
        inner = self.derived_.get(operand)
        if inner is not None and inner[1].multiplier is not None:
            operand, multiplier = inner[1].operand, multiplier * inner[1].multiplier
        symbols = sorted(
            (item for item in operand.variables() if self._reference_of(item, self.spec) is not None),
            key=str,
        )
        read = (
            FieldOperatorKernel(
                operand=operand,
                symbols=symbols,
                apply=self._spectral_apply(multiplier, dimensions),
                multiplier=multiplier,
                dimensions=dimensions,
            )
            .check()
            .prepare()
        )
        return self._derived(place, read)

    def _symbols(self, expression: Expression) -> list[Expression]:
        """Return the symbols of an expression that the spec reads, sorted by name."""
        return sorted(
            (item for item in SR(expression).variables() if self._reference_of(item, self.spec) is not None),
            key=str,
        )

    def _screening(self, screening: Expression) -> float:
        """Return the value of a screening at the current parameters, raising an error when it is not a real number."""
        value = SR(screening).subs(self.initial_)
        if value.variables() or not value.is_real() or float(value) < 0:
            error_msg = (
                f'A law of {self.spec.owner} inverts with the screening {screening}, which is not a non-negative '
                'number of the parameters. Use a screening whose square is non-negative and that reads no state.'
            )
            raise ValidationError(error_msg)
        return float(value)

    def _given_boundary(self, operands: list[object]) -> tuple[str, float] | None:
        """Return the boundary condition given to an inversion by its code and value, or `None` when none is given."""
        if not operands:
            return None
        code, value = SR(operands[0]), SR(operands[1])
        if code.variables() or value.variables() or not 0 <= int(code) < len(INVERSION_BOUNDARIES):
            error_msg = (
                f'A law of {self.spec.owner} inverts with the boundary code {code} and value {value}. Give '
                '`inverse_helmholtz` its boundary as `boundary=`, with a number as the fixed value.'
            )
            raise ValidationError(error_msg)
        return INVERSION_BOUNDARIES[int(code)], float(value)

    def _inverse(
        self,
        operand: Expression,
        screening: float,
        place: FieldElement,
        given: tuple[str, float] | None = None,
    ) -> Expression:
        """Return the symbol of the solution of a modified-Helmholtz equation on a lattice, a network, or a mesh.

        Args:
            operand:
                The right-hand side.

            screening:
                The screening, a non-negative number.

            place:
                The field.

            given:
                The boundary condition of the solution, or `None` for the boundary condition of the operand's states.

        Returns:
            solution:
                The symbol of the solution.

        Raises:
            ValidationError:
                The field's space or the boundary has no inversion, such as a boundary given on a network.
        """
        space = place.space
        kind, value = self._boundary(operand, place) if given is None else given
        apply: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        if isinstance(space, NetworkSpace) and given is not None:
            error_msg = f'A law of {self.spec.owner} inverts on a network with a boundary. Give the inversion none.'
            raise ValidationError(error_msg)
        if isinstance(space, MeshSpace):
            apply = self._mesh_inverse(space, screening, kind, value)
        elif isinstance(space, NetworkSpace):
            apply = self._network_inverse(space, screening)
        elif isinstance(space, LatticeSpace) and (kind != 'fixed' or isinstance(value, float)):
            apply = self._lattice_inverse(space, screening, kind, float(value) if isinstance(value, float) else 0.0)
        else:
            error_msg = (
                f'A law of {self.spec.owner} inverts {operand} on a {type(space).__name__} with the boundary {kind}, '
                'which needs a lattice, a network, or a mesh, with a number outside fixed lattice edges.'
            )
            raise ValidationError(error_msg)
        read = FieldOperatorKernel(operand=operand, symbols=self._symbols(operand), apply=apply).check().prepare()
        solution = self._derived(place, read)
        if given is not None:
            self.solution_boundaries_[solution] = (kind, value)
        return solution

    def _kernel_weights(
        self,
        kernel: Expression,
        space: LatticeSpace,
        radius: Expression,
        *,
        wrapped: bool,
    ) -> npt.NDArray[np.float64]:
        """Return a kernel sampled at the displacements of a lattice within a radius, times the cell volume.

        Args:
            kernel:
                The kernel, an expression of the displacement and the parameters.

            space:
                The lattice.

            radius:
                The distance beyond which the kernel is zero, or infinity.

            wrapped:
                Whether to sample every displacement of a periodic lattice in FFT order, rather than a centred window.

        Returns:
            weights:
                The sampled kernel.

        Raises:
            ValidationError:
                The kernel reads a symbol that is not a displacement or a parameter.
        """
        axes = [displacement(axis) for axis in range(space.n_dim_)]
        parameters = {
            key: item
            for key, item in self.initial_.items()
            if not getattr(self._reference_of(key, self.spec), 'is_state', True)
        }
        value = SR(kernel).subs(parameters)
        unknown = sorted(set(value.variables()) - set(axes), key=str)
        if unknown:
            error_msg = (
                f'A law of {self.spec.owner} convolves with the kernel {kernel}, which reads '
                f'{", ".join(map(str, unknown))}. Write the kernel in the displacements and the parameters.'
            )
            raise ValidationError(error_msg)
        distance = float('inf') if SR(radius).is_infinity() else float(SR(radius).subs(parameters))
        if wrapped:
            steps = [np.fft.fftfreq(size, d=1.0 / size) for size in space.shape_]
        else:
            reach = [
                size - 1 if np.isinf(distance) else min(size - 1, int(np.ceil(distance / step - WHOLE_SITE)))
                for size, step in zip(space.shape_, space.spacings_, strict=True)
            ]
            steps = [np.arange(-item, item + 1, dtype=float) for item in reach]
        grids = np.meshgrid(*[item * step for item, step in zip(steps, space.spacings_, strict=True)], indexing='ij')
        printer = ExpressionPrinter(time=TIME, compiled=True).check().prepare(axes)
        function = ExpressionPrinter._function_of(printer.run([value]).code_)
        computed = ExpressionPrinter._values_of(function(0.0, *grids))[0]
        sampled = np.broadcast_to(np.asarray(computed, dtype=float), grids[0].shape)
        inside = np.sqrt(sum((grid**2 for grid in grids), np.zeros(grids[0].shape))) <= distance * (1 + WHOLE_SITE)
        return np.where(inside, sampled, 0.0) * float(np.prod(space.spacings_))

    def _convolve(self, operand: Expression, kernel: Expression, radius: Expression, place: FieldElement) -> Expression:
        """Return the symbol of the convolution of an expression with a kernel on a lattice."""
        space = place.space
        if not isinstance(space, LatticeSpace):
            error_msg = (
                f'A law of {self.spec.owner} convolves {operand} on a {type(space).__name__}, which needs a lattice '
                'space. Give the field a lattice space.'
            )
            raise ValidationError(error_msg)
        if place.discretization == 'spectral':
            weights = self._kernel_weights(kernel, space, radius, wrapped=True)
            return self._multiplied(operand, np.fft.rfftn(weights).astype(complex), place)
        kind, value = self._boundary(operand, place)
        if kind == 'fixed' and not isinstance(value, float):
            error_msg = f'A law of {self.spec.owner} convolves {operand}, which needs a number outside fixed edges.'
            raise ValidationError(error_msg)
        window = self._kernel_weights(kernel, space, radius, wrapped=False)
        if kind == 'periodic' and max(window.shape) > SMALL_KERNEL:
            weights = self._kernel_weights(kernel, space, radius, wrapped=True)
            apply = self._spectral_apply(np.fft.rfftn(weights).astype(complex), space.shape_)
        else:
            apply = self._window_apply(window, space.shape_, kind, float(value) if isinstance(value, float) else 0.0)
        matrix = None if SR(radius).is_infinity() else self._window_matrix(window, place, kind, value)
        read = FieldOperatorKernel(operand=operand, symbols=self._symbols(operand), apply=apply, matrix=matrix)
        return self._derived(place, read.check().prepare())

    def _window_matrix(
        self,
        window: npt.NDArray[np.float64],
        place: FieldElement,
        kind: str,
        value: object,
    ) -> csr_matrix:
        """Return the sparse matrix of the linear part of a convolution with a window, the sites within its radius."""
        size = place.size_
        total = csr_matrix((size, size))
        centre = np.asarray(window.shape) // 2
        for index in zip(*np.nonzero(window), strict=True):
            offset = tuple(int(item) for item in centre - np.asarray(index))
            table = self._neighbor_table(place, kind, value, offset)
            inside = table < size
            selection = (np.full(inside.sum(), window[index]), (np.flatnonzero(inside), table[inside]))
            total = total + csr_matrix(selection, shape=(size, size))
        return csr_matrix(total)

    def _shifted(self, operand: Expression, operands: list[Expression]) -> Expression:
        """Return an expression read at a physical displacement, a neighbour read for whole sites, else interpolated.

        Args:
            operand:
                The expression.

            operands:
                The displacement along every axis, followed by the order of the interpolation when it is not 1.

        Returns:
            read:
                The expression read at the displacement.

        Raises:
            ValidationError:
                The field is not on a lattice, the displacement does not have one component per axis, or a fixed edge
                has no number.
        """
        place = self._place(operand)
        space = place.space
        dimensions = space.n_dim_ if isinstance(space, LatticeSpace) else 0
        offset, orders = operands[:dimensions], operands[dimensions:]
        if not isinstance(space, LatticeSpace) or len(offset) != space.n_dim_ or len(orders) > 1:
            error_msg = (
                f'A law of {self.spec.owner} reads {operand} at a shifted position, which needs a field on a lattice '
                'space and one displacement per axis. Give the field a lattice space.'
            )
            raise ValidationError(error_msg)
        sites = self._whole_sites(offset, space.spacings_)
        if sites is not None:
            return self._shift(operand, sites)
        kind, value = self._boundary(operand, place)
        if kind == 'fixed' and not isinstance(value, float):
            error_msg = f'A law of {self.spec.owner} reads {operand} beyond fixed edges, which needs a number there.'
            raise ValidationError(error_msg)
        symbols = sorted(set(self._symbols(operand)).union(*[self._symbols(item) for item in offset]), key=str)
        apply = self._interpolated_apply(
            space.shape_,
            space.spacings_,
            kind,
            float(value) if isinstance(value, float) else 0.0,
            int(orders[0]) if orders else 1,
        )
        read = FieldOperatorKernel(
            operand=operand,
            symbols=symbols,
            apply=apply,
            displacement=offset,
        )
        return self._derived(place, read.check().prepare())

    def _finite_element(self, name: str, operand: Expression, place: FieldElement) -> Expression:
        """Return the symbol of the finite element Laplacian of an expression."""
        if name != 'laplacian':
            error_msg = (
                f'A law of {self.spec.owner} takes the {name} of a finite element field, which takes Laplacians only.'
            )
            raise ValidationError(error_msg)
        if not isinstance(place.space, MeshSpace):
            error_msg = f'A law of {self.spec.owner} uses a finite element field, which needs a mesh.'
            raise ValidationError(error_msg)
        matrix = self._mesh_laplacian(place.space)
        symbols = sorted(
            (item for item in operand.variables() if self._reference_of(item, self.spec) is not None),
            key=str,
        )
        read = (
            FieldOperatorKernel(
                operand=operand,
                symbols=symbols,
                apply=lambda values: matrix @ values,
                matrix=matrix,
            )
            .check()
            .prepare()
        )
        return self._derived(place, read)

    def _note_volume_limit(self, coefficient: Expression, stiffness: float) -> None:
        """Record the diffusive step limit of a finite-volume field as a function of the parameters it reads."""
        symbols = sorted(set(coefficient.variables()) - {TIME}, key=str)
        found = [self._reference_of(item, self.spec) for item in symbols]
        references = [reference for reference in found if reference is not None]
        if len(references) < len(symbols) or any(reference.is_state for reference in references):
            return
        function = ExpressionPrinter(time=TIME, compiled=True).check().prepare(symbols).run([coefficient]).code_
        if not callable(function):
            error_msg = f'A compiled printer returns a function, not a {type(function).__name__}.'
            raise TypeError(error_msg)
        self.volume_limits_.append(DiffusiveLimitSpec(function, tuple(references), stiffness))

    def _note_limit(self, value: object) -> None:
        """Record the diffusive step limit of the Laplacians in a law value.

        A finite-difference field records the limit when the coefficient is a number at the initial values, and a
        finite-volume field records it as a function of the parameters the coefficient reads.

        Args:
            value:
                The value of a rate law before lowering.
        """
        if not isinstance(value, Expression):
            return
        for call in self._operator_calls(value, 'laplacian'):
            placeholder = symbol('_laplacian_')
            expression = SR(value.subs({call: placeholder})).diff(placeholder)
            place = self._place(call.operands()[0])
            if place.discretization not in {'finite-difference', 'finite-volume'}:
                continue
            space = place.space
            if isinstance(space, NetworkSpace):
                stiffness = max((item for _, item in space.graph.degree), default=1)
            elif isinstance(space, LatticeSpace):
                stiffness = sum(2 / step**2 for step in space.spacings_)
            else:
                continue
            if stiffness == 0:
                continue
            if place.discretization == 'finite-volume':
                self._note_volume_limit(expression, stiffness)
                continue
            coefficient = expression.subs(self.initial_)
            if coefficient.variables():
                continue
            magnitude = abs(float(coefficient))
            if magnitude == 0:
                continue
            self.limits_.append(1 / (stiffness * magnitude))

    @staticmethod
    def _tag(offset: IntTuple | int) -> str:
        """Return a name fragment for a lattice offset or a network neighbour slot."""
        if isinstance(offset, int):
            return f'k{offset}'
        return '_'.join(f'{"m" if step < 0 else "p"}{abs(step)}' for step in offset)

    @staticmethod
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

    @staticmethod
    def _spectral_apply(
        multiplier: npt.NDArray[np.complex128],
        dimensions: tuple[int, ...],
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that applies a Fourier multiplier to the values of a periodic lattice.

        Args:
            multiplier:
                The factor of every real Fourier mode.

            dimensions:
                The lattice shape.

        Returns:
            apply:
                The function of the values at every site.
        """

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            transformed = np.fft.rfftn(np.reshape(values, dimensions)) * multiplier
            return np.fft.irfftn(transformed, s=dimensions, axes=range(len(dimensions))).ravel()

        return apply

    @staticmethod
    def _whole_sites(offset: list[Expression], spacings: tuple[float, ...]) -> IntTuple | None:
        """Return a displacement in whole sites when every component is a number of whole spacings, else `None`."""
        sites = []
        for item, step in zip(offset, spacings, strict=True):
            if SR(item).variables() or not SR(item).is_real():
                return None
            ratio = float(SR(item)) / step
            if abs(ratio - round(ratio)) > WHOLE_SITE:
                return None
            sites.append(round(ratio))
        return tuple(sites)

    @staticmethod
    def _window_apply(
        window: npt.NDArray[np.float64],
        dimensions: tuple[int, ...],
        kind: str,
        fill: float,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that convolves the values of a lattice with a centred window of weights.

        Windows of at most `SMALL_KERNEL` sites per axis use a direct filter, and larger ones an overlap-add FFT of the
        values padded beyond the edges with the fixed value or their mirror image.

        Args:
            window:
                The weights at the displacements from `-reach` to `reach` sites along every axis.

            dimensions:
                The lattice shape.

            kind:
                The boundary, `'periodic'`, `'zero-flux'`, or `'fixed'`.

            fill:
                The value beyond fixed edges.

        Returns:
            apply:
                The function of the values at every site.
        """
        reach = [(size // 2, size // 2) for size in window.shape]

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            grid = np.reshape(values, dimensions)
            if max(window.shape) <= SMALL_KERNEL:
                return np.asarray(convolve(grid, window, mode=FILTER_MODES[kind], cval=fill)).ravel()
            if kind == 'fixed':
                padded = np.pad(grid, reach, mode='constant', constant_values=fill)
            else:
                padded = np.pad(grid, reach, mode='symmetric')
            return np.asarray(oaconvolve(padded, window, mode='valid')).ravel()

        return apply

    @staticmethod
    def _interpolated_apply(
        dimensions: tuple[int, ...],
        spacings: tuple[float, ...],
        kind: str,
        fill: float,
        order: int = 1,
    ) -> Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that reads the values of a lattice at displaced positions by spline interpolation.

        Args:
            dimensions:
                The lattice shape.

            spacings:
                The distance between sites along every axis.

            kind:
                The boundary, `'periodic'`, `'zero-flux'`, or `'fixed'`.

            fill:
                The value beyond fixed edges.

            order:
                The order of the spline, 1 for linear or 3 for cubic.

        Returns:
            apply:
                The function of the values and the displacement along every axis at every site.
        """
        sites = np.indices(dimensions).reshape(len(dimensions), -1).astype(float)
        steps = np.asarray(spacings, dtype=float)[:, None]

        def apply(values: npt.NDArray[np.float64], offset: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            grid = np.reshape(values, dimensions)
            positions = sites + np.reshape(offset, sites.shape) / steps
            return np.asarray(map_coordinates(grid, positions, order=order, mode=SHIFTED_MODES[kind], cval=fill))

        return apply

    @staticmethod
    def _lattice_inverse(
        space: LatticeSpace,
        screening: float,
        kind: str,
        fill: float,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that inverts the lattice stencil of `laplacian - screening**2` by a fast transform.

        Periodic edges use real FFTs, fixed edges the sine transform of type 1 with the fixed value lifted into the
        right-hand side, and zero-flux edges the cosine transform of type 2, each with the eigenvalues of the stencil.
        The mode of zero eigenvalue, the mean, is set to zero.

        Args:
            space:
                The lattice.

            screening:
                The screening, a non-negative number.

            kind:
                The boundary, `'periodic'`, `'zero-flux'`, or `'fixed'`.

            fill:
                The value beyond fixed edges.

        Returns:
            apply:
                The function of the right-hand side at every site.
        """
        dimensions, spacings = space.shape_, space.spacings_
        if kind == 'periodic':
            waves = FieldLowering._wavenumbers(dimensions, spacings)
            eigenvalues = sum(
                ((2 * np.cos(wave * step) - 2) / step**2 for wave, step in zip(waves, spacings, strict=True)),
                np.zeros(waves[0].shape),
            )
            denominator = eigenvalues - screening**2
            nonzero = np.abs(denominator) > 0
            multiplier = np.where(nonzero, 1.0 / np.where(nonzero, denominator, 1.0), 0.0).astype(complex)
            return FieldLowering._spectral_apply(multiplier, dimensions)
        if kind == 'fixed':
            modes = [np.arange(1, size + 1) / (size + 1) for size in dimensions]
        else:
            modes = [np.arange(size) / size for size in dimensions]
        axes = [(2 * np.cos(np.pi * mode) - 2) / step**2 for mode, step in zip(modes, spacings, strict=True)]
        grids = np.meshgrid(*axes, indexing='ij')
        denominator = sum(grids, np.zeros(dimensions)) - screening**2
        nonzero = np.abs(denominator) > 0
        inverse = np.where(nonzero, 1.0 / np.where(nonzero, denominator, 1.0), 0.0)
        lift = np.zeros(dimensions)
        for axis, step in enumerate(spacings):
            edges = [slice(None)] * len(dimensions)
            for index in (0, -1):
                edges[axis] = slice(index, index + 1 if index == 0 else None)
                lift[tuple(edges)] += fill / step**2
        kind_number = 1 if kind == 'fixed' else 2
        transform = scipy.fft.dstn if kind == 'fixed' else scipy.fft.dctn
        backward = scipy.fft.idstn if kind == 'fixed' else scipy.fft.idctn

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            right = np.reshape(values, dimensions) - (lift if kind == 'fixed' else 0.0)
            spectrum = transform(right, type=kind_number, norm='ortho') * inverse
            return np.asarray(backward(spectrum, type=kind_number, norm='ortho')).ravel()

        return apply

    @staticmethod
    def _grounded_solver(
        matrix: csr_matrix,
        free: npt.NDArray[np.intp],
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that solves a sparse system on its free rows, factorised once, zero on the others."""
        solve = factorized(matrix[free][:, free].tocsc())
        size = matrix.shape[0]

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            solution = np.zeros(size)
            solution[free] = solve(np.asarray(values, dtype=float)[free])
            return solution

        return apply

    @staticmethod
    def _network_inverse(
        space: NetworkSpace,
        screening: float,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that inverts the graph Laplacian minus `screening**2`, with zero mean when unscreened.

        Args:
            space:
                The network.

            screening:
                The screening, a non-negative number.

        Returns:
            apply:
                The function of the right-hand side at every node.
        """
        table = space._index_table(AdjacencyNeighborhood())
        size = table.shape[0]
        rows = np.repeat(np.arange(size), table.shape[1])
        columns = table.ravel()
        inside = columns < size
        adjacency = csr_matrix((np.ones(inside.sum()), (rows[inside], columns[inside])), shape=(size, size))
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        matrix = csr_matrix(adjacency - diags(degrees) - screening**2 * identity(size))
        if screening > 0:
            return FieldLowering._grounded_solver(matrix, np.arange(size))
        solve = FieldLowering._grounded_solver(matrix, np.arange(1, size))

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            right = np.asarray(values, dtype=float)
            solution = solve(right - right.mean())
            return solution - solution.mean()

        return apply

    @staticmethod
    def _mesh_inverse(
        space: MeshSpace,
        screening: float,
        kind: str,
        fill: object,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that inverts the lumped finite element Laplacian minus `screening**2` on a mesh.

        It solves `(K + screening**2 M) psi = -M f` with the stiffness matrix `K` and the lumped mass `M`, with the
        fixed value on the boundary nodes of fixed edges and zero mean for unscreened zero-flux edges.

        Args:
            space:
                The mesh.

            screening:
                The screening, a non-negative number.

            kind:
                The boundary, `'zero-flux'` or `'fixed'`.

            fill:
                The value on the boundary nodes of fixed edges, a number or a function of the node coordinates.

        Returns:
            apply:
                The function of the right-hand side at every node.
        """
        stiffness, lumped = FieldLowering._mesh_matrices(space)
        size = lumped.size
        matrix = csr_matrix(stiffness + screening**2 * diags(lumped))
        if kind == 'fixed':
            nodes = space.boundary_nodes_
            coordinates = space.positions_[nodes]
            fixed = np.zeros(size)
            fixed[nodes] = np.broadcast_to(fill(*coordinates.T) if callable(fill) else float(str(fill)), nodes.shape)
            free = np.setdiff1d(np.arange(size), nodes)
            solve = FieldLowering._grounded_solver(matrix, free)
            carried = matrix @ fixed

            def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return solve(-lumped * np.asarray(values, dtype=float) - carried) + fixed

            return apply
        if screening > 0:
            solve = FieldLowering._grounded_solver(matrix, np.arange(size))
            return lambda values: solve(-lumped * np.asarray(values, dtype=float))
        solve = FieldLowering._grounded_solver(matrix, np.arange(1, size))

        def apply_mean(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            right = np.asarray(values, dtype=float)
            solution = solve(-lumped * (right - np.average(right, weights=lumped)))
            return solution - np.average(solution, weights=lumped)

        return apply_mean

    @staticmethod
    def _has_operators(expression: object) -> bool:
        """Check whether an expression applies a differential operator.

        Args:
            expression:
                The expression.

        Returns:
            found:
                Whether it calls `laplacian`, `partial`, `upwind`, `inverse_helmholtz`, `convolve`, or `shifted`.
        """
        if not isinstance(expression, Expression):
            return False
        return bool(re.search(r'\b(laplacian|partial|upwind|inverse_helmholtz|convolve|shifted)\(', str(expression)))

    @staticmethod
    @functools.cache
    def _cached_mesh_matrices(
        points: bytes,
        cells: bytes,
        shape: tuple[int, int],
        cell_shape: tuple[int, int],
    ) -> tuple[csr_matrix, npt.NDArray[np.float64]]:
        """Return the stiffness matrix and the lumped mass of a mesh, cached by its nodes and cells."""
        nodes = np.frombuffer(points).reshape(shape)
        elements = np.frombuffer(cells, dtype=np.int64).reshape(cell_shape)
        space = MeshSpace(nodes, elements)
        basis = Basis(space.check().prepare().mesh_, ElementTriP1() if shape[0] == PLANE else ElementTetP1())
        stiffness = asm(laplace, basis).tocsr()
        lumped = np.asarray(asm(mass, basis).sum(axis=1)).ravel()
        return stiffness, lumped

    @staticmethod
    def _mesh_matrices(space: MeshSpace) -> tuple[csr_matrix, npt.NDArray[np.float64]]:
        """Return the stiffness matrix and the lumped mass of the linear elements of a mesh.

        Args:
            space:
                The mesh.

        Returns:
            matrices:
                The stiffness matrix and the lumped mass of every node.
        """
        points = np.ascontiguousarray(np.asarray(space.points, dtype=float))
        cells = np.ascontiguousarray(np.asarray(space.cells, dtype=np.int64))
        return FieldLowering._cached_mesh_matrices(points.tobytes(), cells.tobytes(), points.shape, cells.shape)

    @staticmethod
    def _mesh_laplacian(space: MeshSpace) -> csr_matrix:
        """Return the discrete Laplacian of linear elements with lumped mass, `-M^-1 K`, as a sparse matrix.

        Args:
            space:
                The mesh.

        Returns:
            laplacian:
                The sparse matrix that maps nodal values to the nodal Laplacian.
        """
        stiffness, lumped = FieldLowering._mesh_matrices(space)
        return -diags(1.0 / lumped) @ stiffness

    @staticmethod
    @functools.cache
    def _wavenumbers(dimensions: tuple[int, ...], spacings: tuple[float, ...]) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the angular wavenumbers of the real Fourier modes of a periodic lattice, broadcast along each axis."""
        axes = [2 * np.pi * np.fft.fftfreq(size, d=step) for size, step in zip(dimensions[:-1], spacings, strict=False)]
        axes.append(2 * np.pi * np.fft.rfftfreq(dimensions[-1], d=spacings[-1]))
        return tuple(np.meshgrid(*axes, indexing='ij'))

    @staticmethod
    @functools.cache
    def _dealiasing_mask(dimensions: tuple[int, ...]) -> npt.NDArray[np.bool_]:
        """Return the Fourier modes the two-thirds rule keeps, below two thirds of the largest along every axis."""
        axes = [np.abs(np.fft.fftfreq(size, d=1.0 / size)) < size / 3 for size in dimensions[:-1]]
        axes.append(np.fft.rfftfreq(dimensions[-1], d=1.0 / dimensions[-1]) < dimensions[-1] / 3)
        grids = np.meshgrid(*axes, indexing='ij')
        return np.logical_and.reduce(grids)

    @staticmethod
    def _spectral_multiplier(
        name: str,
        axis: int,
        dimensions: tuple[int, ...],
        spacings: tuple[float, ...],
        screening: float = 0.0,
    ) -> npt.NDArray[np.complex128]:
        """Return the Fourier multiplier of an operator on a periodic lattice.

        The Laplacian and its inverse act on every mode, so the Laplacian damps the Nyquist mode by `-|k|^2` like any
        other. A first derivative is zero at the Nyquist mode of its axis, and keeps only the modes of the two-thirds
        rule, which dealiases the derivatives of products.

        Args:
            name:
                `'laplacian'`, `'partial'`, or `'inverse_helmholtz'`.

            axis:
                The axis of a derivative.

            dimensions:
                The lattice shape.

            spacings:
                The distance between sites along every axis.

            screening:
                The screening of the inverse Helmholtz operator.

        Returns:
            multiplier:
                The factor of every real Fourier mode.
        """
        waves = FieldLowering._wavenumbers(dimensions, spacings)
        squared = np.sum([wave**2 for wave in waves], axis=0)
        if name == 'laplacian':
            return -squared.astype(complex)
        if name == 'inverse_helmholtz':
            screened = squared + screening**2
            return np.where(screened > 0, -1.0 / np.where(screened > 0, screened, 1.0), 0.0).astype(complex)
        factor = 1j * waves[axis]
        if dimensions[axis] % 2 == 0:
            nyquist = np.isclose(np.abs(waves[axis]), np.pi / spacings[axis])
            factor = np.where(nyquist, 0.0, factor)
        return np.where(FieldLowering._dealiasing_mask(dimensions), factor, 0.0)
