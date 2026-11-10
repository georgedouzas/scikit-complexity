"""Lower differential operators on fields to stencils and graph Laplacians."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import scipy.fft
from sage.all__sagemath_symbolics import SR, Expression, fast_callable
from sage.modules.free_module_element import FreeModuleElement
from scipy.ndimage import convolve, map_coordinates
from scipy.signal import oaconvolve
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import factorized
from skfem import Basis, ElementTetP1, ElementTriP1, asm
from skfem.models.poisson import laplace, mass

from ...components.elements import FieldElement
from ...components.laws import BaseLaw, BoundaryLaw, FluxLaw, NoiseLaw, RateLaw, UpdateLaw
from ...components.laws.utils import coordinate, displacement, integration_variable, where
from ...components.laws.utils._operators import COLLISION_COSINE, COLLISION_SPEED, INVERSION_BOUNDARIES
from ...components.laws.utils._phase import COORDINATE_PREFIX
from ...components.spaces import (
    AdjacencyNeighborhood,
    LatticeNeighborhood,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
    PhaseSpace,
    SimplexSpace,
)
from ...errors import ValidationError
from ...specs import NeighborReadSpec, RunStateSpec
from ...types import Array, Indices, IntTuple
from .._utils import domain_of, symbol
from ..problems._collision_kernel import CollisionKernel
from ..problems._declarations_spec import DeclarationsSpec
from ..problems._diffusive_limit_spec import DiffusiveLimitSpec
from ..problems._expression_printer import ExpressionPrinter
from ..problems._field_operator_kernel import FieldOperatorKernel
from ..problems._kernel_spec import KernelSpec
from ..problems._law_kernel import LATTICE_BOUNDARIES, MODES
from ..problems._reference_spec import ReferenceSpec
from ..types import ReadKind
from ._base import BaseLowering, BasePass
from ._compilation import Compilation

OPERATOR_NAMES = (
    'laplacian',
    'partial',
    'upwind',
    'inverse_helmholtz',
    'convolve',
    'shifted',
    'reduce_axes',
    'maximum_entropy',
    'collision_integral',
    'integral_transform',
    'cumulative_integral',
)
PHASE_OPERATOR_NAMES = ('reduce_axes', 'maximum_entropy', 'collision_integral', 'integral_transform')
NEWTON_STEPS = 50
NEWTON_TOLERANCE = 1e-14
LARGEST_EXPONENT_CHANGE = 5.0
TIME = symbol('t')
PLANE = 2
SMALL_KERNEL = 7
WHOLE_SITE = 1e-9
SHIFTED_MODES = {'periodic': 'grid-wrap', 'zero-flux': 'reflect', 'fixed': 'grid-constant'}
FILTER_MODES = {'periodic': 'wrap', 'zero-flux': 'reflect', 'fixed': 'constant'}


class FieldLowering(BaseLowering[list[BaseLaw], list[BaseLaw]]):
    """A lowering of the differential operators in the laws of one part to stencils.

    The lowering also returns the symbols the stencils read. `prepare(compilation)` reads the fields and the initial
    values of the compilation, and `run(laws)` returns the laws with every operator replaced by its stencil, noting the
    stable steps of the Laplacians of rate laws.

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

        applied_:
            The lowered value of every operator call so far, by its name and its lowered operands, so an identical call
            shares one value.
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
    applied_: dict[tuple[object, ...], Expression]

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
        self.fields_ = self._get_field_entries(declarations)
        self.initial_ = self._get_initial_values(self.spec, declarations)
        self.shifted_, self.derived_, self.tables_, self.solution_boundaries_, self.applied_ = {}, {}, {}, {}, {}
        self.limits_, self.volume_limits_ = [], []
        self.written_ = defaultdict(list)

    def _run(self, inputs: list[BaseLaw] | None) -> None:
        """Set laws whose differential operators are lowered.

        The stable steps of rate laws are noted along the way.

        Args:
            inputs:
                The laws of the component.
        """
        result: list[BaseLaw] = []
        for law in inputs or []:
            if isinstance(law, FluxLaw) and self._get_law_has_operators(law):
                result.append(
                    BasePass._get_replaced(law, flux=[self._lower(SR(item)) for item in self._get_components(law)]),
                )
                continue
            if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw) or not self._has_operators(law.value):
                result.append(law)
                continue
            if isinstance(law, RateLaw):
                self._note_limit(law.value)
            lowered = self._lower(law.value)
            if isinstance(law.value, Expression):
                for reference in self.spec._get_target_references(law._get_targets()[0]):
                    self.written_[(reference.entry.name, reference.column)].append((lowered, law.value))
            result.append(BasePass._get_replaced(law, value=lowered))
        self.lowered_ = result

    @staticmethod
    def _get_field_entries(declarations: DeclarationsSpec) -> dict[str, FieldElement]:
        """Return the fields by the name of each of their entries."""
        return {
            entry.name: block.node.component
            for block in declarations.blocks.values()
            if isinstance(block.node.component, FieldElement)
            for entry in block.entries.values()
        }

    @staticmethod
    def _get_initial_values(spec: KernelSpec, declarations: DeclarationsSpec) -> dict[Expression, float]:
        """Return the initial value of every symbol of a component.

        For groups, it is the value of the first member.
        """
        states = np.concatenate([np.zeros(0), *declarations.initial_states])
        parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
        return {
            item: float((states if reference.is_state else parameters)[reference.entry.start + reference.column])
            for item, reference in spec.references.items()
            if reference.entry.start + reference.column < (states if reference.is_state else parameters).size
        }

    def _get_known_reference(self, item: Expression) -> ReferenceSpec:
        """Return the reference of a symbol of the spec.

        A symbol the spec does not read raises an error.

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
        reference = self._get_reference_of(item, self.spec)
        if reference is None:
            error_msg = f'A law of {self.spec.owner} reads {item}, which is not a state or parameter of its namespace.'
            raise ValidationError(error_msg)
        return reference

    def _get_field_of(self, item: Expression) -> FieldElement | None:
        """Return the field whose member a symbol reads, or `None` for a single value."""
        reference = self._get_known_reference(item)
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

    def _get_table(
        self,
        item: Expression,
        place: FieldElement,
        offset: IntTuple | int,
    ) -> tuple[Indices, float | None, int | None]:
        """Return the neighbour table, fixed value, and mode of a symbol at an offset."""
        reference = self._get_known_reference(item)
        variable = reference.entry.name.rpartition('.')[2]
        if reference.is_state:
            kind, value = place._get_boundary(variable, reference.column)
        else:
            kind, value = (
                'periodic' if getattr(place.space, 'boundary', 'periodic') == 'periodic' else 'zero-flux',
                0.0,
            )
        key = (id(place), kind, value, offset)
        if key not in self.tables_:
            space = place.space
            if isinstance(space, NetworkSpace):
                table = space._get_index_table(AdjacencyNeighborhood())
                sites = np.arange(table.shape[0])[:, None]
                self.tables_[key] = (np.where(table == table.shape[0], sites, table)[:, offset], None)
            else:
                if not isinstance(space, LatticeSpace) or isinstance(offset, int) or not isinstance(value, float):
                    error_msg = (
                        f'A law of {self.spec.owner} reads {item} at a lattice offset, which needs a field on a '
                        'lattice with a number outside fixed edges.'
                    )
                    raise ValidationError(error_msg)
                lattice = space._copy_with_edges(LATTICE_BOUNDARIES[kind], value)
                table = lattice._get_index_table(LatticeNeighborhood((offset,)))[:, 0]
                self.tables_[key] = (table, value if kind == 'fixed' else None)
        table, ghost = self.tables_[key]
        return table, ghost, self._read_mode(place.space, kind, offset)

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
                replacements[item] = self._get_shifted_derived(base, self._get_total_offset(current, offset))
                continue
            reference = self._get_reference_of(base, self.spec)
            if reference is None:
                continue
            if reference.source is ReadKind.NEIGHBOR and base == item and base not in self.shifted_:
                error_msg = (
                    f'A law of {self.spec.owner} differentiates {base}, a field over position read on a phase space, '
                    'by a stencil. Use the finite-volume discretization, whose fluxes read it cell by cell.'
                )
                raise ValidationError(error_msg)
            place = self._get_field_of(base)
            if place is None:
                continue
            total = self._get_total_offset(current, offset)
            if not isinstance(total, int) and not any(total):
                replacements[item] = base
                continue
            name = f'{base}__{self._tag(total)}'
            target = symbol(name, domain_of(base))
            if self._get_reference_of(target, self.spec) is None:
                reference = self._get_known_reference(base)
                table, ghost, mode = self._get_table(base, place, total)
                space = place.space
                lattice = space if isinstance(space, LatticeSpace) and mode is not None else None
                read = NeighborReadSpec(
                    table,
                    ghost,
                    None if lattice is None else lattice.shape_,
                    total if lattice is not None and not isinstance(total, int) else None,
                    mode or 0,
                )
                self._add_reference(
                    self.spec,
                    target,
                    ReferenceSpec(reference.entry, reference.is_state, reference.column, ReadKind.NEIGHBOR, read),
                )
                self.shifted_[target] = (base, total)
            replacements[item] = target
        return expression.subs(replacements)

    def _read_mode(self, space: object, kind: str, offset: IntTuple | int) -> int | None:
        """Return the mode of the edges a read at an offset crosses.

        It is `None` when the read crosses edges of two kinds.
        """
        if not isinstance(space, PhaseSpace) or isinstance(offset, int):
            return MODES[kind]
        kinds = {
            kind if axis in space.position_axes_ else ('periodic' if space.axes_periodic_[axis] else 'zero-flux')
            for axis in range(space.n_dim_)
            if offset[axis]
        }
        return MODES[kinds.pop() if kinds else kind] if len(kinds) < PLANE else None

    def _get_total_offset(self, current: IntTuple | int | None, offset: IntTuple | int) -> IntTuple | int:
        """Return the offset of a read at an offset from a read that is already shifted."""
        if isinstance(offset, int) or isinstance(current, int):
            if current is not None:
                error_msg = f'A law of {self.spec.owner} nests operators on a network, which only take one step.'
                raise ValidationError(error_msg)
            return offset
        return tuple(a + b for a, b in zip(current or (0,) * len(offset), offset, strict=True))

    def _get_boundaries(self, expression: Expression) -> list[tuple[str, object]]:
        """Return the boundary conditions of the field states an expression reads.

        The states read through derived values count too.
        """
        found = []
        for item in SR(expression).variables():
            if item in self.solution_boundaries_:
                found.append(self.solution_boundaries_[item])
                continue
            if item in self.derived_:
                found += self._get_boundaries(self.derived_[item][1].operand)
                continue
            base = self.shifted_.get(item, (item, None))[0]
            reference = self._get_reference_of(base, self.spec)
            if reference is None or not reference.is_state or reference.source is not ReadKind.MEMBER:
                continue
            place = self._get_field_of(base)
            if place is not None:
                found.append(place._get_boundary(reference.entry.name.rpartition('.')[2], reference.column))
        return found

    def _get_boundary(self, expression: Expression, place: FieldElement) -> tuple[str, object]:
        """Return the one boundary condition of the field states an expression reads.

        Without one, it is the edges of the space.
        """
        found = set(self._get_boundaries(expression))
        if len(found) > 1:
            error_msg = (
                f'A law of {self.spec.owner} applies a nonlocal operator to {expression}, whose states have different '
                'boundary conditions. Give the states of the operand one boundary condition, or give an inversion '
                'the boundary of its solution.'
            )
            raise ValidationError(error_msg)
        return found.pop() if found else place._get_boundary('')

    def _get_neighbor_table(self, place: FieldElement, kind: str, value: object, offset: IntTuple | int) -> Indices:
        """Return the neighbour of every site at an offset.

        Reads beyond fixed edges get the number of sites.
        """
        space = place.space
        if isinstance(space, NetworkSpace) and isinstance(offset, int):
            table = space._get_index_table(AdjacencyNeighborhood())
            sites = np.arange(table.shape[0])[:, None]
            return np.asarray(np.where(table == table.shape[0], sites, table)[:, offset], dtype=np.intp)
        if not isinstance(space, LatticeSpace) or isinstance(offset, int) or not isinstance(value, float):
            error_msg = (
                f'A law of {self.spec.owner} reads a nonlocal value at a neighbour, which needs a field on a lattice '
                'or a network with a number outside fixed edges.'
            )
            raise ValidationError(error_msg)
        lattice = space._copy_with_edges(LATTICE_BOUNDARIES[kind], value)
        return np.asarray(lattice._get_index_table(LatticeNeighborhood((offset,)))[:, 0], dtype=np.intp)

    def _get_shifted_derived(self, base: Expression, total: IntTuple | int) -> Expression:
        """Return the symbol of a derived value read at a neighbouring site.

        Its values are indexed by the neighbour table.

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
        kind, value = self._get_boundary(base, place)
        table = self._get_neighbor_table(place, kind, value, total)
        size = table.shape[0]
        inside = table < size
        matrix = csr_matrix((np.ones(inside.sum()), (np.flatnonzero(inside), table[inside])), shape=(size, size))
        ghost = float(value) if kind == 'fixed' and isinstance(value, float) else 0.0

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.append(values, ghost)[table]

        read = FieldOperatorKernel(operand=base, symbols=[base], apply=apply, matrix=matrix).prepare()
        self.shifted_[target] = (base, total)
        return self._get_derived(place, read, target)

    def _get_places(self, expression: Expression) -> dict[int, FieldElement]:
        """Return the fields an expression reads.

        The fields are read through member symbols, neighbour reads, and derived values.
        """
        places: dict[int, FieldElement] = {}
        place: FieldElement | None
        for item in SR(expression).variables():
            if item in self.derived_:
                place = self.derived_[item][0]
            else:
                base = self.shifted_.get(item, (item, None))[0]
                place = self._get_field_of(base) if self._get_reference_of(base, self.spec) is not None else None
            if place is not None:
                places[id(place)] = place
        return places

    def _place(self, expression: Expression) -> FieldElement:
        """Return the one field an expression reads, or raise an error."""
        places = self._get_places(expression)
        discretizations = {place.discretization for place in places.values()}
        spaces = {id(place.space) for place in places.values()}
        if len(spaces) != 1 or len(discretizations) != 1:
            error_msg = (
                f'A law of {self.spec.owner} applies an operator to {expression}, which reads {len(spaces)} fields. An '
                'operator acts on the states of fields of one space and one discretization.'
            )
            raise ValidationError(error_msg)
        return next(iter(places.values()))

    def _get_space(self, expression: Expression) -> LatticeSpace | NetworkSpace:
        """Return the space of the fields an expression reads."""
        spaces = {id(place.space): place.space for place in self._get_places(expression).values()}
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

    def _get_laplacian(self, expression: Expression) -> Expression:
        """Return the stencil of the Laplacian of an expression."""
        space = self._get_space(expression)
        if isinstance(space, NetworkSpace):
            width = space._get_index_table(AdjacencyNeighborhood()).shape[1]
            return sum((self._shift(expression, slot) - expression for slot in range(width)), SR(0))
        total = SR(0)
        for axis in range(space.n_dim_):
            step = tuple(int(axis == other) for other in range(space.n_dim_))
            back = tuple(-item for item in step)
            total += (self._shift(expression, step) + self._shift(expression, back) - 2 * expression) / (
                space.spacings_[axis] ** 2
            )
        return total

    def _get_lattice(self, expression: Expression, axis: int, name: str) -> tuple[LatticeSpace, IntTuple, IntTuple]:
        """Return the lattice of an expression and the unit offsets along an axis."""
        space = self._get_space(expression)
        if not isinstance(space, LatticeSpace) or not 0 <= axis < space.n_dim_:
            error_msg = (
                f'A law of {self.spec.owner} takes the {name} of {expression} along axis {axis}, which needs a lattice.'
            )
            raise ValidationError(error_msg)
        step = tuple(int(axis == other) for other in range(space.n_dim_))
        return space, step, tuple(-item for item in step)

    def _lower(self, expression: object) -> object:
        """Return an expression with every operator replaced by its stencil.

        The innermost operators are replaced first.

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
        key = (name, *operands)
        if key not in self.applied_:
            self.applied_[key] = self._get_lowered_call(name, operands)
        return self.applied_[key]

    def _get_lowered_call(self, name: str, operands: list[object]) -> Expression:
        """Return the value of one operator whose operands are lowered."""
        if name == 'shifted':
            return self._get_shifted(SR(operands[0]), [SR(item) for item in operands[1:]])
        if name in {'laplacian', 'partial', 'upwind'} and not SR(operands[0]).variables():
            return SR(0)
        place = self._place(SR(operands[0]))
        if name in PHASE_OPERATOR_NAMES:
            return self._get_phase_operator(name, [SR(item) for item in operands], place)
        if name == 'convolve':
            return self._convolve(SR(operands[0]), SR(operands[1]), SR(operands[2]), place)
        if name == 'cumulative_integral':
            return self._get_cumulative(SR(operands[0]), int(SR(operands[1])), bool(SR(operands[2])), place)
        return self._get_discretized(name, operands, place)

    def _get_phase_operator(self, name: str, operands: list[Expression], place: FieldElement) -> Expression:
        """Return the value of an operator over the class axes of a phase space."""
        if name == 'reduce_axes':
            return self._get_reduced(operands[0], operands[1:], place)
        if name == 'maximum_entropy':
            return self._get_equilibrium(operands[0], int(operands[1]), operands[2:], place)
        if name == 'integral_transform':
            return self._get_transformed(operands[0], operands[1], operands[2:], place)
        return self._get_collision(operands[0], operands[1], place)

    def _get_phase_space(self, operand: Expression, place: FieldElement) -> PhaseSpace:
        """Return the phase space of a field whose class axes an operator reads.

        Another space raises an error.
        """
        if not isinstance(place.space, PhaseSpace):
            error_msg = (
                f'A law of {self.spec.owner} reduces {operand} over class axes, but its field lives on '
                f'{type(place.space).__name__}. Give the field a `PhaseSpace`.'
            )
            raise ValidationError(error_msg)
        return place.space

    def _get_axes_of(self, coordinates: list[Expression], space: PhaseSpace) -> tuple[int, ...]:
        """Return the indices of the class axes of their coordinates.

        An axis the space lacks raises an error.
        """
        names = [str(item).removeprefix(COORDINATE_PREFIX) for item in coordinates]
        missing = [name for name in names if name not in space.class_axes_]
        if missing:
            error_msg = (
                f'A law of {self.spec.owner} reads the class axes {missing}, but the phase space holds the axes '
                f'{list(space.class_axes_)}.'
            )
            raise ValidationError(error_msg)
        return tuple(space.class_axes_[name] for name in names)

    def _get_reduced(self, operand: Expression, coordinates: list[Expression], place: FieldElement) -> Expression:
        """Return the symbol of the integral of an expression over class axes.

        The integral is broadcast back along the axes.
        """
        space = self._get_phase_space(operand, place)
        axes = self._get_axes_of(coordinates, space)
        width = float(np.prod([space.spacings_[axis] for axis in axes]))
        dimensions = space.shape_

        over_positions = self.spec.rows == space.position_size_ != space.size_

        volumes = self._get_volumes(space)

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            reduced = (np.reshape(values, dimensions) * volumes).sum(axis=axes, keepdims=True) * width
            return reduced.ravel() if over_positions else np.broadcast_to(reduced, dimensions).ravel()

        read = FieldOperatorKernel(operand=operand, symbols=self._get_symbols(operand), apply=apply)
        return self._get_derived(place, read.prepare())

    def _get_transformed(
        self,
        operand: Expression,
        kernel: Expression,
        coordinates: list[Expression],
        place: FieldElement,
    ) -> Expression:
        """Return the symbol of the integral of an expression against a two-point kernel.

        The integral runs over class axes.
        """
        space = self._get_phase_space(operand, place)
        axes = self._get_axes_of(coordinates, space)
        names = [str(item).removeprefix(COORDINATE_PREFIX) for item in coordinates]
        sizes = [space.shape_[axis] for axis in axes]
        targets = np.meshgrid(*(space.coordinates_[name] for name in names), indexing='ij')
        points = [np.reshape(item, (-1, 1)) for item in targets]
        sources = [np.reshape(item, (1, -1)) for item in targets]
        symbols = [coordinate(name) for name in names] + [integration_variable(name) for name in names]
        sampled = self._get_sampled_kernel(kernel, symbols, [*points, *sources], 'integrates against')
        width = float(np.prod([space.spacings_[axis] for axis in axes]))
        matrix = np.broadcast_to(sampled, (int(np.prod(sizes)),) * 2) * width
        volumes = np.broadcast_to(self._get_volumes(space), space.shape_)
        dimensions = space.shape_
        ends = tuple(range(len(dimensions) - len(axes), len(dimensions)))

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            moved = np.moveaxis(np.reshape(values, dimensions) * volumes, axes, ends)
            flat = np.reshape(moved, (*moved.shape[: -len(axes)], -1)) @ matrix.T
            return np.moveaxis(np.reshape(flat, moved.shape), ends, axes).ravel()

        read = FieldOperatorKernel(operand=operand, symbols=self._get_symbols(operand), apply=apply)
        return self._get_derived(place, read.prepare())

    def _get_cumulative(self, operand: Expression, axis: int, reverse: bool, place: FieldElement) -> Expression:
        """Return the symbol of the integral of an expression along a lattice axis.

        The integral runs up to every site or beyond it.
        """
        space = place.space
        if not isinstance(space, LatticeSpace) or not 0 <= axis < space.n_dim_:
            error_msg = (
                f'A law of {self.spec.owner} integrates {operand} along axis {axis}, which needs a lattice with that '
                'axis. Give the field a lattice space, or integrate along one of its axes.'
            )
            raise ValidationError(error_msg)
        dimensions = space.shape_
        volumes = self._get_volumes(space)
        width = space.spacings_[axis]

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            cells = np.reshape(values, dimensions) * volumes * width
            ordered = np.flip(cells, axis) if reverse else cells
            totals = np.cumsum(ordered, axis=axis) - ordered / 2
            return (np.flip(totals, axis) if reverse else totals).ravel()

        read = FieldOperatorKernel(operand=operand, symbols=self._get_symbols(operand), apply=apply)
        return self._get_derived(place, read.prepare())

    def _get_equilibrium(
        self,
        operand: Expression,
        order: int,
        coordinates: list[Expression],
        place: FieldElement,
    ) -> Expression:
        """Return the symbol of the density with the largest entropy.

        The density keeps the moments of an expression.
        """
        space = self._get_phase_space(operand, place)
        apply = self._get_equilibrium_apply(space, self._get_axes_of(coordinates, space), order)
        read = FieldOperatorKernel(operand=operand, symbols=self._get_symbols(operand), apply=apply)
        return self._get_derived(place, read.prepare())

    def _get_collision_kernel(self, kernel: Expression) -> Callable[[Array, Array], Array]:
        """Return the NumPy function of speeds and cosines that a collision kernel gives.

        A kernel that reads a state or a parameter raises an error, since the weights are assembled once.
        """
        arguments = [COLLISION_SPEED, COLLISION_COSINE]
        references = self.spec.references
        read = sorted(
            references[item].entry.name if item in references else str(item)
            for item in kernel.variables()
            if item not in arguments
        )
        if read:
            error_msg = (
                f'The collision integral of a law of {self.spec.owner} has a kernel that reads {", ".join(read)}, but '
                'its weights are assembled once. Write the kernel with the speed, the cosine, and numbers.'
            )
            raise ValidationError(error_msg)
        if kernel.is_numeric():
            value = float(kernel)
            return lambda speed, cosine: np.full(np.shape(speed), value)
        return np.vectorize(fast_callable(kernel, vars=arguments, domain=float), otypes=[float])

    def _get_collision(self, operand: Expression, kernel: Expression, place: FieldElement) -> Expression:
        """Return the symbol of the binary collision integral of a density with a kernel.

        The integral runs over the class axes of its phase space.
        """
        space = self._get_phase_space(operand, place)
        spacings = (space.spacings_[-PLANE], space.spacings_[-1])
        kernel = CollisionKernel(space.shape_, spacings, self._get_collision_kernel(kernel)).prepare()
        empty = np.zeros(0)

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return kernel.run(RunStateSpec(0.0, values, empty)).values_

        read = FieldOperatorKernel(operand=operand, symbols=self._get_symbols(operand), apply=apply)
        return self._get_derived(place, read.prepare())

    def _get_discretized(self, name: str, operands: list[object], place: FieldElement) -> Expression:
        """Return a local operator or inversion lowered for its field's discretization."""
        if isinstance(place.space, SimplexSpace):
            error_msg = (
                f'A law of {self.spec.owner} applies {name} to a field on a simplex, which only takes finite-volume '
                'fluxes, reductions, and integrals. Write the transport as a flux of the finite-volume discretization.'
            )
            raise ValidationError(error_msg)
        if place.discretization == 'spectral':
            return self._get_spectral(name, operands, place)
        if name == 'inverse_helmholtz':
            given = self._get_given_boundary(operands[2:])
            return self._get_inverse(SR(operands[0]), self._get_screening(SR(operands[1])), place, given)
        if place.discretization == 'finite-element':
            return self._get_finite_element(name, SR(operands[0]), place)
        return self._get_stencil(name, operands)

    def _get_stencil(self, name: str, operands: list[object]) -> Expression:
        """Return the finite-difference stencil of an operator on a lattice or a network."""
        if name == 'laplacian':
            return self._get_laplacian(SR(operands[0]))
        axis = int(SR(operands[1]))
        value = SR(operands[0])
        space, step, back = self._get_lattice(value, axis, 'derivative' if name == 'partial' else 'advection')
        if name == 'partial':
            return (self._shift(value, step) - self._shift(value, back)) / (2 * space.spacings_[axis])
        velocity = SR(operands[2])
        behind = (value - self._shift(value, back)) / space.spacings_[axis]
        ahead = (self._shift(value, step) - value) / space.spacings_[axis]
        return velocity * where(velocity > 0, behind, ahead)

    def _get_derived(
        self,
        place: FieldElement,
        read: FieldOperatorKernel,
        target: Expression | None = None,
    ) -> Expression:
        """Return a symbol that stands for a derived value of a field, with its read.

        The symbol is a new one by default.
        """
        target = symbol(f'_derived_{place.label}_{len(self.derived_)}') if target is None else target
        references = [self._get_known_reference(item) for item in read.symbols]
        entry = next(
            reference.entry
            for reference in references
            if reference.source in {ReadKind.MEMBER, ReadKind.NEIGHBOR, ReadKind.DERIVED}
        )
        self._add_reference(self.spec, target, ReferenceSpec(entry, True, 0, ReadKind.DERIVED, read))
        self.derived_[target] = (place, read)
        return target

    def _get_spectral(self, name: str, operands: list[object], place: FieldElement) -> Expression:
        """Return the symbol of a spectral operator applied to an expression.

        Chains of operators are composed.
        """
        space = place.space
        if not isinstance(space, LatticeSpace) or space.boundary != 'periodic':
            error_msg = f'A law of {self.spec.owner} uses a spectral field, which needs a periodic lattice.'
            raise ValidationError(error_msg)
        if name == 'upwind':
            velocity = SR(operands[2])
            return velocity * self._get_spectral('partial', [operands[0], operands[1]], place)
        if name == 'inverse_helmholtz' and self._get_given_boundary(operands[2:]) not in {None, ('periodic', 0.0)}:
            error_msg = (
                f'A law of {self.spec.owner} inverts on a spectral field with a boundary that is not periodic. Give '
                'the inversion no boundary, or use the finite-difference discretization.'
            )
            raise ValidationError(error_msg)
        axis = int(SR(operands[1])) if name == 'partial' else 0
        if name == 'partial' and not 0 <= axis < space.n_dim_:
            error_msg = f'A law of {self.spec.owner} takes a derivative along axis {axis} of a {space.n_dim_}D lattice.'
            raise ValidationError(error_msg)
        screening = self._get_screening(SR(operands[1])) if name == 'inverse_helmholtz' else 0.0
        if isinstance(space, PhaseSpace) and name == 'partial' and space._get_axis_boundary(axis) != 'periodic':
            error_msg = (
                f'A law of {self.spec.owner} takes a spectral derivative along axis {axis}, which is an interval. Use '
                'a periodic class axis or the finite-volume discretization.'
            )
            raise ValidationError(error_msg)
        multiplier = self._get_spectral_multiplier(
            name,
            axis,
            space.shape_,
            space.spacings_,
            screening,
            axes=self._get_transported_axes(space, name),
        )
        return self._get_multiplied(SR(operands[0]), multiplier, place)

    def _get_multiplied(
        self,
        operand: Expression,
        multiplier: npt.NDArray[np.complex128],
        place: FieldElement,
    ) -> Expression:
        """Return the symbol of a Fourier multiplier applied to an expression.

        Chains of multipliers are composed.
        """
        dimensions = place.space.shape_ if isinstance(place.space, LatticeSpace) else ()
        inner = self.derived_.get(operand)
        if inner is not None and inner[1].multiplier is not None:
            operand, multiplier = inner[1].operand, multiplier * inner[1].multiplier
        symbols = sorted(
            (item for item in operand.variables() if self._get_reference_of(item, self.spec) is not None),
            key=str,
        )
        read = FieldOperatorKernel(
            operand=operand,
            symbols=symbols,
            apply=self._get_spectral_apply(multiplier, dimensions),
            multiplier=multiplier,
            dimensions=dimensions,
        ).prepare()
        return self._get_derived(place, read)

    def _get_symbols(self, expression: Expression) -> list[Expression]:
        """Return the symbols of an expression that the spec reads, sorted by name."""
        return sorted(
            (item for item in SR(expression).variables() if self._get_reference_of(item, self.spec) is not None),
            key=str,
        )

    def _get_screening(self, screening: Expression) -> float:
        """Return the value of a screening at the current parameters.

        A value that is not a real number raises an error.
        """
        value = SR(screening).subs(self.initial_)
        if value.variables() or not value.is_real() or float(value) < 0:
            error_msg = (
                f'A law of {self.spec.owner} inverts with the screening {screening}, which is not a non-negative '
                'number of the parameters. Use a screening whose square is non-negative and that reads no state.'
            )
            raise ValidationError(error_msg)
        return float(value)

    def _get_given_boundary(self, operands: list[object]) -> tuple[str, float] | None:
        """Return the boundary condition given to an inversion by its code and value.

        It is `None` when none is given.
        """
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

    def _get_inverse(
        self,
        operand: Expression,
        screening: float,
        place: FieldElement,
        given: tuple[str, float] | None = None,
    ) -> Expression:
        """Return the symbol of the solution of a modified-Helmholtz equation.

        The equation is on a lattice, a network, or a mesh.

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
        kind, value = self._get_boundary(operand, place) if given is None else given
        apply: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        if isinstance(space, NetworkSpace) and given is not None:
            error_msg = f'A law of {self.spec.owner} inverts on a network with a boundary. Give the inversion none.'
            raise ValidationError(error_msg)
        if isinstance(space, MeshSpace):
            apply = self._get_mesh_inverse(space, screening, kind, value)
        elif isinstance(space, NetworkSpace):
            apply = self._get_network_inverse(space, screening)
        elif isinstance(space, PhaseSpace) and (kind != 'fixed' or isinstance(value, float)):
            fill = float(value) if isinstance(value, float) else 0.0
            apply = self._get_over_positions(space, self._get_lattice_inverse(space.position, screening, kind, fill))
        elif isinstance(space, LatticeSpace) and (kind != 'fixed' or isinstance(value, float)):
            apply = self._get_lattice_inverse(space, screening, kind, float(value) if isinstance(value, float) else 0.0)
        else:
            error_msg = (
                f'A law of {self.spec.owner} inverts {operand} on a {type(space).__name__} with the boundary {kind}, '
                'which needs a lattice, a network, or a mesh, with a number outside fixed lattice edges.'
            )
            raise ValidationError(error_msg)
        read = FieldOperatorKernel(operand=operand, symbols=self._get_symbols(operand), apply=apply).prepare()
        solution = self._get_derived(place, read)
        if given is not None:
            self.solution_boundaries_[solution] = (kind, value)
        return solution

    def _get_parameter_values(self) -> dict[Expression, float]:
        """Return the initial value of every parameter the spec reads."""
        return {
            key: item
            for key, item in self.initial_.items()
            if not getattr(self._get_reference_of(key, self.spec), 'is_state', True)
        }

    def _get_sampled_kernel(
        self,
        kernel: Expression,
        symbols: list[Expression],
        grids: Sequence[npt.NDArray[np.float64]],
        action: str,
    ) -> npt.NDArray[np.float64]:
        """Return a kernel of some symbols and the parameters evaluated on their grids.

        Args:
            kernel:
                The kernel.

            symbols:
                The symbols the kernel is a function of besides the parameters.

            grids:
                The values of every symbol, arrays that broadcast together.

            action:
                What the operator does with the kernel in an error message, such as `'convolves with'`.

        Returns:
            sampled:
                The kernel at the broadcast points of the grids.

        Raises:
            ValidationError:
                The kernel reads a symbol that is neither one of the symbols nor a parameter.
        """
        parameters = {key: item for key, item in self._get_parameter_values().items() if key not in symbols}
        value = SR(kernel).subs(parameters)
        unknown = sorted(set(value.variables()) - set(symbols), key=str)
        if unknown:
            error_msg = (
                f'A law of {self.spec.owner} {action} the kernel {kernel}, which reads {", ".join(map(str, unknown))}. '
                f'Write the kernel in {", ".join(map(str, symbols))} and the parameters.'
            )
            raise ValidationError(error_msg)
        printer = ExpressionPrinter(time=TIME, compiled=True).prepare(symbols)
        function = ExpressionPrinter._get_function_of(printer.run([value]).code_)
        computed = ExpressionPrinter._get_values_of(function(0.0, *grids))[0]
        return np.asarray(computed, dtype=float)

    def _get_kernel_weights(
        self,
        kernel: Expression,
        space: LatticeSpace,
        radius: Expression,
        *,
        wrapped: bool,
    ) -> npt.NDArray[np.float64]:
        """Return a kernel sampled at the lattice displacements within a radius.

        The samples are multiplied by the cell volume.

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
        distance = float('inf') if SR(radius).is_infinity() else float(SR(radius).subs(self._get_parameter_values()))
        if wrapped:
            steps = [np.fft.fftfreq(size, d=1.0 / size) for size in space.shape_]
        else:
            reach = [
                size - 1 if np.isinf(distance) else min(size - 1, int(np.ceil(distance / step - WHOLE_SITE)))
                for size, step in zip(space.shape_, space.spacings_, strict=True)
            ]
            steps = [np.arange(-item, item + 1, dtype=float) for item in reach]
        grids = np.meshgrid(*[item * step for item, step in zip(steps, space.spacings_, strict=True)], indexing='ij')
        sampled = np.broadcast_to(self._get_sampled_kernel(kernel, axes, grids, 'convolves with'), grids[0].shape)
        inside = np.sqrt(sum((grid**2 for grid in grids), np.zeros(grids[0].shape))) <= distance * (1 + WHOLE_SITE)
        return np.where(inside, sampled, 0.0) * float(np.prod(space.spacings_))

    def _convolve(self, operand: Expression, kernel: Expression, radius: Expression, place: FieldElement) -> Expression:
        """Return the symbol of the convolution of an expression with a lattice kernel.

        The convolution runs along the position axes of the lattice.
        """
        space = place.space
        if not isinstance(space, LatticeSpace):
            error_msg = (
                f'A law of {self.spec.owner} convolves {operand} on a {type(space).__name__}, which needs a lattice '
                'space. Give the field a lattice space.'
            )
            raise ValidationError(error_msg)
        phase = isinstance(space, PhaseSpace)
        if place.discretization == 'spectral' and not phase:
            weights = self._get_kernel_weights(kernel, space, radius, wrapped=True)
            return self._get_multiplied(operand, np.fft.rfftn(weights).astype(complex), place)
        kind, value = self._get_boundary(operand, place)
        if kind == 'fixed' and not isinstance(value, float):
            error_msg = f'A law of {self.spec.owner} convolves {operand}, which needs a number outside fixed edges.'
            raise ValidationError(error_msg)
        lattice = space.position if isinstance(space, PhaseSpace) else space
        window = self._get_kernel_weights(kernel, lattice, radius, wrapped=False)
        window = np.reshape(window, window.shape + (1,) * (space.n_dim_ - window.ndim))
        if kind == 'periodic' and max(window.shape) > SMALL_KERNEL:
            modes = np.fft.rfftn(self._get_kernel_weights(kernel, lattice, radius, wrapped=True))
            multiplier = np.reshape(modes, modes.shape + (1,) * (space.n_dim_ - modes.ndim))
            apply = self._get_spectral_apply(multiplier, space.shape_, tuple(range(lattice.n_dim_)))
        else:
            apply = self._get_window_apply(
                window,
                space.shape_,
                kind,
                float(value) if isinstance(value, float) else 0.0,
            )
        matrix = None if SR(radius).is_infinity() else self._get_window_matrix(window, place, kind, value)
        read = FieldOperatorKernel(operand=operand, symbols=self._get_symbols(operand), apply=apply, matrix=matrix)
        return self._get_derived(place, read.prepare())

    def _get_window_matrix(
        self,
        window: npt.NDArray[np.float64],
        place: FieldElement,
        kind: str,
        value: object,
    ) -> csr_matrix:
        """Return the sparse matrix of the linear part of a convolution with a window.

        The window covers the sites within its radius.
        """
        size = place.size_
        total = csr_matrix((size, size))
        centre = np.asarray(window.shape) // 2
        for index in zip(*np.nonzero(window), strict=True):
            offset = tuple(int(item) for item in centre - np.asarray(index))
            table = self._get_neighbor_table(place, kind, value, offset)
            inside = table < size
            selection = (np.full(inside.sum(), window[index]), (np.flatnonzero(inside), table[inside]))
            total = total + csr_matrix(selection, shape=(size, size))
        return csr_matrix(total)

    def _get_shifted(self, operand: Expression, operands: list[Expression]) -> Expression:
        """Return an expression read at a physical displacement.

        A displacement of whole sites is a neighbour read, and others are interpolated.

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
        sites = self._get_whole_sites(offset, space.spacings_)
        if sites is not None:
            return self._shift(operand, sites)
        kind, value = self._get_boundary(operand, place)
        if kind == 'fixed' and not isinstance(value, float):
            error_msg = f'A law of {self.spec.owner} reads {operand} beyond fixed edges, which needs a number there.'
            raise ValidationError(error_msg)
        symbols = sorted(set(self._get_symbols(operand)).union(*[self._get_symbols(item) for item in offset]), key=str)
        apply = self._get_interpolated_apply(
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
        return self._get_derived(place, read.prepare())

    def _get_finite_element(self, name: str, operand: Expression, place: FieldElement) -> Expression:
        """Return the symbol of the finite element Laplacian of an expression."""
        if name != 'laplacian':
            error_msg = (
                f'A law of {self.spec.owner} takes the {name} of a finite element field, which takes Laplacians only.'
            )
            raise ValidationError(error_msg)
        if not isinstance(place.space, MeshSpace):
            error_msg = f'A law of {self.spec.owner} uses a finite element field, which needs a mesh.'
            raise ValidationError(error_msg)
        matrix = self._get_mesh_laplacian(place.space)
        symbols = sorted(
            (item for item in operand.variables() if self._get_reference_of(item, self.spec) is not None),
            key=str,
        )
        read = FieldOperatorKernel(
            operand=operand,
            symbols=symbols,
            apply=lambda values: matrix @ values,
            matrix=matrix,
        ).prepare()
        return self._get_derived(place, read)

    def _note_volume_limit(self, coefficient: Expression, stiffness: float) -> None:
        """Record the diffusive step limit of a finite-volume field.

        The limit is a function of the parameters it reads.
        """
        symbols = sorted(set(coefficient.variables()) - {TIME}, key=str)
        found = [self._get_reference_of(item, self.spec) for item in symbols]
        references = [reference for reference in found if reference is not None]
        if len(references) < len(symbols) or any(reference.is_state for reference in references):
            return
        function = ExpressionPrinter(time=TIME, compiled=True).prepare(symbols).run([coefficient]).code_
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
        for call in self._get_operator_calls(value, 'laplacian'):
            place = self._place(call.operands()[0])
            if place.discretization not in {'finite-difference', 'finite-volume'}:
                continue
            placeholder = symbol('_laplacian_')
            expression = SR(value.subs({call: placeholder})).diff(placeholder)
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
    def _get_basis(space: PhaseSpace, axes: tuple[int, ...], order: int) -> npt.NDArray[np.float64]:
        """Return the functions whose moments an equilibrium keeps at every class.

        There is one column per function.

        Args:
            space:
                The phase space.

            axes:
                The indices of the class axes of the equilibrium.

            order:
                The highest order of the moments, 0, 1, or 2.

        Returns:
            basis:
                The value 1, the coordinates for an order above 0, and their squared length for the order 2.
        """
        names = {index: name for name, index in space.class_axes_.items()}
        grids = np.meshgrid(*(space.coordinates_[names[axis]] for axis in axes), indexing='ij')
        points = [grid.ravel() for grid in grids]
        columns = [np.ones(points[0].size)]
        if order >= 1:
            columns += points
        if order == PLANE:
            columns.append(sum(point**2 for point in points))
        return np.column_stack(columns)

    @staticmethod
    def _get_newton(
        basis: npt.NDArray[np.float64],
        moments: npt.NDArray[np.float64],
        width: float,
        guess: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the exponents of the densities `exp(basis @ a)` of discrete moments.

        The exponents are found by damped Newton steps.

        Args:
            basis:
                The functions at every class, one column per function.

            moments:
                The discrete moments of every density, one row per density.

            width:
                The volume of a cell of the class axes.

            guess:
                The exponents to start from, one row per density.

        Returns:
            exponents:
                The exponents, one row per density.
        """
        exponents = guess.copy()
        scale = np.maximum(np.abs(moments).max(axis=1, keepdims=True), 1.0)
        for _ in range(NEWTON_STEPS):
            density = np.exp(exponents @ basis.T)
            error = width * density @ basis - moments
            if np.abs(error / scale).max() <= NEWTON_TOLERANCE:
                break
            hessian = width * np.einsum('ij,jk,jl->ikl', density, basis, basis, optimize=True)
            step = np.linalg.solve(hessian, -error[..., None])[..., 0]
            largest = np.abs(step @ basis.T).max(axis=1, keepdims=True)
            exponents += step * np.minimum(1.0, LARGEST_EXPONENT_CHANGE / np.maximum(largest, 1e-300))
        return exponents

    @staticmethod
    def _get_equilibrium_apply(
        space: PhaseSpace,
        axes: tuple[int, ...],
        order: int,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function of a density to its maximum-entropy equilibrium.

        It maps the values of the density at every position.

        Args:
            space:
                The phase space.

            axes:
                The indices of the class axes of the equilibrium.

            order:
                The highest order of the moments that are kept.

        Returns:
            apply:
                The function of the values at every site.
        """
        basis = FieldLowering._get_basis(space, axes, order)
        width = float(np.prod([space.spacings_[axis] for axis in axes]))
        moved = tuple(range(space.n_dim_ - len(axes), space.n_dim_))
        dimension = len(axes)

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            grid = np.moveaxis(np.reshape(values, space.shape_), axes, moved)
            shape = grid.shape
            rows = grid.reshape(-1, basis.shape[0])
            moments = width * rows @ basis
            total = moments[:, 0]
            filled = total > 0
            result = np.zeros(rows.shape)
            if filled.any():
                guess = FieldLowering._get_equilibrium_guess(moments[filled], width, basis.shape[0], dimension)
                exponents = FieldLowering._get_newton(basis, moments[filled], width, guess)
                result[filled] = np.exp(exponents @ basis.T)
            return np.moveaxis(result.reshape(shape), moved, axes).ravel()

        return apply

    @staticmethod
    def _get_equilibrium_guess(
        moments: npt.NDArray[np.float64],
        width: float,
        classes: int,
        dimension: int,
    ) -> npt.NDArray[np.float64]:
        """Return the exponents of the continuous Maxwellian of given moments.

        Below order 2, it is a uniform density.
        """
        guess = np.zeros(moments.shape)
        total = moments[:, 0]
        if moments.shape[1] == dimension + PLANE:
            mean = moments[:, 1:-1] / total[:, None]
            squared = np.sum(mean**2, axis=1)
            spread = np.maximum((moments[:, -1] / total - squared) / dimension, width ** (2 / dimension))
            guess[:, 0] = np.log(total) - dimension * np.log(2 * np.pi * spread) / 2 - squared / (2 * spread)
            guess[:, 1:-1] = mean / spread[:, None]
            guess[:, -1] = -1 / (2 * spread)
            return guess
        guess[:, 0] = np.log(total / (classes * width))
        return guess

    @staticmethod
    def _get_components(law: FluxLaw) -> list[object]:
        """Return the flux of every dimension of a flux law."""
        return list(law.flux) if isinstance(law.flux, FreeModuleElement | list | tuple) else [law.flux]

    @staticmethod
    def _get_law_has_operators(law: BaseLaw) -> bool:
        """Check whether the value of a law applies a differential operator.

        The flux of a flux law counts as its value.
        """
        if isinstance(law, FluxLaw):
            return any(FieldLowering._has_operators(SR(item)) for item in FieldLowering._get_components(law))
        return FieldLowering._has_operators(getattr(law, 'value', None))

    @staticmethod
    def _get_volumes(space: object) -> npt.NDArray[np.float64] | float:
        """Return the fraction of the volume of every cell inside a simplex domain.

        Outside a simplex space, the fraction is one.
        """
        return space.volumes_ if isinstance(space, SimplexSpace) else 1.0

    @staticmethod
    def _get_over_positions(
        space: PhaseSpace,
        apply: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return an operator of the position lattice on values over the phase space.

        The values do not vary with class.
        """
        classes = space.size_ // space.position_size_

        def over(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.repeat(apply(np.reshape(values, (space.position_size_, classes))[:, 0]), classes)

        return over

    @staticmethod
    def _get_transported_axes(space: LatticeSpace, name: str) -> tuple[int, ...] | None:
        """Return the axes a spectral operator of a phase space acts along.

        It is `None` for all of them on a lattice.
        """
        if not isinstance(space, PhaseSpace):
            return None
        if name == 'inverse_helmholtz':
            return space.position_axes_
        return tuple(axis for axis in range(space.n_dim_) if space._get_axis_boundary(axis) == 'periodic')

    @staticmethod
    def _tag(offset: IntTuple | int) -> str:
        """Return a name fragment for a lattice offset or a network neighbour slot."""
        if isinstance(offset, int):
            return f'k{offset}'
        return '_'.join(f'{"m" if step < 0 else "p"}{abs(step)}' for step in offset)

    @staticmethod
    def _get_operator_calls(expression: Expression, name: str) -> list[Expression]:
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
    def _get_spectral_apply(
        multiplier: npt.NDArray[np.complex128],
        dimensions: tuple[int, ...],
        axes: tuple[int, ...] | None = None,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that applies a Fourier multiplier to lattice values.

        The lattice is periodic.

        Args:
            multiplier:
                The factor of every real Fourier mode.

            dimensions:
                The lattice shape.

            axes:
                The axes of the transform, or `None` for every axis.

        Returns:
            apply:
                The function of the values at every site.
        """
        transformed_axes = tuple(range(len(dimensions))) if axes is None else axes
        sizes = [dimensions[axis] for axis in transformed_axes]

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            transformed = np.fft.rfftn(np.reshape(values, dimensions), axes=transformed_axes) * multiplier
            return np.fft.irfftn(transformed, s=sizes, axes=transformed_axes).ravel()

        return apply

    @staticmethod
    def _get_whole_sites(offset: list[Expression], spacings: tuple[float, ...]) -> IntTuple | None:
        """Return a displacement in whole sites, or `None`.

        It is `None` unless every component is a number of whole spacings.
        """
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
    def _get_window_apply(
        window: npt.NDArray[np.float64],
        dimensions: tuple[int, ...],
        kind: str,
        fill: float,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that convolves lattice values with a centred window.

        The window holds weights. Windows of at most `SMALL_KERNEL` sites per axis use a direct filter, and larger ones
        an overlap-add FFT of the values padded beyond the edges with the fixed value or their mirror image. Windows act
        along the axes they span.

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
    def _get_interpolated_apply(
        dimensions: tuple[int, ...],
        spacings: tuple[float, ...],
        kind: str,
        fill: float,
        order: int = 1,
    ) -> Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that reads lattice values at displaced positions.

        It reads them by spline interpolation.

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
    def _get_lattice_inverse(
        space: LatticeSpace,
        screening: float,
        kind: str,
        fill: float,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that inverts the lattice stencil `laplacian - screening**2`.

        It inverts it by a fast transform. Periodic edges use real FFTs, fixed edges the sine transform of type 1 with
        the fixed value lifted into the right-hand side, and zero-flux edges the cosine transform of type 2, each with
        the eigenvalues of the stencil. The mode of zero eigenvalue, the mean, is set to zero.

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
            waves = FieldLowering._get_wavenumbers(dimensions, spacings)
            eigenvalues = sum(
                ((2 * np.cos(wave * step) - 2) / step**2 for wave, step in zip(waves, spacings, strict=True)),
                np.zeros(waves[0].shape),
            )
            denominator = eigenvalues - screening**2
            nonzero = np.abs(denominator) > 0
            multiplier = np.where(nonzero, 1.0 / np.where(nonzero, denominator, 1.0), 0.0).astype(complex)
            return FieldLowering._get_spectral_apply(multiplier, dimensions)
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
    def _get_grounded_solver(
        matrix: csr_matrix,
        free: npt.NDArray[np.intp],
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that solves a sparse system on its free rows.

        It is factorised once, and the other rows are zero.
        """
        solve = factorized(matrix[free][:, free].tocsc())
        size = matrix.shape[0]

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            solution = np.zeros(size)
            solution[free] = solve(np.asarray(values, dtype=float)[free])
            return solution

        return apply

    @staticmethod
    def _get_network_inverse(
        space: NetworkSpace,
        screening: float,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that inverts the graph Laplacian minus `screening**2`.

        The solution has zero mean when unscreened.

        Args:
            space:
                The network.

            screening:
                The screening, a non-negative number.

        Returns:
            apply:
                The function of the right-hand side at every node.
        """
        table = space._get_index_table(AdjacencyNeighborhood())
        size = table.shape[0]
        rows = np.repeat(np.arange(size), table.shape[1])
        columns = table.ravel()
        inside = columns < size
        adjacency = csr_matrix((np.ones(inside.sum()), (rows[inside], columns[inside])), shape=(size, size))
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        matrix = csr_matrix(adjacency - diags(degrees) - screening**2 * identity(size))
        if screening > 0:
            return FieldLowering._get_grounded_solver(matrix, np.arange(size))
        solve = FieldLowering._get_grounded_solver(matrix, np.arange(1, size))

        def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            right = np.asarray(values, dtype=float)
            solution = solve(right - right.mean())
            return solution - solution.mean()

        return apply

    @staticmethod
    def _get_mesh_inverse(
        space: MeshSpace,
        screening: float,
        kind: str,
        fill: object,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Return the function that inverts the finite element Laplacian on a mesh.

        The Laplacian has lumped mass and is shifted by minus `screening**2`. It solves `(K + screening**2 M) psi = -M
        f` with the stiffness matrix `K` and the lumped mass `M`, with the fixed value on the boundary nodes of fixed
        edges and zero mean for unscreened zero-flux edges.

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
        stiffness, lumped = FieldLowering._get_mesh_matrices(space)
        size = lumped.size
        matrix = csr_matrix(stiffness + screening**2 * diags(lumped))
        if kind == 'fixed':
            nodes = space.boundary_nodes_
            coordinates = space.positions_[nodes]
            fixed = np.zeros(size)
            fixed[nodes] = np.broadcast_to(fill(*coordinates.T) if callable(fill) else float(str(fill)), nodes.shape)
            free = np.setdiff1d(np.arange(size), nodes)
            solve = FieldLowering._get_grounded_solver(matrix, free)
            carried = matrix @ fixed

            def apply(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return solve(-lumped * np.asarray(values, dtype=float) - carried) + fixed

            return apply
        if screening > 0:
            solve = FieldLowering._get_grounded_solver(matrix, np.arange(size))
            return lambda values: solve(-lumped * np.asarray(values, dtype=float))
        solve = FieldLowering._get_grounded_solver(matrix, np.arange(1, size))

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
                Whether it calls one of `OPERATOR_NAMES`.
        """
        if not isinstance(expression, Expression):
            return False
        names = '|'.join(OPERATOR_NAMES)
        return bool(re.search(rf'\b({names})\(', str(expression)))

    @staticmethod
    @functools.cache
    def _get_cached_mesh_matrices(
        points: bytes,
        cells: bytes,
        shape: tuple[int, int],
        cell_shape: tuple[int, int],
    ) -> tuple[csr_matrix, npt.NDArray[np.float64]]:
        """Return the stiffness matrix and lumped mass of a mesh, cached.

        The cache key is its nodes and cells.
        """
        nodes = np.frombuffer(points).reshape(shape)
        elements = np.frombuffer(cells, dtype=np.int64).reshape(cell_shape)
        space = MeshSpace(nodes, elements)
        basis = Basis(space.prepare().mesh_, ElementTriP1() if shape[0] == PLANE else ElementTetP1())
        stiffness = asm(laplace, basis).tocsr()
        lumped = np.asarray(asm(mass, basis).sum(axis=1)).ravel()
        return stiffness, lumped

    @staticmethod
    def _get_mesh_matrices(space: MeshSpace) -> tuple[csr_matrix, npt.NDArray[np.float64]]:
        """Return the stiffness matrix and lumped mass of the linear elements of a mesh.

        Args:
            space:
                The mesh.

        Returns:
            matrices:
                The stiffness matrix and the lumped mass of every node.
        """
        points = np.ascontiguousarray(np.asarray(space.points, dtype=float))
        cells = np.ascontiguousarray(np.asarray(space.cells, dtype=np.int64))
        return FieldLowering._get_cached_mesh_matrices(points.tobytes(), cells.tobytes(), points.shape, cells.shape)

    @staticmethod
    def _get_mesh_laplacian(space: MeshSpace) -> csr_matrix:
        """Return the discrete Laplacian of linear elements with lumped mass.

        It is `-M^-1 K`, as a sparse matrix.

        Args:
            space:
                The mesh.

        Returns:
            laplacian:
                The sparse matrix that maps nodal values to the nodal Laplacian.
        """
        stiffness, lumped = FieldLowering._get_mesh_matrices(space)
        return -diags(1.0 / lumped) @ stiffness

    @staticmethod
    @functools.cache
    def _get_wavenumbers(
        dimensions: tuple[int, ...],
        spacings: tuple[float, ...],
    ) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the angular wavenumbers of the real Fourier modes of a periodic lattice.

        They are broadcast along each axis.
        """
        axes = [2 * np.pi * np.fft.fftfreq(size, d=step) for size, step in zip(dimensions[:-1], spacings, strict=False)]
        axes.append(2 * np.pi * np.fft.rfftfreq(dimensions[-1], d=spacings[-1]))
        return tuple(np.meshgrid(*axes, indexing='ij'))

    @staticmethod
    @functools.cache
    def _get_dealiasing_mask(dimensions: tuple[int, ...]) -> npt.NDArray[np.bool_]:
        """Return the Fourier modes the two-thirds rule keeps.

        They are below two thirds of the largest along every axis.
        """
        axes = [np.abs(np.fft.fftfreq(size, d=1.0 / size)) < size / 3 for size in dimensions[:-1]]
        axes.append(np.fft.rfftfreq(dimensions[-1], d=1.0 / dimensions[-1]) < dimensions[-1] / 3)
        grids = np.meshgrid(*axes, indexing='ij')
        return np.logical_and.reduce(grids)

    @staticmethod
    def _get_spectral_multiplier(
        name: str,
        axis: int,
        dimensions: tuple[int, ...],
        spacings: tuple[float, ...],
        screening: float = 0.0,
        *,
        axes: tuple[int, ...] | None = None,
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

            axes:
                The axes the operator acts along, whose modes the two-thirds rule alone cuts, or `None` for every axis.

        Returns:
            multiplier:
                The factor of every real Fourier mode.
        """
        waves = FieldLowering._get_wavenumbers(dimensions, spacings)
        squared = np.sum([waves[item] ** 2 for item in (range(len(waves)) if axes is None else axes)], axis=0)
        if name == 'laplacian':
            return -squared.astype(complex)
        if name == 'inverse_helmholtz':
            screened = squared + screening**2
            return np.where(screened > 0, -1.0 / np.where(screened > 0, screened, 1.0), 0.0).astype(complex)
        factor = 1j * waves[axis]
        if dimensions[axis] % 2 == 0:
            nyquist = np.isclose(np.abs(waves[axis]), np.pi / spacings[axis])
            factor = np.where(nyquist, 0.0, factor)
        if axes is None:
            return np.where(FieldLowering._get_dealiasing_mask(dimensions), factor, 0.0)
        return np.where(np.abs(waves[axis]) * spacings[axis] < 2 * np.pi / 3, factor, 0.0)
