"""Lower differential operators on fields to finite-difference stencils and graph Laplacians."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from scipy.sparse import diags

from ...core.elements import FieldElement
from ...core.errors import ValidationError
from ...core.laws import BaseLaw, BoundaryLaw, NoiseLaw, RateLaw, UpdateLaw
from ...core.laws.utils import where
from ...core.spaces import AdjacencyNeighborhood, LatticeNeighborhood, LatticeSpace, MeshSpace, NetworkSpace
from .._utils import domain_of, symbol
from ..problems import ExpressionPrinter
from ..problems._diffusive_limit import DiffusiveLimit
from ..problems._field_operator_kernel import FieldOperatorKernel
from ..problems._law_kernel import LATTICE_BOUNDARIES, MODES
from ..problems._neighbor_read import NeighborRead
from ..problems._read_kind import ReadKind
from ..problems._reference import Reference
from ._base import BaseLowering, BasePass
from ._declarations import Declarations
from ._kernel_spec import KernelSpec

type Indices = npt.NDArray[np.intp]
type Offset = tuple[int, ...]

OPERATOR_NAMES = ('laplacian', 'partial', 'upwind', 'inverse_laplacian')
TIME = symbol('t')


class FieldLowering(BaseLowering):
    """Lowering of the differential operators in the laws of one component to stencils, with the symbols they read.

    `prepare(compilation)` reads the fields and the initial values of the compilation, and `run(laws)` returns the laws
    with every operator replaced by its stencil, noting the stable steps of the Laplacians of rate laws.

    Args:
        spec:
            The kernel spec of the component, whose references receive the neighbour reads.

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
    """

    fields_: Mapping[str, FieldElement]
    initial_: Mapping[Expression, float]
    shifted_: dict[Expression, tuple[Expression, Offset | int]]
    derived_: dict[Expression, tuple[FieldElement, FieldOperatorKernel]]
    limits_: list[float]
    volume_limits_: list[DiffusiveLimit]
    written_: dict[tuple[str, int], list[tuple[Expression, Expression]]]
    tables_: dict[tuple[object, ...], tuple[Indices, float | None]]

    def __init__(self, spec: KernelSpec) -> None:
        self.spec = spec

    def _prepare(self, context: object) -> Self:
        """Read the fields and the initial values of the compilation, with no stencil yet.

        Args:
            context:
                The compilation.

        Returns:
            self:
                The prepared lowering.
        """
        super()._prepare(context)
        declarations = cast('Declarations', self.compilation_.layout)
        self.fields_ = self._field_entries(declarations)
        self.initial_ = self._initial_values(self.spec, declarations)
        self.shifted_, self.derived_, self.tables_ = {}, {}, {}
        self.limits_, self.volume_limits_ = [], []
        self.written_ = defaultdict(list)
        return self

    def _run(self, inputs: object) -> list[BaseLaw]:
        """Return laws whose differential operators are lowered, noting the stable steps of rate laws.

        Args:
            inputs:
                The laws of the component.

        Returns:
            laws:
                The lowered laws.
        """
        result: list[BaseLaw] = []
        for law in cast('list[BaseLaw]', inputs):
            if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw) or not self._has_operators(law.value):
                result.append(law)
                continue
            if isinstance(law, RateLaw):
                self._note_limit(law.value)
            lowered = self._lower(law.value)
            if isinstance(law.value, Expression):
                for reference in KernelSpec._target_references(law._get_targets()[0], self.spec):
                    self.written_[(reference.entry.name, reference.column)].append((lowered, law.value))
            result.append(BasePass._replaced(law, value=lowered))
        return result

    @staticmethod
    def _field_entries(declarations: Declarations) -> dict[str, FieldElement]:
        """Return the fields by the name of each of their entries."""
        return {
            entry.name: block.node.component
            for block in declarations.blocks.values()
            if isinstance(block.node.component, FieldElement)
            for entry in block.entries.values()
        }

    @staticmethod
    def _initial_values(spec: KernelSpec, declarations: Declarations) -> dict[Expression, float]:
        """Return the initial value of every symbol of a component, the first member's for groups."""
        states = np.concatenate([np.zeros(0), *declarations.initial_states])
        parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
        return {
            item: float((states if reference.is_state else parameters)[reference.entry.start + reference.column])
            for item, reference in spec.references.items()
            if reference.entry.start + reference.column < (states if reference.is_state else parameters).size
        }

    def _field_of(self, item: Expression) -> FieldElement | None:
        """Return the field whose member a symbol reads, or `None` for a single value."""
        reference = self.spec.references[item]
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
        offset: Offset | int,
    ) -> tuple[Indices, float | None, int]:
        """Return the neighbour table, the fixed value, and the mode of a symbol at an offset."""
        reference = self.spec.references[item]
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
                dimensions = cast('LatticeSpace', space).shape_
                lattice = LatticeSpace(dimensions, boundary=LATTICE_BOUNDARIES[kind], fill=value).check().prepare()
                table = lattice._index_table(LatticeNeighborhood((offset,)))[:, 0]  # type: ignore[arg-type]
                self.tables_[key] = (table, value if kind == 'fixed' else None)
        table, ghost = self.tables_[key]
        return table, ghost, MODES[kind]

    def _shift(self, expression: Expression, offset: Offset | int) -> Expression:
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
            if base not in self.spec.references:
                continue
            place = self._field_of(base)
            if place is None:
                continue
            if isinstance(offset, int) or isinstance(current, int):
                if current is not None:
                    error_msg = f'A law of {self.spec.owner} nests operators on a network, which only take one step.'
                    raise ValidationError(error_msg)
                total: Offset | int = offset
            else:
                total = tuple(a + b for a, b in zip(current or (0,) * len(offset), offset, strict=True))
            if not isinstance(total, int) and not any(total):
                replacements[item] = base
                continue
            name = f'{base}__{self._tag(total)}'
            target = symbol(name, domain_of(base))
            if target not in self.spec.references:
                reference = self.spec.references[base]
                table, ghost, mode = self._table(base, place, total)
                lattice = isinstance(place.space, LatticeSpace)
                read = NeighborRead(
                    table,
                    ghost,
                    place.space.shape_ if lattice else None,  # type: ignore[union-attr]
                    total if lattice else None,  # type: ignore[arg-type]
                    mode,
                )
                self.spec.references[target] = Reference(
                    reference.entry,
                    reference.is_state,
                    reference.column,
                    ReadKind.NEIGHBOR,
                    read,
                )
                self.shifted_[target] = (base, total)
            replacements[item] = target
        return expression.subs(replacements)

    def _places(self, expression: Expression) -> dict[int, FieldElement]:
        """Return the fields an expression reads, through member symbols, neighbour reads, and derived values."""
        places: dict[int, FieldElement] = {}
        for item in SR(expression).variables():
            if item in self.derived_:
                place = self.derived_[item][0]
            else:
                base = self.shifted_.get(item, (item, None))[0]
                place = self._field_of(base) if base in self.spec.references else None  # type: ignore[assignment]
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
        return cast('LatticeSpace | NetworkSpace', next(iter(spaces.values())))

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

    def _lattice(self, expression: Expression, axis: int, name: str) -> tuple[LatticeSpace, Offset, Offset]:
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
        if not self._has_operators(expression):
            return expression
        call = cast('Expression', expression)
        operator = call.operator()
        operands = [self._lower(item) for item in call.operands()]
        name = getattr(operator, 'name', lambda: '')()
        if name not in OPERATOR_NAMES:
            return operator(*operands)
        place = self._place(SR(operands[0]))
        if place.discretization == 'spectral':
            return self._spectral(name, operands, place)
        if place.discretization == 'finite-element':
            return self._finite_element(name, SR(operands[0]), place)
        return self._stencil(name, operands)

    def _stencil(self, name: str, operands: list[Any]) -> Expression:
        """Return the finite-difference stencil of an operator on a lattice or a network."""
        if name == 'laplacian':
            return self._laplacian(SR(operands[0]))
        if name == 'inverse_laplacian':
            error_msg = f'A law of {self.spec.owner} solves a Poisson equation, which needs a spectral field.'
            raise ValidationError(error_msg)
        axis = int(SR(operands[1]))
        value = SR(operands[0])
        space, step, back = self._lattice(value, axis, 'derivative' if name == 'partial' else 'advection')
        if name == 'partial':
            return (self._shift(value, step) - self._shift(value, back)) / (2 * space.spacings_[axis])
        velocity = SR(operands[2])
        behind = (value - self._shift(value, back)) / space.spacings_[axis]
        ahead = (self._shift(value, step) - value) / space.spacings_[axis]
        return velocity * where(velocity > 0, behind, ahead)

    def _derived(self, place: FieldElement, read: FieldOperatorKernel) -> Expression:
        """Return a new symbol that stands for a derived value of a field, registered with its reference."""
        target = symbol(f'_derived_{place.label}_{len(self.derived_)}')
        entry = next(
            self.spec.references[item].entry
            for item in read.symbols
            if self.spec.references[item].source in {ReadKind.MEMBER, ReadKind.NEIGHBOR, ReadKind.DERIVED}
        )
        self.spec.references[target] = Reference(entry, True, 0, ReadKind.DERIVED, read)
        self.derived_[target] = (place, read)
        return target

    def _spectral(self, name: str, operands: list[Any], place: FieldElement) -> Expression:
        """Return the symbol of a spectral operator applied to an expression, composing chains of operators."""
        space = place.space
        if not isinstance(space, LatticeSpace) or space.boundary != 'periodic':
            error_msg = f'A law of {self.spec.owner} uses a spectral field, which needs a periodic lattice.'
            raise ValidationError(error_msg)
        if name == 'upwind':
            velocity = SR(operands[2])
            return velocity * self._spectral('partial', [operands[0], operands[1]], place)
        axis = int(SR(operands[1])) if name == 'partial' else 0
        if name == 'partial' and not 0 <= axis < space.n_dim_:
            error_msg = f'A law of {self.spec.owner} takes a derivative along axis {axis} of a {space.n_dim_}D lattice.'
            raise ValidationError(error_msg)
        dimensions = space.shape_
        multiplier = self._spectral_multiplier(name, axis, dimensions, space.spacings_)
        operand = SR(operands[0])
        inner = self.derived_.get(operand)
        if inner is not None and inner[1].multiplier is not None:
            operand, multiplier = inner[1].operand, multiplier * inner[1].multiplier
        symbols = sorted((item for item in operand.variables() if item in self.spec.references), key=str)
        read = FieldOperatorKernel(
            operand, symbols, self._spectral_apply(multiplier, dimensions), multiplier, dimensions,
        ).check()
        return self._derived(place, read)

    def _finite_element(self, name: str, operand: Expression, place: FieldElement) -> Expression:
        """Return the symbol of the finite element Laplacian of an expression."""
        if name != 'laplacian':
            error_msg = (
                f'A law of {self.spec.owner} takes the {name} of a finite element field, which takes Laplacians only.'
            )
            raise ValidationError(error_msg)
        matrix = self._mesh_laplacian(place.space)  # type: ignore[arg-type]
        symbols = sorted((item for item in operand.variables() if item in self.spec.references), key=str)
        read = FieldOperatorKernel(operand, symbols, lambda values: matrix @ values, matrix=matrix).check()
        return self._derived(place, read)

    def _note_volume_limit(self, coefficient: Expression, stiffness: float) -> None:
        """Record the diffusive step limit of a finite-volume field as a function of the parameters it reads."""
        symbols = sorted(set(coefficient.variables()) - {TIME}, key=str)
        references = [self.spec.references[item] for item in symbols if item in self.spec.references]
        if len(references) < len(symbols) or any(reference.is_state for reference in references):
            return
        function = ExpressionPrinter(time=TIME, compiled=True).check().prepare(symbols).run([coefficient])
        self.volume_limits_.append(DiffusiveLimit(function, tuple(references), stiffness))

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
            else:
                stiffness = sum(2 / step**2 for step in cast('LatticeSpace', space).spacings_)
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
    def _tag(offset: Offset | int) -> str:
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
    def _spectral_apply(multiplier: npt.NDArray[np.complex128], dimensions: tuple[int, ...]) -> Callable[[Any], Any]:
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
    def _has_operators(expression: object) -> bool:
        """Check whether an expression applies a differential operator.

        Args:
            expression:
                The expression.

        Returns:
            found:
                Whether it calls `laplacian`, `partial`, `upwind`, or `inverse_laplacian`.
        """
        if not isinstance(expression, Expression):
            return False
        return bool(re.search(r'\b(laplacian|partial|upwind|inverse_laplacian)\(', str(expression)))

    @staticmethod
    @functools.cache
    def _cached_mesh_matrices(
        points: bytes,
        cells: bytes,
        shape: tuple[int, int],
        cell_shape: tuple[int, int],
    ) -> Any:  # noqa: ANN401
        """Return the stiffness matrix and the lumped mass of a mesh, cached by its nodes and cells."""
        from skfem import Basis, ElementTetP1, ElementTriP1, asm  # noqa: PLC0415  # only finite elements need scikit-fem
        from skfem.models.poisson import laplace, mass  # noqa: PLC0415

        nodes = np.frombuffer(points).reshape(shape)
        elements = np.frombuffer(cells, dtype=np.int64).reshape(cell_shape)
        space = MeshSpace(nodes, elements)
        basis = Basis(space.check().prepare().mesh_, ElementTriP1() if shape[0] == 2 else ElementTetP1())  # noqa: PLR2004
        stiffness = asm(laplace, basis).tocsr()
        lumped = np.asarray(asm(mass, basis).sum(axis=1)).ravel()
        return stiffness, lumped

    @staticmethod
    def _mesh_matrices(space: MeshSpace) -> tuple[Any, npt.NDArray[np.float64]]:
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
    def _mesh_laplacian(space: MeshSpace) -> Any:  # noqa: ANN401
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
    ) -> npt.NDArray[np.complex128]:
        """Return the Fourier multiplier of an operator on a periodic lattice.

        The Laplacian and its inverse act on every mode, so the Laplacian damps the Nyquist mode by `-|k|^2` like any
        other. A first derivative is zero at the Nyquist mode of its axis, and keeps only the modes of the two-thirds
        rule, which dealiases the derivatives of products.

        Args:
            name:
                `'laplacian'`, `'partial'`, or `'inverse_laplacian'`.

            axis:
                The axis of a derivative.

            dimensions:
                The lattice shape.

            spacings:
                The distance between sites along every axis.

        Returns:
            multiplier:
                The factor of every real Fourier mode.
        """
        waves = FieldLowering._wavenumbers(dimensions, spacings)
        squared = np.sum([wave**2 for wave in waves], axis=0)
        if name == 'laplacian':
            return -squared.astype(complex)
        if name == 'inverse_laplacian':
            with np.errstate(divide='ignore'):
                return np.where(squared > 0, -1.0 / np.where(squared > 0, squared, 1.0), 0.0).astype(complex)
        factor = 1j * waves[axis]
        if dimensions[axis] % 2 == 0:
            nyquist = np.isclose(np.abs(waves[axis]), np.pi / spacings[axis])
            factor = np.where(nyquist, 0.0, factor)
        return np.where(FieldLowering._dealiasing_mask(dimensions), factor, 0.0)
