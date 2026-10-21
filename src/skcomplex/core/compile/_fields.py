"""Lower differential operators on fields to finite-difference stencils and graph Laplacians."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR
from scipy.sparse import diags

from ..base import Expression, ValidationError, domain_of, symbol
from ..model import Adjacency, Field, LatticeSpace, Mesh, Neighborhood, NetworkSpace, where
from ._kernels import Reference, Source
from ._printing import numpy_function

type Indices = npt.NDArray[np.intp]
type Offset = tuple[int, ...]

MODES = {'periodic': 0, 'zero-flux': 1, 'fixed': 2}
OPERATOR_NAMES = ('laplacian', 'partial', 'upwind', 'inverse_laplacian')
LATTICE_BOUNDARIES = {'periodic': 'periodic', 'zero-flux': 'reflecting', 'fixed': 'fixed'}
TIME = symbol('t')


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


@functools.cache
def wavenumbers(dimensions: tuple[int, ...], spacings: tuple[float, ...]) -> tuple[npt.NDArray[np.float64], ...]:
    """Return the angular wavenumbers of the real Fourier modes of a periodic lattice, broadcast along every axis.

    Args:
        dimensions:
            The lattice shape.

        spacings:
            The distance between sites along every axis.

    Returns:
        wavenumbers:
            One array per axis, the last axis holding the non-negative frequencies of a real transform.
    """
    axes = [2 * np.pi * np.fft.fftfreq(size, d=step) for size, step in zip(dimensions[:-1], spacings, strict=False)]
    axes.append(2 * np.pi * np.fft.rfftfreq(dimensions[-1], d=spacings[-1]))
    return tuple(np.meshgrid(*axes, indexing='ij'))


@functools.cache
def dealiasing_mask(dimensions: tuple[int, ...]) -> npt.NDArray[np.bool_]:
    """Return the Fourier modes the two-thirds rule keeps, those below two thirds of the largest along every axis.

    Args:
        dimensions:
            The lattice shape.

    Returns:
        mask:
            Whether every real Fourier mode is kept.
    """
    axes = [np.abs(np.fft.fftfreq(size, d=1.0 / size)) < size / 3 for size in dimensions[:-1]]
    axes.append(np.fft.rfftfreq(dimensions[-1], d=1.0 / dimensions[-1]) < dimensions[-1] / 3)
    grids = np.meshgrid(*axes, indexing='ij')
    return np.logical_and.reduce(grids)


def spectral_multiplier(
    name: str,
    axis: int,
    dimensions: tuple[int, ...],
    spacings: tuple[float, ...],
) -> npt.NDArray[np.complex128]:
    """Return the Fourier multiplier of an operator on a periodic lattice.

    The Laplacian and its inverse act on every mode, so the Laplacian damps the Nyquist mode by `-|k|^2` like any other.
    A first derivative is zero at the Nyquist mode of its axis, and keeps only the modes of the two-thirds rule, which
    dealiases the derivatives of products.

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
    waves = wavenumbers(dimensions, spacings)
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
    return np.where(dealiasing_mask(dimensions), factor, 0.0)


@dataclass(eq=False)
class DerivedRead:
    """Value of a global operator, a spectral multiplier or a finite element matrix, applied to an expression.

    Args:
        operand:
            The expression of member symbols the operator acts on.

        symbols:
            The symbols of the operand in the order its function takes them.

        apply:
            The operator on the operand's values at every site.

        multiplier:
            The Fourier multiplier of a spectral operator, or `None`.

        dimensions:
            The lattice shape of a spectral operator, or `None`.

        matrix:
            The sparse matrix of an operator that acts on the sites as one, such as the finite element Laplacian, or
            `None` for spectral operators.
    """

    operand: Expression
    symbols: list[Expression]
    apply: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    multiplier: npt.NDArray[np.complex128] | None = None
    dimensions: tuple[int, ...] | None = None
    matrix: Any = None
    _function: Callable[..., Any] | None = field(default=None, repr=False)

    def evaluate(self, t: float, values: list[Any], size: int) -> npt.NDArray[np.float64]:
        """Return the operator applied to the operand evaluated from the values of its symbols.

        Args:
            t:
                The time.

            values:
                The values of the operand's symbols.

            size:
                The number of sites.

        Returns:
            values:
                The operator's value at every site.
        """
        if self._function is None:
            self._function = numpy_function(symbol('t'), self.symbols, [self.operand])
        operand = np.broadcast_to(np.asarray(self._function(t, *values)[0], dtype=float), (size,))
        return self.apply(operand)


def spectral_apply(multiplier: npt.NDArray[np.complex128], dimensions: tuple[int, ...]) -> Callable[[Any], Any]:
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


def _tag(offset: Offset | int) -> str:
    """Return a name fragment for a lattice offset or a network neighbour slot."""
    if isinstance(offset, int):
        return f'k{offset}'
    return '_'.join(f'{"m" if step < 0 else "p"}{abs(step)}' for step in offset)


@dataclass(frozen=True, eq=False)
class DiffusiveLimit:
    """Stability limit of explicit steps for the diffusion of a finite-volume field, at the current parameters.

    Args:
        coefficient:
            The NumPy function of time and parameters that returns the diffusion coefficient.

        parameters:
            The references of the parameters the coefficient reads, in argument order.

        stiffness:
            The largest eigenvalue magnitude of the unit Laplacian, `sum(2 / h^2)` over the axes.
    """

    coefficient: Callable[..., list[Any]]
    parameters: tuple[Reference, ...]
    stiffness: float

    def step(self, p: npt.NDArray[np.float64]) -> float:
        """Return the largest stable step of explicit Euler for the diffusion at parameter values.

        Args:
            p:
                The parameter vector.

        Returns:
            step:
                The step `1 / (stiffness * max |D|)`, infinite without diffusion.
        """
        values = [reference.column_of(np.zeros(0), p) for reference in self.parameters]
        magnitude = float(np.max(np.abs(self.coefficient(0.0, *values)[0])))
        return math.inf if magnitude == 0 else 1 / (self.stiffness * magnitude)


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
    derived: dict[Expression, tuple[Field, DerivedRead]] = field(default_factory=dict)
    limits: list[float] = field(default_factory=list)
    volume_limits: list[DiffusiveLimit] = field(default_factory=list)
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
                table = space.index_table(Adjacency())
                sites = np.arange(table.shape[0])[:, None]
                self._tables[key] = (np.where(table == table.shape[0], sites, table)[:, offset], None)
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

    def _places(self, expression: Expression) -> dict[int, Field]:
        """Return the fields an expression reads, through member symbols, neighbour reads, and derived values."""
        places: dict[int, Field] = {}
        for item in SR(expression).variables():
            if item in self.derived:
                place = self.derived[item][0]
            else:
                base = self.shifted.get(item, (item, None))[0]
                place = self._field_of(base) if base in self.references else None  # type: ignore[assignment]
            if place is not None:
                places[id(place)] = place
        return places

    def _place(self, expression: Expression) -> Field:
        """Return the one field an expression reads, or raise an error."""
        places = self._places(expression)
        discretizations = {place.discretization for place in places.values()}
        spaces = {id(place.space) for place in places.values()}
        if len(spaces) != 1 or len(discretizations) != 1:
            error_msg = (
                f'A law of {self.owner} applies an operator to {expression}, which reads {len(spaces)} fields. An '
                'operator acts on the states of fields of one space and one discretization.'
            )
            raise ValidationError(error_msg)
        return next(iter(places.values()))

    def _space(self, expression: Expression) -> LatticeSpace | NetworkSpace:
        """Return the space of the fields an expression reads."""
        spaces = {id(place.space): place.space for place in self._places(expression).values()}
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
            width = space.index_table(Adjacency()).shape[1]
            return sum((self.shift(expression, slot) - expression for slot in range(width)), SR(0))
        total = SR(0)
        for axis in range(space.n_dim):
            step = tuple(int(axis == other) for other in range(space.n_dim))
            back = tuple(-item for item in step)
            total += (self.shift(expression, step) + self.shift(expression, back) - 2 * expression) / (
                space.spacings[axis] ** 2
            )
        return total

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
            error_msg = f'A law of {self.owner} solves a Poisson equation, which needs a spectral field.'
            raise ValidationError(error_msg)
        axis = int(SR(operands[1]))
        value = SR(operands[0])
        space, step, back = self._lattice(value, axis, 'derivative' if name == 'partial' else 'advection')
        if name == 'partial':
            return (self.shift(value, step) - self.shift(value, back)) / (2 * space.spacings[axis])
        velocity = SR(operands[2])
        behind = (value - self.shift(value, back)) / space.spacings[axis]
        ahead = (self.shift(value, step) - value) / space.spacings[axis]
        return velocity * where(velocity > 0, behind, ahead)

    def _derived(self, place: Field, read: DerivedRead) -> Expression:
        """Return a new symbol that stands for a derived value of a field, registered with its reference."""
        target = symbol(f'_derived_{place.label}_{len(self.derived)}')
        entry = next(
            self.references[item].entry
            for item in read.symbols
            if self.references[item].source in {Source.MEMBER, Source.NEIGHBOR, Source.DERIVED}
        )
        self.references[target] = Reference(entry, True, 0, Source.DERIVED, read)
        self.derived[target] = (place, read)
        return target

    def _spectral(self, name: str, operands: list[Any], place: Field) -> Expression:
        """Return the symbol of a spectral operator applied to an expression, composing chains of operators."""
        space = place.space
        if not isinstance(space, LatticeSpace) or space.boundary != 'periodic':
            error_msg = f'A law of {self.owner} uses a spectral field, which needs a periodic lattice.'
            raise ValidationError(error_msg)
        if name == 'upwind':
            velocity = SR(operands[2])
            return velocity * self._spectral('partial', [operands[0], operands[1]], place)
        axis = int(SR(operands[1])) if name == 'partial' else 0
        if name == 'partial' and not 0 <= axis < space.n_dim:
            error_msg = f'A law of {self.owner} takes a derivative along axis {axis} of a {space.n_dim}D lattice.'
            raise ValidationError(error_msg)
        dimensions = space.dimensions
        multiplier = spectral_multiplier(name, axis, dimensions, space.spacings)
        operand = SR(operands[0])
        inner = self.derived.get(operand)
        if inner is not None and inner[1].multiplier is not None:
            operand, multiplier = inner[1].operand, multiplier * inner[1].multiplier
        symbols = sorted((item for item in operand.variables() if item in self.references), key=str)
        read = DerivedRead(operand, symbols, spectral_apply(multiplier, dimensions), multiplier, dimensions)
        return self._derived(place, read)

    def _finite_element(self, name: str, operand: Expression, place: Field) -> Expression:
        """Return the symbol of the finite element Laplacian of an expression."""
        if name != 'laplacian':
            error_msg = (
                f'A law of {self.owner} takes the {name} of a finite element field, which takes Laplacians only.'
            )
            raise ValidationError(error_msg)
        matrix = mesh_laplacian(place.space)  # type: ignore[arg-type]
        symbols = sorted((item for item in operand.variables() if item in self.references), key=str)
        read = DerivedRead(operand, symbols, lambda values: matrix @ values, matrix=matrix)
        return self._derived(place, read)

    def _note_volume_limit(self, coefficient: Expression, stiffness: float) -> None:
        """Record the diffusive step limit of a finite-volume field as a function of the parameters it reads."""
        symbols = sorted(set(coefficient.variables()) - {TIME}, key=str)
        references = [self.references[item] for item in symbols if item in self.references]
        if len(references) < len(symbols) or any(reference.is_state for reference in references):
            return
        function = numpy_function(TIME, symbols, [coefficient])
        self.volume_limits.append(DiffusiveLimit(function, tuple(references), stiffness))

    def note_limit(self, value: object) -> None:
        """Record the diffusive step limit of the Laplacians in a law value.

        A finite-difference field records the limit when the coefficient is a number at the initial values, and a
        finite-volume field records it as a function of the parameters the coefficient reads.

        Args:
            value:
                The value of a rate law before lowering.
        """
        if not isinstance(value, Expression):
            return
        for call in _operator_calls(value, 'laplacian'):
            placeholder = symbol('_laplacian_')
            expression = SR(value.subs({call: placeholder})).diff(placeholder)
            place = self._place(call.operands()[0])
            if place.discretization not in {'finite-difference', 'finite-volume'}:
                continue
            space = place.space
            if isinstance(space, NetworkSpace):
                stiffness = max((item for _, item in space.graph.degree), default=1)
            else:
                stiffness = sum(2 / step**2 for step in cast('LatticeSpace', space).spacings)
            if place.discretization == 'finite-volume':
                self._note_volume_limit(expression, stiffness)
                continue
            coefficient = expression.subs(self.initial)
            if coefficient.variables():
                continue
            magnitude = abs(float(coefficient))
            if magnitude == 0:
                continue
            self.limits.append(1 / (stiffness * magnitude))


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
            Whether it calls `laplacian`, `partial`, `upwind`, or `inverse_laplacian`.
    """
    if not isinstance(expression, Expression):
        return False
    return bool(re.search(r'\b(laplacian|partial|upwind|inverse_laplacian)\(', str(expression)))


@functools.cache
def _mesh_matrices(
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
    space = Mesh(nodes, elements)
    basis = Basis(space.skfem_mesh(), ElementTriP1() if shape[0] == 2 else ElementTetP1())  # noqa: PLR2004
    stiffness = asm(laplace, basis).tocsr()
    lumped = np.asarray(asm(mass, basis).sum(axis=1)).ravel()
    return stiffness, lumped


def mesh_matrices(space: Mesh) -> tuple[Any, npt.NDArray[np.float64]]:
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
    return _mesh_matrices(points.tobytes(), cells.tobytes(), points.shape, cells.shape)


def mesh_laplacian(space: Mesh) -> Any:  # noqa: ANN401
    """Return the discrete Laplacian of linear elements with lumped mass, `-M^-1 K`, as a sparse matrix.

    Args:
        space:
            The mesh.

    Returns:
        laplacian:
            The sparse matrix that maps nodal values to the nodal Laplacian.
    """
    stiffness, lumped = mesh_matrices(space)
    return -diags(1.0 / lumped) @ stiffness
