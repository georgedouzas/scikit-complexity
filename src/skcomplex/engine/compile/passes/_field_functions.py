"""Pass that writes operators back into the equations and adds flux differences, symmetries, and fixed nodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, solve
from sage.modules.free_module_element import FreeModuleElement

from ....core.elements import FieldElement
from ....core.errors import ValidationError
from ....core.laws import BaseLaw, FluxLaw
from ....core.spaces import LatticeSpace, MeshSpace
from ....core.types import Vector
from ...problems import (
    Layout,
    Reference,
    Source,
    VariableKind,
    declarations_of,
    numpy_function,
)
from .._block import Block
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._field_volumes import TIME, FiniteVolume
from .._kernel_spec import KernelSpec
from .._pipeline import BasePass


class FieldFunctionsPass(BasePass):
    """Pass that writes operators back into the equations and adds flux differences, symmetries, and fixed nodes."""

    reads = ('functions', 'written', 'reduced', 'layout')
    writes = ('functions', 'volumes', 'symmetries', 'constrained')

    def _run(self, compilation: Compilation) -> None:
        """Add the discretizations of the fields to the numeric functions.

        Args:
            compilation:
                The compilation state.
        """
        functions = cast('dict[str, Any]', compilation.functions)
        declarations = cast('Declarations', compilation.layout)
        states = Layout(declarations.states)
        functions['equations'] = self._written_equations(functions['equations'], states, compilation.facts['written'])
        blocks = list(declarations.blocks.values())
        volumes, constrained = self._with_discretizations(
            functions,
            cast('Laws', compilation.reduced),
            declarations,
            blocks,
        )
        compilation.facts['volumes'] = volumes
        compilation.facts['symmetries'] = tuple(self._symmetries(blocks))
        compilation.facts['constrained'] = constrained

    @staticmethod
    def _written_equations(
        equations: dict[str, Expression],
        states: Layout,
        written: dict[tuple[str, int], list[tuple[Expression, Expression]]],
    ) -> dict[str, Expression]:
        """Return equations with the stencils of operators replaced by the operators as written."""
        result = dict(equations)
        for (name, column), pairs in written.items():
            key = states.entry(name).generic_names[column]
            if key in result:
                result[key] = SR(result[key]) - sum(SR(item) for item, _ in pairs) + sum(SR(item) for _, item in pairs)
        return result

    @staticmethod
    def _compile_fluxes(
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
        declarations: Declarations,
    ) -> tuple[FiniteVolume, ...]:
        """Return the conservation laws of every finite-volume field from the flux laws of its component."""
        volumes = []
        for spec, laws in specs:
            fluxes = [law for law in laws if isinstance(law, FluxLaw)]
            if not fluxes:
                continue
            targets = [(law, KernelSpec._target_references(law._get_targets()[0], spec)[0]) for law in fluxes]
            owners = {reference.entry.name.rpartition('.')[0] for _, reference in targets}
            for owner in sorted(owners):
                block = declarations.blocks[owner]
                place = block.node.component
                if not isinstance(place, FieldElement):
                    error_msg = f'{spec.owner} declares fluxes of {owner}, which is not a field.'
                    raise ValidationError(error_msg)
                mine = [(law, reference) for law, reference in targets if reference.entry.name.startswith(f'{owner}.')]
                volumes.append(FieldFunctionsPass._compile_volumes(mine, spec.references, place, owner))
        return tuple(volumes)

    @staticmethod
    def _mesh_constraints(blocks: list[Block]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
        """Return the state indices and values of the boundary nodes of finite element fields with fixed boundaries."""
        indices: list[npt.NDArray[np.intp]] = []
        values: list[npt.NDArray[np.float64]] = []
        for block in blocks:
            place = block.node.component
            if not isinstance(place, FieldElement) or not isinstance(place.space, MeshSpace):
                continue
            if place.discretization != 'finite-element':
                error_msg = f'Field {block.node.path} lives on a mesh, which needs the finite-element discretization.'
                raise ValidationError(error_msg)
            nodes = place.space.boundary_nodes_
            coordinates = place.space.positions_[nodes]
            for variable, entry in block.entries.items():
                if variable not in declarations_of(place.element, VariableKind.STATE):
                    continue
                kind, value = place._get_boundary(variable)
                if kind != 'fixed':
                    continue
                fixed = value(*coordinates.T) if callable(value) else value
                indices.append(entry.start + nodes * entry.columns)
                values.append(np.broadcast_to(np.asarray(fixed, dtype=float), nodes.shape).copy())
        if not indices:
            return np.zeros(0, dtype=np.intp), np.zeros(0)
        return np.concatenate(indices).astype(np.intp), np.concatenate(values)

    @staticmethod
    def _symmetries(blocks: list[Block]) -> list[tuple[npt.NDArray[np.intp], tuple[int, ...], list[tuple[int, int]]]]:
        """Return the state indices, the lattice shape, and the axes with their signs of every symmetric field state."""
        found = []
        for block in blocks:
            place = block.node.component
            if not isinstance(place, FieldElement) or not place.symmetry:
                continue
            space = place.space
            if not isinstance(space, LatticeSpace) or space.boundary != 'periodic':
                error_msg = f'Field {block.node.path} declares symmetries, which need a periodic lattice.'
                raise ValidationError(error_msg)
            for variable, parities in place.symmetry.items():
                entry = block.entries.get(variable)
                if entry is None or len(parities) != space.n_dim_:
                    error_msg = f'The symmetry of {block.node.path}.{variable} needs a state and one parity per axis.'
                    raise ValidationError(error_msg)
                signs = []
                for axis, parity in enumerate(parities):
                    if parity not in {'odd', 'even', None}:
                        error_msg = (
                            f"The parity {parity!r} of {block.node.path}.{variable} is not 'odd', 'even', or None."
                        )
                        raise ValidationError(error_msg)
                    if parity is not None:
                        signs.append((axis, -1 if parity == 'odd' else 1))
                indices = entry.start + np.arange(place.size_) * entry.columns
                found.append((indices, space.shape_, signs))
        return found

    @staticmethod
    def _with_symmetries(drift: Any, symmetries: list[Any]) -> Any:  # noqa: ANN401
        """Return the drift with the rates of symmetric states projected onto their odd or even parts."""
        if not symmetries or drift is None:
            return drift

        def symmetric(t: float, x: Vector, p: Vector) -> Vector:
            rates = np.array(drift(t, x, p), dtype=float)
            for indices, dimensions, signs in symmetries:
                values = np.reshape(rates[indices], dimensions)
                for axis, sign in signs:
                    mirrored = np.roll(np.flip(values, axis=axis), 1, axis=axis)
                    values = (values + sign * mirrored) / 2
                rates[indices] = values.ravel()
            return rates

        return symmetric

    @staticmethod
    def _with_constraints(drift: Any, indices: npt.NDArray[np.intp]) -> Any:  # noqa: ANN401
        """Return the drift with the rates of fixed boundary nodes held at zero."""
        if not indices.size or drift is None:
            return drift

        def constrained(t: float, x: Vector, p: Vector) -> Vector:
            rates = np.array(drift(t, x, p), dtype=float)
            rates[indices] = 0.0
            return rates

        return constrained

    @staticmethod
    def _with_volumes(drift: Any, volumes: tuple[FiniteVolume, ...]) -> Any:  # noqa: ANN401
        """Return the drift with the flux differences of finite-volume fields added."""
        if not volumes or drift is None:
            return drift

        def conserved(t: float, x: Vector, p: Vector) -> Vector:
            rates = drift(t, x, p)
            for volume in volumes:
                rates = rates + volume.rates(x, p)
            return rates

        return conserved

    @staticmethod
    def _with_discretizations(
        functions: dict[str, Any],
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
        declarations: Declarations,
        blocks: list[Block],
    ) -> tuple[tuple[FiniteVolume, ...], tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]]:
        """Add flux differences, symmetries, and fixed boundary nodes to the drift, returning laws and fixed nodes."""
        volumes = FieldFunctionsPass._compile_fluxes(specs, declarations)
        symmetries = FieldFunctionsPass._symmetries(blocks)
        constrained, fixed = FieldFunctionsPass._mesh_constraints(blocks)
        drift = FieldFunctionsPass._with_volumes(functions.get('drift'), volumes)
        drift = FieldFunctionsPass._with_symmetries(drift, symmetries)
        drift = FieldFunctionsPass._with_constraints(drift, constrained)
        if volumes or symmetries or constrained.size:
            functions['drift'] = drift
            functions['accelerated'] = None
        return volumes, (constrained, fixed)

    @staticmethod
    def _symbol(target: Expression, references: Mapping[Expression, Reference]) -> Expression:
        """Return the member symbol of a flux target after checking it is a member state."""
        reference = references.get(target)
        if reference is None or not reference.is_state or reference.source is not Source.MEMBER:
            error_msg = f'A flux targets {target}, which is not a state of a field.'
            raise ValidationError(error_msg)
        return target

    @staticmethod
    def _compile_volumes(
        laws: Sequence[tuple[FluxLaw, Reference]],
        references: Mapping[Expression, Reference],
        place: FieldElement,
        owner: str,
    ) -> FiniteVolume:
        """Return the conservation laws of one finite-volume field.

        Args:
            laws:
                The flux laws of the field with the references of their targets.

            references:
                The references of the symbols of the field's namespace.

            place:
                The field.

            owner:
                The path of the field.

        Returns:
            volumes:
                The compiled conservation laws.

        Raises:
            ValidationError:
                The field is not a finite-volume field on a lattice, or the fluxes are not one per dimension.
        """
        space = place.space
        if place.discretization != 'finite-volume' or not isinstance(space, LatticeSpace):
            error_msg = f'{owner} declares fluxes, which need a finite-volume field on a lattice.'
            raise ValidationError(error_msg)
        states = [FieldFunctionsPass._symbol(law._get_targets()[0], references) for law, _ in laws]
        n_dim = space.n_dim_
        expressions = []
        for law, _ in laws:
            components = list(law.flux) if isinstance(law.flux, FreeModuleElement | list | tuple) else [law.flux]
            if len(components) != n_dim:
                error_msg = f'A flux of {owner} has {len(components)} components for a {n_dim}D lattice.'
                raise ValidationError(error_msg)
            expressions.append([SR(item) for item in components])
        parameters = sorted(
            {
                item
                for row in expressions
                for value in row
                for item in value.variables()
                if item not in states and item in references
            },
            key=str,
        )
        unknown = {
            item for row in expressions for value in row for item in value.variables() if item not in states
        } - set(
            parameters,
        )
        if unknown:
            error_msg = (
                f'The fluxes of {owner} read {", ".join(map(str, unknown))}, which are not states or parameters.'
            )
            raise ValidationError(error_msg)
        columns = list(zip(*expressions, strict=True))
        fluxes = tuple(numpy_function(TIME, [*states, *parameters], list(column)) for column in columns)
        jacobians = tuple(
            numpy_function(TIME, [*states, *parameters], [value.diff(state) for value in column for state in states])
            for column in columns
        )
        first = laws[0][0]
        critical: tuple[Callable[..., Any], ...] = ()
        if first.riemann == 'godunov':
            if len(states) != 1:
                error_msg = f'The fluxes of {owner} ask for the Godunov flux, which is exact for one state only.'
                raise ValidationError(error_msg)
            roots = []
            for column in columns:
                found = [item.rhs() for item in solve(column[0].diff(states[0]) == 0, states[0])]
                roots.append(numpy_function(TIME, parameters, found) if found else numpy_function(TIME, parameters, []))
            critical = tuple(roots)
        boundaries = tuple(
            (kind, float(value) if not callable(value) else 0.0)
            for kind, value in (place._get_boundary(reference.entry.name.rpartition('.')[2]) for _, reference in laws)
        )
        return FiniteVolume(
            owner,
            tuple(reference.entry.start for _, reference in laws),
            space.shape_,
            space.spacings_,
            boundaries,
            fluxes,
            jacobians,
            tuple(references[item] for item in parameters),
            first.limiter,
            first.riemann,
            critical,
        )
