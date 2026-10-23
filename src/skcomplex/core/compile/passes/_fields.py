"""Compile fields: differential operators, finite volumes, finite elements, symmetries, and fixed boundaries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ...base import Expression, ValidationError, VariableKind, declarations_of
from ...components import Field
from ...laws import Boundary, Flux, Law, Noise, Rate, Update
from ...space import LatticeSpace, Mesh
from .._compilation import Compilation, Laws
from .._declarations import Block, Declarations
from .._fields import FieldLowering, has_operators
from .._kernels import Source
from .._layout import Layout
from .._pipeline import Pass
from .._problem import Problem, Vector
from .._specs import KernelSpec, target_references
from .._splitting import LinearTerm, linear_terms
from .._volumes import FiniteVolume, compile_volumes


def _initial_values(spec: KernelSpec, declarations: Declarations) -> dict[Expression, float]:
    """Return the initial value of every symbol of a component, the first member's for groups."""
    states = np.concatenate([np.zeros(0), *declarations.initial_states])
    parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
    return {
        item: float((states if reference.is_state else parameters)[reference.entry.start + reference.column])
        for item, reference in spec.references.items()
        if reference.entry.start + reference.column < (states if reference.is_state else parameters).size
    }


def _linear_terms_of(spec: KernelSpec, laws: list[Law], places: dict[str, Field]) -> list[LinearTerm]:
    """Return the linear terms with constant coefficients of the rates of field states."""
    terms = []
    for law in laws:
        if not isinstance(law, Rate) or not isinstance(law.value, Expression):
            continue
        for reference in target_references(law.target, spec):
            if reference.entry.name in places and reference.source is Source.MEMBER:
                terms += linear_terms(law.value, reference, spec.references, places)
    return terms


def _with_fields(spec: KernelSpec, laws: list[Law], declarations: Declarations, compilation: Compilation) -> list[Law]:
    """Return laws whose differential operators are lowered, noting stable steps and the linear terms of rates."""
    places = declarations.places
    compilation.linear += _linear_terms_of(spec, laws, places)
    if not any(has_operators(getattr(law, 'value', None)) for law in laws):
        return laws
    lowering = FieldLowering(spec.owner, spec.references, places, _initial_values(spec, declarations))
    result: list[Law] = []
    for law in laws:
        if not isinstance(law, Rate | Noise | Update | Boundary) or not has_operators(law.value):
            result.append(law)
            continue
        if isinstance(law, Rate):
            lowering.note_limit(law.value)
        lowered = lowering.lower(law.value)
        if isinstance(law.value, Expression):
            for reference in target_references(law.target, spec):
                compilation.written[(reference.entry.name, reference.column)].append((lowered, law.value))
        if isinstance(law, Rate | Noise):
            result.append(replace(law, value=lowered))
        else:
            result.append(type(law)(law.target, lowered))
    compilation.limits += lowering.limits
    compilation.volume_limits += lowering.volume_limits
    return result


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


def _compile_fluxes(
    specs: list[tuple[KernelSpec, list[Law]]],
    declarations: Declarations,
) -> tuple[FiniteVolume, ...]:
    """Return the conservation laws of every finite-volume field from the flux laws of its component."""
    volumes = []
    for spec, laws in specs:
        fluxes = [law for law in laws if isinstance(law, Flux)]
        if not fluxes:
            continue
        targets = [(law, target_references(law.target, spec)[0]) for law in fluxes]
        owners = {reference.entry.name.rpartition('.')[0] for _, reference in targets}
        for owner in sorted(owners):
            block = declarations.blocks[owner]
            place = block.node.component
            if not isinstance(place, Field):
                error_msg = f'{spec.owner} declares fluxes of {owner}, which is not a field.'
                raise ValidationError(error_msg)
            mine = [(law, reference) for law, reference in targets if reference.entry.name.startswith(f'{owner}.')]
            volumes.append(compile_volumes(mine, spec.references, place, owner))
    return tuple(volumes)


def _mesh_constraints(blocks: list[Block]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Return the state indices and values of the boundary nodes of finite element fields with fixed boundaries."""
    indices: list[npt.NDArray[np.intp]] = []
    values: list[npt.NDArray[np.float64]] = []
    for block in blocks:
        place = block.node.component
        if not isinstance(place, Field) or not isinstance(place.space, Mesh):
            continue
        if place.discretization != 'finite-element':
            error_msg = f'Field {block.node.path} lives on a mesh, which needs the finite-element discretization.'
            raise ValidationError(error_msg)
        nodes = place.space.boundary_nodes()
        coordinates = place.space.nodes[nodes]
        for variable, entry in block.entries.items():
            if variable not in declarations_of(place.element, VariableKind.STATE):
                continue
            kind, value = place.boundary_of(variable)
            if kind != 'fixed':
                continue
            fixed = value(*coordinates.T) if callable(value) else value
            indices.append(entry.start + nodes * entry.columns)
            values.append(np.broadcast_to(np.asarray(fixed, dtype=float), nodes.shape).copy())
    if not indices:
        return np.zeros(0, dtype=np.intp), np.zeros(0)
    return np.concatenate(indices).astype(np.intp), np.concatenate(values)


def _symmetries(blocks: list[Block]) -> list[tuple[npt.NDArray[np.intp], tuple[int, ...], list[tuple[int, int]]]]:
    """Return the state indices, the lattice shape, and the axes with their signs of every symmetric field state."""
    found = []
    for block in blocks:
        place = block.node.component
        if not isinstance(place, Field) or not place.symmetry:
            continue
        space = place.space
        if not isinstance(space, LatticeSpace) or space.boundary != 'periodic':
            error_msg = f'Field {block.node.path} declares symmetries, which need a periodic lattice.'
            raise ValidationError(error_msg)
        for variable, parities in place.symmetry.items():
            entry = block.entries.get(variable)
            if entry is None or len(parities) != space.n_dim:
                error_msg = f'The symmetry of {block.node.path}.{variable} needs a state and one parity per axis.'
                raise ValidationError(error_msg)
            signs = []
            for axis, parity in enumerate(parities):
                if parity not in {'odd', 'even', None}:
                    error_msg = f"The parity {parity!r} of {block.node.path}.{variable} is not 'odd', 'even', or None."
                    raise ValidationError(error_msg)
                if parity is not None:
                    signs.append((axis, -1 if parity == 'odd' else 1))
            indices = entry.start + np.arange(place.size or 0) * entry.columns
            found.append((indices, space.dimensions, signs))
    return found


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


def _with_constraints(drift: Any, indices: npt.NDArray[np.intp]) -> Any:  # noqa: ANN401
    """Return the drift with the rates of fixed boundary nodes held at zero."""
    if not indices.size or drift is None:
        return drift

    def constrained(t: float, x: Vector, p: Vector) -> Vector:
        rates = np.array(drift(t, x, p), dtype=float)
        rates[indices] = 0.0
        return rates

    return constrained


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


def _with_discretizations(
    functions: dict[str, Any],
    specs: list[tuple[KernelSpec, list[Law]]],
    declarations: Declarations,
    blocks: list[Block],
) -> tuple[tuple[FiniteVolume, ...], tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]]:
    """Add flux differences, symmetries, and fixed boundary nodes to the drift, returning the laws and fixed nodes."""
    volumes = _compile_fluxes(specs, declarations)
    symmetries = _symmetries(blocks)
    constrained, fixed = _mesh_constraints(blocks)
    drift = _with_volumes(functions.get('drift'), volumes)
    drift = _with_symmetries(drift, symmetries)
    drift = _with_constraints(drift, constrained)
    if volumes or symmetries or constrained.size:
        functions['drift'] = drift
        functions['accelerated'] = None
    return volumes, (constrained, fixed)


class LowerFieldsPass(Pass):
    """Pass that lowers the differential operators of laws to stencils and notes stable steps and linear terms."""

    reads = ('corrected', 'layout')
    writes = ('corrected', 'written', 'limits', 'volume_limits', 'linear')

    def run(self, compilation: Compilation) -> None:
        """Lower the differential operators of the corrected laws.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        compilation.corrected = [
            (spec, _with_fields(spec, laws, declarations, compilation))
            for spec, laws in cast('Laws', compilation.corrected)
        ]


class FieldFunctionsPass(Pass):
    """Pass that writes operators back into the equations and adds flux differences, symmetries, and fixed nodes."""

    reads = ('functions', 'written', 'reduced', 'layout')
    writes = ('functions', 'volumes', 'symmetries', 'constrained')

    def run(self, compilation: Compilation) -> None:
        """Add the discretizations of the fields to the numeric functions.

        Args:
            compilation:
                The compilation state.
        """
        functions = cast('dict[str, Any]', compilation.functions)
        declarations = cast('Declarations', compilation.layout)
        states = Layout(declarations.states)
        functions['equations'] = _written_equations(functions['equations'], states, compilation.written)
        blocks = list(declarations.blocks.values())
        volumes, constrained = _with_discretizations(functions, cast('Laws', compilation.reduced), declarations, blocks)
        compilation.facts['volumes'] = volumes
        compilation.facts['symmetries'] = tuple(_symmetries(blocks))
        compilation.facts['constrained'] = constrained


class FieldJacobianPass(Pass):
    """Pass that adds the conservation laws, symmetries, and fixed nodes of the fields to the exact Jacobian."""

    reads = ('jacobian', 'volumes', 'symmetries', 'constrained')
    writes = ('jacobian',)

    def run(self, compilation: Compilation) -> None:
        """Add the discretizations of the fields to the Jacobian.

        Args:
            compilation:
                The compilation state.
        """
        if compilation.jacobian is None:
            return
        compilation.jacobian = replace(
            compilation.jacobian,
            volumes=compilation.facts['volumes'],
            symmetries=compilation.facts['symmetries'],
            constrained=compilation.facts['constrained'][0],
        )


class FieldProblemPass(Pass):
    """Pass that adds the stable step, the linear terms, the finite volumes, and the fixed nodes to the problem."""

    reads = ('problem', 'limits', 'volume_limits', 'linear', 'volumes', 'constrained')
    writes = ('problem',)

    def run(self, compilation: Compilation) -> None:
        """Add the discretizations of the fields to the problem.

        Args:
            compilation:
                The compilation state.
        """
        problem = cast('Problem', compilation.problem)
        indices, values = compilation.facts['constrained']
        initial_state = problem.initial_state.copy()
        initial_state[indices] = values
        compilation.problem = replace(
            problem,
            initial_state=initial_state,
            stable_step=min(compilation.limits, default=None),
            diffusive_limits=tuple(compilation.volume_limits),
            linear=tuple(compilation.linear),
            volumes=compilation.facts['volumes'],
            constrained=indices,
        )


PASSES = (LowerFieldsPass(), FieldFunctionsPass(), FieldJacobianPass(), FieldProblemPass())
