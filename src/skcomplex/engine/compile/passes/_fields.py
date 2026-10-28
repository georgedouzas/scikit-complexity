"""Compile fields: differential operators, finite volumes, finite elements, symmetries, and fixed boundaries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression

from ....core.elements import FieldElement
from ....core.errors import ValidationError
from ....core.laws import BaseLaw, BoundaryLaw, FluxLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.spaces import LatticeSpace, MeshSpace
from ....core.types import Vector
from ...problems import Layout, Output, Problem, Source, VariableKind, declarations_of
from .._census import Census, CensusRecord
from .._compilation import Compilation, Laws
from .._declarations import Block, Declarations
from .._field_conserved import conserved_outputs
from .._field_lowering import FieldLowering, _has_operators
from .._field_splitting import LinearTerm, _linear_terms
from .._field_volumes import FiniteVolume, _compile_volumes
from .._issues import Issue
from .._laws import replaced
from .._pipeline import BasePass
from .._specs import KernelSpec, target_references
from .._tree import Node


def _places(declarations: Declarations) -> dict[str, FieldElement]:
    """Return the fields by the name of each of their entries."""
    return {
        entry.name: block.node.component
        for block in declarations.blocks.values()
        if isinstance(block.node.component, FieldElement)
        for entry in block.entries.values()
    }


def _is_lattice_field(component: object) -> bool:
    """Check whether a component is a finite-difference field on a square lattice."""
    return (
        isinstance(component, FieldElement)
        and isinstance(component.space, LatticeSpace)
        and component.space.geometry == 'square'
        and component.discretization == 'finite-difference'
    )


def _conserved_issue(record: CensusRecord, components: Mapping[str, object]) -> Issue | None:
    """Return the issue of a conserved noise law that drives a state outside a finite-difference lattice field."""
    outside = [name for name, _ in record.driven if not _is_lattice_field(components.get(name.rpartition('.')[0]))]
    if not outside:
        return None
    message = (
        f'The conserved {type(record.law).__name__} of {record.path} drives {", ".join(dict.fromkeys(outside))}, '
        'but conserved noise needs a state of a finite-difference field on a square lattice, with one process per link.'
    )
    fix = 'Declare the state on a `FieldElement` of a `LatticeSpace`, or remove `conserved=True`.'
    return Issue('noise-unsupported', record.path, message, fix)


def _conserved_issues(compilation: Compilation) -> list[Issue]:
    """Return the conserved noise laws that drive a state that is not of a finite-difference field on a lattice."""
    components = {node.path: node.component for node in cast('list[Node]', compilation.nodes)}
    records = cast('Census', compilation.census).records
    found = [
        _conserved_issue(record, components)
        for record in records
        if isinstance(record.law, NoiseLaw) and record.law.conserved
    ]
    return [issue for issue in found if issue is not None]


class LowerFieldsPass(BasePass):
    """Pass that lowers the differential operators of laws to stencils and notes stable steps and linear terms."""

    reads = ('corrected', 'layout')
    writes = ('corrected', 'written', 'limits', 'volume_limits', 'linear')

    def _run(self, compilation: Compilation) -> None:
        """Lower the differential operators of the corrected laws.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        facts: dict[str, Any] = {'written': defaultdict(list), 'limits': [], 'volume_limits': [], 'linear': []}
        compilation.corrected = [
            (spec, self._with_fields(spec, laws, declarations, facts))
            for spec, laws in cast('Laws', compilation.corrected)
        ]
        compilation.facts.update(facts)

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

    @staticmethod
    def _linear_terms_of(spec: KernelSpec, laws: list[BaseLaw], places: dict[str, FieldElement]) -> list[LinearTerm]:
        """Return the linear terms with constant coefficients of the rates of field states."""
        terms = []
        for law in laws:
            if not isinstance(law, RateLaw) or not isinstance(law.value, Expression):
                continue
            for reference in target_references(law._get_targets()[0], spec):
                if reference.entry.name in places and reference.source is Source.MEMBER:
                    terms += _linear_terms(law.value, reference, spec.references, places)
        return terms

    @staticmethod
    def _with_fields(
        spec: KernelSpec,
        laws: list[BaseLaw],
        declarations: Declarations,
        facts: dict[str, Any],
    ) -> list[BaseLaw]:
        """Return laws whose differential operators are lowered, noting stable steps and the linear terms of rates."""
        places = _places(declarations)
        facts['linear'] += LowerFieldsPass._linear_terms_of(spec, laws, places)
        if not any(_has_operators(getattr(law, 'value', None)) for law in laws):
            return laws
        lowering = FieldLowering(
            spec.owner,
            spec.references,
            places,
            LowerFieldsPass._initial_values(spec, declarations),
        )
        result: list[BaseLaw] = []
        for law in laws:
            if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw) or not _has_operators(law.value):
                result.append(law)
                continue
            if isinstance(law, RateLaw):
                lowering.note_limit(law.value)
            lowered = lowering.lower(law.value)
            if isinstance(law.value, Expression):
                for reference in target_references(law._get_targets()[0], spec):
                    facts['written'][(reference.entry.name, reference.column)].append((lowered, law.value))
            result.append(replaced(law, value=lowered))
        facts['limits'] += lowering.limits
        facts['volume_limits'] += lowering.volume_limits
        return result


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
            targets = [(law, target_references(law._get_targets()[0], spec)[0]) for law in fluxes]
            owners = {reference.entry.name.rpartition('.')[0] for _, reference in targets}
            for owner in sorted(owners):
                block = declarations.blocks[owner]
                place = block.node.component
                if not isinstance(place, FieldElement):
                    error_msg = f'{spec.owner} declares fluxes of {owner}, which is not a field.'
                    raise ValidationError(error_msg)
                mine = [(law, reference) for law, reference in targets if reference.entry.name.startswith(f'{owner}.')]
                volumes.append(_compile_volumes(mine, spec.references, place, owner))
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


class FieldJacobianPass(BasePass):
    """Pass that adds the conservation laws, symmetries, and fixed nodes of the fields to the exact Jacobian."""

    reads = ('jacobian', 'volumes', 'symmetries', 'constrained')
    writes = ('jacobian',)

    def _run(self, compilation: Compilation) -> None:
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


class FieldProblemPass(BasePass):
    """Pass that adds the stable step, the linear terms, the finite volumes, and the fixed nodes to the problem."""

    reads = ('problem', 'limits', 'volume_limits', 'linear', 'volumes', 'constrained')
    writes = ('problem',)

    def _run(self, compilation: Compilation) -> None:
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
            stable_step=min(compilation.facts['limits'], default=None),
            diffusive_limits=tuple(compilation.facts['volume_limits']),
            linear=tuple(compilation.facts['linear']),
            volumes=compilation.facts['volumes'],
            constrained=indices,
        )


class ConservedNoisePass(BasePass):
    """Pass that lowers every conserved noise law to the divergence of one random flux per link of its field."""

    reads = ('reduced', 'layout', 'outputs')
    writes = ('outputs',)

    def _run(self, compilation: Compilation) -> None:
        """Add the outputs of the conserved noise laws.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        compilation.outputs.update(self._conserved_outputs(cast('Laws', compilation.reduced), declarations))

    @staticmethod
    def _conserved_outputs(specs: Laws, declarations: Declarations) -> dict[int, list[Output]]:
        """Return the outputs of every conserved noise law, one process per link of its field, by law identity."""
        places = _places(declarations)
        outputs = {}
        for spec, laws in specs:
            for law in laws:
                if isinstance(law, NoiseLaw) and law.conserved:
                    targets = target_references(law._get_targets()[0], spec)
                    outputs[id(law)] = conserved_outputs(spec.owner, law, targets, places, spec.references)
        return outputs
