"""Compile fields: differential operators, finite volumes, finite elements, symmetries, and fixed boundaries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, solve
from sage.modules.free_module_element import FreeModuleElement

from ....core.elements import FieldElement
from ....core.errors import ValidationError
from ....core.laws import BaseLaw, BoundaryLaw, FluxLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.spaces import LatticeSpace, MeshSpace
from ....core.types import Vector
from ...problems import (
    Layout,
    NoiseProcess,
    Output,
    Problem,
    Reference,
    Source,
    VariableKind,
    declarations_of,
    numpy_function,
)
from .._census import Census, CensusRecord
from .._compilation import Compilation, Laws
from .._declarations import Block, Declarations
from .._field_conserved import Indices
from .._field_lowering import FieldLowering, _has_operators
from .._field_splitting import OPERATORS, LinearTerm, Terms
from .._field_volumes import TIME, FiniteVolume
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
                    terms += LowerFieldsPass._linear_terms(law.value, reference, spec.references, places)
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

    @staticmethod
    def _name_of(operator: object) -> str:
        """Return the name of a Python operator of a Sage expression, or an empty string."""
        return getattr(operator, '__name__', '')

    @staticmethod
    def _sum_terms(operands: list[Any], states: Mapping[Expression, Any]) -> Terms | None:
        """Return the linear terms of a sum, or `None` when a summand is not linear."""
        terms: Terms = []
        for item in operands:
            part = LowerFieldsPass._linear(item, states)
            if part is None:
                return None
            terms += part
        return terms

    @staticmethod
    def _product_terms(operands: list[Any], states: Mapping[Expression, Any]) -> Terms | None:
        """Return the linear terms of a constant times a linear expression, or `None`."""
        constant, variable = SR(1), []
        for item in operands:
            if any(symbol in states for symbol in item.variables()):
                variable.append(item)
            else:
                constant *= item
        if len(variable) != 1:
            return None
        part = LowerFieldsPass._linear(variable[0], states)
        return None if part is None else [(constant * coefficient, chain, state) for coefficient, chain, state in part]

    @staticmethod
    def _operator_terms(name: str, operands: list[Any], states: Mapping[Expression, Any]) -> Terms | None:
        """Return the linear terms of an operator applied to a linear expression, or `None`."""
        inner = LowerFieldsPass._linear(operands[0], states)
        if inner is None:
            return None
        axis = int(operands[1]) if name == 'partial' else 0
        return [(coefficient, ((name, axis), *chain), state) for coefficient, chain, state in inner]

    @staticmethod
    def _linear(expression: Any, states: Mapping[Expression, Any]) -> Terms | None:  # noqa: ANN401
        """Return the linear terms of an expression in field states, or `None` when it is not linear."""
        expression = SR(expression)
        if not any(item in states for item in expression.variables()):
            return []
        if expression in states:
            return [(SR(1), (), expression)]
        operator = expression.operator()
        name = getattr(operator, 'name', lambda: '')()
        if name in OPERATORS:
            return LowerFieldsPass._operator_terms(name, expression.operands(), states)
        combine = {'add_vararg': LowerFieldsPass._sum_terms, 'mul_vararg': LowerFieldsPass._product_terms}.get(
            LowerFieldsPass._name_of(operator),
        )
        return None if combine is None else combine(expression.operands(), states)

    @staticmethod
    def _linear_terms(
        value: object,
        target: Reference,
        references: Mapping[Expression, Reference],
        fields: Mapping[str, FieldElement],
    ) -> list[LinearTerm]:
        """Return the linear terms with constant coefficients of the rate of a field state.

        Additive terms that are not linear in the field's states, or whose coefficients depend on states, are left out,
        and solvers treat them as the nonlinear remainder.

        Args:
            value:
                The rate as written, with its operators.

            target:
                The component the rate belongs to.

            references:
                The references of the symbols of the rate's component.

            fields:
                The field of every member entry by entry name.

        Returns:
            terms:
                The linear terms.
        """
        if not isinstance(value, Expression):
            return []
        states = {
            item: reference
            for item, reference in references.items()
            if reference.is_state and reference.source is Source.MEMBER and reference.entry.name in fields
        }
        expanded = SR(value).expand()
        parts = expanded.operands() if LowerFieldsPass._name_of(expanded.operator()) == 'add_vararg' else [expanded]
        terms = []
        for part in parts:
            found = LowerFieldsPass._linear(part, states)
            for coefficient, chain, state in found or []:
                if any(item in states or references.get(item, None) is None for item in coefficient.variables()):
                    continue
                source = states[state]
                terms.append(
                    LinearTerm(
                        target,
                        source,
                        fields[source.entry.name],
                        chain,
                        coefficient,
                        {item: references[item] for item in coefficient.variables()},
                    ),
                )
        return terms


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
                    outputs[id(law)] = ConservedNoisePass._law_outputs(
                        spec.owner, law, targets, places, spec.references,
                    )
        return outputs

    @staticmethod
    def _amplitude(owner: str, law: NoiseLaw, references: Mapping[Expression, Reference]) -> Expression:
        """Return the amplitude of a conserved noise, or raise an error when it reads a state or is not symbolic."""
        if not isinstance(law.value, Expression | int | float):
            error_msg = (
                f'The conserved noise law of {owner} has the value {law.value!r}. Give a number or an expression.'
            )
            raise ValidationError(error_msg)
        amplitude = SR(law.value)
        reads = [item for item in amplitude.variables() if item in references and references[item].is_state]
        if reads:
            error_msg = (
                f'The amplitude of the conserved noise law of {owner} reads the states {", ".join(map(str, reads))}. '
                'Write it as a number or an expression of parameters.'
            )
            raise ValidationError(error_msg)
        return amplitude

    @staticmethod
    def _check_place(owner: str, reference: Reference, place: object) -> LatticeSpace:
        """Return the lattice of the field a conserved noise drives, or raise an error when there is none."""
        space = getattr(place, 'space', None)
        is_lattice = (
            isinstance(place, FieldElement)
            and isinstance(space, LatticeSpace)
            and space.geometry == 'square'
            and place.discretization == 'finite-difference'
            and reference.source is Source.MEMBER
        )
        if not is_lattice:
            error_msg = (
                f'The conserved noise law of {owner} drives {reference.entry.name}, which is not a state of a '
                'finite-difference field on a square lattice. Declare the state on a `FieldElement` of a '
                '`LatticeSpace`.'
            )
            raise ValidationError(error_msg)
        return space  # type: ignore[return-value]

    @staticmethod
    def _link_tables(dimensions: tuple[int, ...], periodic: bool) -> list[tuple[Indices, Indices]]:
        """Return the links of every site along every dimension of a lattice, numbered `k n + i` for site `i`.

        The link `(i, k)` joins site `i` to its next neighbour along dimension `k`. Without periodic edges, the links
        that leave the lattice carry no process.

        Args:
            dimensions:
                The number of sites along every dimension.

            periodic:
                Whether opposite edges are joined.

        Returns:
            tables:
                For every dimension, the link to the next neighbour and the link from the previous neighbour of every
                site in member order, or -1 where the link carries no process.
        """
        size = int(np.prod(dimensions))
        sites = np.arange(size, dtype=np.intp).reshape(dimensions)
        tables = []
        for axis in range(len(dimensions)):
            following = axis * size + sites
            preceding = np.roll(following, 1, axis=axis)
            if not periodic:
                following = following.copy()
                np.moveaxis(following, axis, 0)[-1] = -1
                np.moveaxis(preceding, axis, 0)[0] = -1
            tables.append((following.ravel(), preceding.ravel()))
        return tables

    @staticmethod
    def _law_outputs(
        owner: str,
        law: NoiseLaw,
        targets: list[Reference],
        places: Mapping[str, FieldElement],
        references: Mapping[Expression, Reference],
    ) -> list[Output]:
        """Return the outputs of a conserved noise law, the divergence of a random flux on the links of a lattice.

        Every target component gets one process per link. Site `i` takes `a / h` times the increment of the link from
        its previous neighbour and `-a / h` times the increment of the link to its next neighbour along every dimension.

        Args:
            owner:
                The path of the component that returned the law.

            law:
                The conserved noise law.

            targets:
                The target components of the law.

            places:
                The fields by the name of each of their entries.

            references:
                A mapping from every symbol the law may use to the component it stands for.

        Returns:
            outputs:
                Two outputs per dimension and target component, with the link process of every site.

        Raises:
            ValidationError:
                A target is not a state of a finite-difference field on a square lattice, or the amplitude reads a
                state.
        """
        amplitude = ConservedNoisePass._amplitude(owner, law, references)
        outputs = []
        for reference in targets:
            place = places.get(reference.entry.name)
            space = ConservedNoisePass._check_place(owner, reference, place)
            kind, _ = place._get_boundary(reference.entry.name.rpartition('.')[2])  # type: ignore[arg-type]
            tables = ConservedNoisePass._link_tables(space.shape_, kind == 'periodic')
            key = ('conserved', reference.entry.name, reference.column)
            dimension = len(tables) * space.size_
            for spacing, (following, preceding) in zip(space.spacings_, tables, strict=True):
                scaled = amplitude / spacing
                outputs.append(
                    Output(reference, -scaled, process=NoiseProcess(key, dimension=dimension, links=following)),
                )
                outputs.append(
                    Output(reference, scaled, process=NoiseProcess(key, dimension=dimension, links=preceding)),
                )
        return outputs
