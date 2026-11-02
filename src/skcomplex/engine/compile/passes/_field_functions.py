"""Pass that writes operators back into the equations and adds flux differences, symmetries, and fixed nodes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, I, matrix, solve
from sage.modules.free_module_element import FreeModuleElement

from ....core.elements import FieldElement
from ....core.laws import BaseLaw, FluxLaw
from ....core.laws.utils import COORDINATE_PREFIX
from ....core.spaces import LatticeSpace, MeshSpace, PhaseSpace
from ....errors import PipelineError, ValidationError
from ....specs import LayoutSpec, RunStateSpec
from ....types import Array, VariableKind, VectorFunction
from ..._utils import declarations_of
from ...problems import BlockSpec, DeclarationsSpec, ExpressionPrinter, KernelSpec, ReferenceSpec, SymmetrySpec
from ...problems._finite_volume_kernel import TIME, FiniteVolumeKernel
from ...problems._law_kernel import LawKernel
from ...problems._problem_functions_spec import _ProblemFunctionsSpec
from ...types import ReadKind
from .._base import BasePass
from .._compilation import Compilation

PAIR = 2


class FieldFunctionsPass(BasePass):
    """Pass that writes operators back into the equations and adds flux differences, symmetries, and fixed nodes."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('functions', 'written', 'reduced', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('functions', 'volumes', 'symmetries', 'constrained')

    def _run(self, inputs: object = None) -> None:
        """Add the discretizations of the fields to the numeric functions."""
        compilation = self.compilation_
        functions, written = compilation.functions, compilation.written
        if functions is None or written is None:
            error_msg = 'FieldFunctionsPass reads the numeric functions and the written operators, which no pass wrote.'
            raise PipelineError(error_msg)
        declarations = Compilation._required(compilation.layout, 'layout')
        states = LayoutSpec(tuple(declarations.states))
        functions = replace(functions, equations=self._written_equations(functions.equations, states, written))
        blocks = list(declarations.blocks.values())
        functions, volumes, constrained = self._with_discretizations(
            functions,
            Compilation._required(compilation.reduced, 'reduced'),
            declarations,
            blocks,
        )
        compilation.functions = functions
        compilation.volumes = volumes
        compilation.symmetries = tuple(self._symmetries(blocks))
        compilation.constrained = constrained

    @staticmethod
    def _written_equations(
        equations: Mapping[str, Expression],
        states: LayoutSpec,
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
        declarations: DeclarationsSpec,
    ) -> tuple[FiniteVolumeKernel, ...]:
        """Return the conservation laws of every finite-volume field from the flux laws of all its components.

        The flux laws of one field, from its element and from any interaction, form one system, whose wave speeds come
        from the Jacobian of all its fluxes.
        """
        found: dict[str, list[tuple[FluxLaw, ReferenceSpec, Mapping[Expression, ReferenceSpec]]]] = {}
        for spec, laws in specs:
            for law in laws:
                if not isinstance(law, FluxLaw):
                    continue
                reference = spec._target_references(law._get_targets()[0])[0]
                owner = reference.entry.name.rpartition('.')[0]
                if not isinstance(declarations.blocks[owner].node.component, FieldElement):
                    error_msg = f'{spec.owner} declares fluxes of {owner}, which is not a field.'
                    raise ValidationError(error_msg)
                found.setdefault(owner, []).append((law, reference, spec.references))
        volumes = []
        for owner in sorted(found):
            place = declarations.blocks[owner].node.component
            if not isinstance(place, FieldElement):
                error_msg = f'{owner} declares fluxes, but it is not a field.'
                raise ValidationError(error_msg)
            volumes.append(FieldFunctionsPass._compile_volumes(found[owner], place, owner, declarations))
        return tuple(volumes)

    @staticmethod
    def _mesh_constraints(blocks: list[BlockSpec]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
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
    def _symmetries(
        blocks: list[BlockSpec],
    ) -> list[SymmetrySpec]:
        """Return the mirror symmetries of the field states."""
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
                found.append(SymmetrySpec(indices, space.shape_, tuple(signs)))
        return found

    @staticmethod
    def _with_symmetries(
        drift: VectorFunction | None,
        symmetries: list[SymmetrySpec],
    ) -> VectorFunction | None:
        """Return the drift with the rates of symmetric states projected onto their odd or even parts."""
        if not symmetries or drift is None:
            return drift

        def symmetric(t: float, x: Array, p: Array) -> Array:
            rates = np.array(drift(t, x, p), dtype=float)
            for symmetry in symmetries:
                indices = symmetry.indices
                values = np.reshape(rates[indices], symmetry.shape)
                for axis, sign in symmetry.signs:
                    mirrored = np.roll(np.flip(values, axis=axis), 1, axis=axis)
                    values = (values + sign * mirrored) / 2
                rates[indices] = values.ravel()
            return rates

        return symmetric

    @staticmethod
    def _with_constraints(
        drift: VectorFunction | None,
        indices: npt.NDArray[np.intp],
    ) -> VectorFunction | None:
        """Return the drift with the rates of fixed boundary nodes held at zero."""
        if not indices.size or drift is None:
            return drift

        def constrained(t: float, x: Array, p: Array) -> Array:
            rates = np.array(drift(t, x, p), dtype=float)
            rates[indices] = 0.0
            return rates

        return constrained

    @staticmethod
    def _with_volumes(
        drift: VectorFunction | None,
        volumes: tuple[FiniteVolumeKernel, ...],
    ) -> VectorFunction | None:
        """Return the drift with the flux differences of finite-volume fields added."""
        if not volumes or drift is None:
            return drift

        def conserved(t: float, x: Array, p: Array) -> Array:
            rates = np.asarray(drift(t, x, p), dtype=float)
            for volume in volumes:
                rates = rates + np.asarray(volume.run(RunStateSpec(t, x, p)).values_, dtype=float)
            return rates

        return conserved

    @staticmethod
    def _with_discretizations(
        functions: _ProblemFunctionsSpec,
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
        declarations: DeclarationsSpec,
        blocks: list[BlockSpec],
    ) -> tuple[
        _ProblemFunctionsSpec,
        tuple[FiniteVolumeKernel, ...],
        tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]],
    ]:
        """Add flux differences, symmetries, and fixed boundary nodes to the drift, returning laws and fixed nodes."""
        volumes = FieldFunctionsPass._compile_fluxes(specs, declarations)
        symmetries = FieldFunctionsPass._symmetries(blocks)
        constrained, fixed = FieldFunctionsPass._mesh_constraints(blocks)
        drift = FieldFunctionsPass._with_volumes(functions.drift, volumes)
        drift = FieldFunctionsPass._with_symmetries(drift, symmetries)
        drift = FieldFunctionsPass._with_constraints(drift, constrained)
        if volumes or symmetries or constrained.size:
            sources = functions.drift if volumes else None
            functions = replace(functions, drift=drift, sources=sources, accelerated=None)
        return functions, volumes, (constrained, fixed)

    @staticmethod
    def _canonical(
        laws: Sequence[tuple[FluxLaw, ReferenceSpec, Mapping[Expression, ReferenceSpec]]],
        owner: str,
    ) -> tuple[list[list[Expression]], list[Expression], list[ReferenceSpec]]:
        """Return the flux components in symbols shared by all the laws, the state symbols, and the parameters read.

        Every law reads the states and parameters through the symbols of its own namespace, so the conserved states
        become one symbol each, by their entries, and so do the other values the fluxes read, parameters or states that
        no flux conserves, by their entries and how they are read.
        """
        names = [reference.entry.name for _, reference, _ in laws]
        if len(set(names)) != len(names):
            error_msg = f'Two flux laws conserve the same state of {owner}. Give one flux per state.'
            raise ValidationError(error_msg)
        states = [SR.var(f'_volume_state_{index}') for index in range(len(names))]
        parameters: dict[tuple[str, int, object, int], tuple[Expression, ReferenceSpec]] = {}
        expressions = []
        for law, _, references in laws:
            components = list(law.flux) if isinstance(law.flux, FreeModuleElement | list | tuple) else [law.flux]
            replacements = {}
            for item in {variable for value in components for variable in SR(value).variables()}:
                reference = references.get(item)
                if reference is None:
                    error_msg = f'The fluxes of {owner} read {item}, which is not a state or parameter.'
                    raise ValidationError(error_msg)
                if reference.is_state and reference.entry.name in names and reference.source is ReadKind.MEMBER:
                    replacements[item] = states[names.index(reference.entry.name)]
                    continue
                key = (reference.entry.name, reference.column, reference.source, id(reference.neighbor))
                if key not in parameters:
                    parameters[key] = (SR.var(f'_volume_parameter_{len(parameters)}'), reference)
                replacements[item] = parameters[key][0]
            expressions.append([SR(value).subs(replacements) for value in components])
        return expressions, states, [reference for _, reference in parameters.values()]

    @staticmethod
    def _function_of(printed: object, owner: str) -> Callable[..., object]:
        """Return the function a compiled printer returns for the fluxes of a field.

        Args:
            printed:
                What the printer returns.

            owner:
                The path of the field.

        Returns:
            function:
                The function.

        Raises:
            TypeError:
                The printer returns no function.
        """
        if not callable(printed):
            error_msg = f'The printer returns a {type(printed).__name__} for the fluxes of {owner}, not a function.'
            raise TypeError(error_msg)
        return printed

    @staticmethod
    def _eigenvalues(jacobian: list[list[Expression]]) -> list[Expression] | None:
        """Return the closed-form eigenvalues of a flux Jacobian, or `None` when Sage finds no explicit real ones.

        Args:
            jacobian:
                The flux Jacobian, row by row.

        Returns:
            eigenvalues:
                The eigenvalues, or `None`.
        """
        count = len(jacobian)
        if count == 1:
            return [jacobian[0][0]]
        if count == PAIR:
            trace = jacobian[0][0] + jacobian[1][1]
            determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
            root = (trace**2 / 4 - determinant).sqrt()
            return [trace / 2 - root, trace / 2 + root]
        try:
            found = matrix(SR, jacobian).eigenvalues()
        except ArithmeticError:
            return None
        if len(found) != count or any(SR(value).has(I) for value in found):
            return None
        return [SR(value) for value in found]

    @staticmethod
    def _function_speeds(
        laws: Sequence[tuple[FluxLaw, ReferenceSpec, Mapping[Expression, ReferenceSpec]]],
        parameters: Sequence[ReferenceSpec],
        n_dim: int,
        owner: str,
    ) -> tuple[Callable[..., object], ...]:
        """Return one function per dimension of the wave speeds a flux law gives as functions, or none.

        Args:
            laws:
                The flux laws of the field.

            parameters:
                The references of the other values the fluxes read.

            n_dim:
                The number of dimensions of the lattice.

            owner:
                The path of the field.

        Returns:
            speeds:
                The functions of time, the states, and the parameters, or an empty tuple.
        """
        given = [law.speed for law, _, _ in laws if law.speed is not None]
        if not given:
            return ()
        if any(item is not given[0] for item in given):
            error_msg = f'The fluxes of {owner} give different wave speeds. A field has one, so give it in one law.'
            raise ValidationError(error_msg)
        candidates = list(given[0]) if isinstance(given[0], list | tuple) else [given[0]] * n_dim
        functions = [item for item in candidates if callable(item)]
        if len(functions) != n_dim or len(candidates) != n_dim:
            error_msg = f'The speed of a flux of {owner} is {given[0]!r}. Give a function, or one per dimension.'
            raise ValidationError(error_msg)
        names = [
            reference.entry.name.rpartition('.')[2]
            for reference in (*(reference for _, reference, _ in laws), *parameters)
        ]

        def speed_of(function: Callable[..., object]) -> Callable[..., object]:
            def speed(t: float, *values: object) -> list[object]:
                return [function(t, dict(zip(names, values, strict=True)))]

            return speed

        return tuple(speed_of(function) for function in functions)

    @staticmethod
    def _compile_speeds(
        columns: list[tuple[Expression, ...]],
        states: list[Expression],
        printer: ExpressionPrinter,
        owner: str,
    ) -> tuple[Callable[..., object], ...]:
        """Return one function per dimension of the closed-form wave speeds, or none when a Jacobian has no closed form.

        Args:
            columns:
                The flux of every state, one column per dimension.

            states:
                The symbols of the conserved states.

            printer:
                The compiled printer of the states and the parameters.

            owner:
                The path of the field.

        Returns:
            speeds:
                The functions of the eigenvalues, or an empty tuple.
        """
        found = []
        for column in columns:
            eigenvalues = FieldFunctionsPass._eigenvalues([[value.diff(state) for state in states] for value in column])
            if eigenvalues is None:
                return ()
            found.append(eigenvalues)
        return tuple(FieldFunctionsPass._function_of(printer.run(values).code_, owner) for values in found)

    @staticmethod
    def _reads_along_axis(
        drift: Expression,
        symbols: Sequence[Expression],
        parameters: Sequence[ReferenceSpec],
        space: LatticeSpace,
        axis: int,
    ) -> bool:
        """Check whether a drift reads a value that varies along its own axis, a coordinate or a field over position."""
        own = next((name for name, index in getattr(space, 'class_axes_', {}).items() if index == axis), None)
        found = [reference for item, reference in zip(symbols, parameters, strict=True) if item in drift.variables()]
        return any(
            reference.entry.name.endswith(f'.{COORDINATE_PREFIX}{own}')
            or (
                own is None
                and reference.source is not ReadKind.SINGLE
                and COORDINATE_PREFIX not in reference.entry.name
            )
            for reference in found
        )

    @staticmethod
    def _linear_drifts(
        columns: Sequence[tuple[Expression, ...]],
        states: Sequence[Expression],
        symbols: Sequence[Expression],
        *,
        parameters: Sequence[ReferenceSpec],
        printer: ExpressionPrinter,
        owner: str,
        space: LatticeSpace,
    ) -> tuple[Callable[..., object], ...]:
        """Return one function per axis of the drift of every state, or none when a flux is not a linear drift.

        Args:
            columns:
                The flux of every state, one column per dimension.

            states:
                The symbols of the conserved states.

            symbols:
                The symbols of the other values the fluxes read.

            parameters:
                The references of the other values the fluxes read.

            printer:
                The compiled printer of the states and the parameters.

            owner:
                The path of the field.

            space:
                The lattice of the field.

        Returns:
            drifts:
                The functions, or an empty tuple.
        """
        functions = []
        for axis, column in enumerate(columns):
            drifts = [SR(flux).diff(state) for flux, state in zip(column, states, strict=True)]
            linear = all(
                not (set(states) & set(drift.variables()))
                and (SR(flux) - drift * state).is_zero()
                and not FieldFunctionsPass._reads_along_axis(drift, symbols, parameters, space, axis)
                for flux, state, drift in zip(column, states, drifts, strict=True)
            )
            if not linear:
                return ()
            functions.append(FieldFunctionsPass._function_of(printer.run(drifts).code_, owner))
        return tuple(functions)

    @staticmethod
    def _class_edges(space: LatticeSpace) -> tuple[str | None, ...]:
        """Return the edges of every class axis of a phase space and `None` for its position axes, or nothing."""
        if not isinstance(space, PhaseSpace):
            return ()
        return tuple(
            None if axis in space.position_axes_ else space._axis_boundary(axis) for axis in range(space.n_dim_)
        )

    @staticmethod
    def _parameter_reads(
        parameters: Sequence[ReferenceSpec],
        references: Mapping[Expression, ReferenceSpec],
        declarations: DeclarationsSpec,
        owner: str,
    ) -> tuple[Callable[[float, Array, Array], object] | None, ...]:
        """Return the function that reads every value of the fluxes held by no vector, such as a moment, else `None`."""
        reader = LawKernel(
            f'{owner} fluxes',
            declarations.time,
            references,
            0,
            LayoutSpec(tuple(declarations.states)),
            layout_parameters=LayoutSpec(tuple(declarations.parameters)),
            outputs=[],
        )
        symbols = {id(reference): item for item, reference in references.items()}

        def read_of(item: Expression) -> Callable[[float, Array, Array], object]:
            return lambda t, x, p: reader.read(item, t, x, p)

        return tuple(
            read_of(symbols[id(reference)]) if reference.source in {ReadKind.DERIVED, ReadKind.NEIGHBOR} else None
            for reference in parameters
        )

    @staticmethod
    def _compile_volumes(
        laws: Sequence[tuple[FluxLaw, ReferenceSpec, Mapping[Expression, ReferenceSpec]]],
        place: FieldElement,
        owner: str,
        declarations: DeclarationsSpec,
    ) -> FiniteVolumeKernel:
        """Return the conservation laws of one finite-volume field.

        Args:
            laws:
                The flux laws of the field, with the reference of every target and the references of the namespace
                of the component that returns the law.

            place:
                The field.

            owner:
                The path of the field.

            declarations:
                The layout of the problem.

        Returns:
            volumes:
                The compiled conservation laws.

        Raises:
            ValidationError:
                The field is not a finite-volume field on a lattice, the fluxes are not one per dimension, read
                symbols that are neither conserved states nor parameters, or ask for different schemes.
        """
        space = place.space
        if place.discretization != 'finite-volume' or not isinstance(space, LatticeSpace):
            error_msg = f'{owner} declares fluxes, which need a finite-volume field on a lattice.'
            raise ValidationError(error_msg)
        expressions, states, parameters = FieldFunctionsPass._canonical(laws, owner)
        n_dim = space.n_dim_
        if any(len(row) != n_dim for row in expressions):
            error_msg = f'A flux of {owner} has not one component per dimension of its {n_dim}D lattice.'
            raise ValidationError(error_msg)
        first = laws[0][0]
        if any((law.limiter, law.riemann) != (first.limiter, first.riemann) for law, _, _ in laws):
            error_msg = f'The fluxes of {owner} ask for different limiters or numerical fluxes. Give one scheme.'
            raise ValidationError(error_msg)
        symbols = [SR.var(f'_volume_parameter_{index}') for index in range(len(parameters))]
        columns = list(zip(*expressions, strict=True))
        printer = ExpressionPrinter(time=TIME, compiled=True).check().prepare([*states, *symbols])
        fluxes = tuple(FieldFunctionsPass._function_of(printer.run(list(column)).code_, owner) for column in columns)
        jacobians = tuple(
            FieldFunctionsPass._function_of(
                printer.run([value.diff(state) for value in column for state in states]).code_,
                owner,
            )
            for column in columns
        )
        critical: tuple[Callable[..., object], ...] = ()
        if first.riemann == 'godunov':
            if len(states) != 1:
                error_msg = f'The fluxes of {owner} ask for the Godunov flux, which is exact for one state only.'
                raise ValidationError(error_msg)
            roots = []
            critical_printer = ExpressionPrinter(time=TIME, compiled=True).check().prepare(symbols)
            for column in columns:
                found = [item.rhs() for item in solve(column[0].diff(states[0]) == 0, states[0])]
                roots.append(FieldFunctionsPass._function_of(critical_printer.run(found).code_, owner))
            critical = tuple(roots)
        references = {key: item for _, _, read in laws for key, item in read.items()}
        boundaries = tuple(
            (kind, float(SR(value)) if not callable(value) else 0.0)
            for kind, value in (
                place._get_boundary(reference.entry.name.rpartition('.')[2], reference.column)
                for _, reference, _ in laws
            )
        )
        return (
            FiniteVolumeKernel(
                owner,
                tuple(reference.entry.start for _, reference, _ in laws),
                space.shape_,
                space.spacings_,
                boundaries,
                fluxes=fluxes,
                jacobians=jacobians,
                parameters=tuple(parameters),
                reads=FieldFunctionsPass._parameter_reads(parameters, references, declarations, owner),
                drifts=FieldFunctionsPass._linear_drifts(
                    columns,
                    states,
                    symbols,
                    parameters=parameters,
                    printer=printer,
                    owner=owner,
                    space=space,
                ),
                edges=FieldFunctionsPass._class_edges(space),
                limiter=first.limiter,
                riemann=first.riemann,
                critical=critical,
                expressions=tuple(columns),
                speeds=(
                    FieldFunctionsPass._function_speeds(laws, parameters, n_dim, owner)
                    or FieldFunctionsPass._compile_speeds(columns, states, printer, owner)
                ),
            )
            .check()
            .prepare()
        )
