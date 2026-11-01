"""Lower the laws of every kernel spec into kernels, one per law kind and component."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.laws import BaseLaw, BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....errors import PipelineError, ValidationError
from ....specs import CallSpec, LayoutSpec, NoiseProcessSpec, WienerProcessSpec
from ....types import ProcessKey
from ..._utils import components_of, symbols_of
from ...problems import DeclarationsSpec, KernelSpec, LoweredKernelsSpec, OutputSpec, ReferenceSpec
from ...problems._law_kernel import LawKernel
from ...types import ReadKind
from .._base import BasePass
from .._compilation import Compilation, Laws
from .._laws import _KERNEL_FREE_LAWS
from ._noise import NoisePass

_ASSIGNED_KINDS = (UpdateLaw, BoundaryLaw)


class KernelsPass(BasePass):
    """Pass that lowers the reduced laws into the kernels of every law kind."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('reduced', 'layout', 'outputs')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('kernels',)

    def _run(self, inputs: object = None) -> None:
        """Lower the reduced laws into kernels."""
        compilation = self.compilation_
        if compilation.outputs is None:
            error_msg = 'KernelsPass reads the outputs that the passes of formalisms lowered, which no pass wrote.'
            raise PipelineError(error_msg)
        compilation.kernels = self._lower_laws(
            Compilation._required(compilation.reduced, 'reduced'),
            Compilation._required(compilation.layout, 'layout'),
            compilation.outputs,
        )

    @staticmethod
    def _outputs_of(value: object, targets: Sequence[ReferenceSpec | None], owner: str) -> list[OutputSpec]:
        """Return the outputs of one law value for its target components, with the law and component as owner."""
        if isinstance(value, CallSpec):
            arguments = tuple(
                item
                for argument in value.arguments
                for item in KernelsPass._symbolic_components(argument, 1, owner) or []
            )
            if len(arguments) != len(value.arguments):
                error_msg = f'A Call in a {owner} has an argument that is not a symbolic expression.'
                raise ValidationError(error_msg)
            function = value.function
            if not callable(function):
                error_msg = f'A Call in a {owner} calls {function!r}, which is not a function.'
                raise ValidationError(error_msg)
            return [
                OutputSpec(
                    target,
                    function=function,
                    arguments=arguments,
                    component=index,
                    components=len(targets),
                )
                for index, target in enumerate(targets)
            ]
        expressions = KernelsPass._symbolic_components(value, len(targets), owner)
        if expressions is None:
            if not callable(value):
                error_msg = f'A {owner} has the value {value!r}, which is not a symbolic expression or a function.'
                raise ValidationError(error_msg)
            return [
                OutputSpec(target, function=value, component=index, components=len(targets))
                for index, target in enumerate(targets)
            ]
        return [OutputSpec(target, expression) for target, expression in zip(targets, expressions, strict=True)]

    @staticmethod
    def _combines(laws: Sequence[BaseLaw]) -> bool:
        """Check whether update laws of one state combine: each an increment or a move from its own source value."""
        if not all(isinstance(law, UpdateLaw) and law.order is None for law in laws):
            return False
        updates = [law for law in laws if isinstance(law, UpdateLaw)]
        if not all(law.increment or law.source is not None for law in updates):
            return False
        sources = [float(law.source) for law in updates if law.source is not None]
        return len(set(sources)) == len(sources)

    @staticmethod
    def _check_assignments(
        spec: KernelSpec,
        law: BaseLaw,
        targets: list[ReferenceSpec],
        assigned: dict[tuple[object, ...], list[BaseLaw]],
        *,
        once: bool,
    ) -> set[int]:
        """Raise an error for an update or boundary law that sums over members or sets a component it cannot share.

        Returns the positions of the targets that an earlier law sets, whose changes this law adds to. A law of one
        member of a group meets the laws of every member, and a law of every member the laws of each one.
        """
        kind = type(law).__name__
        combined = set()
        for position, target in enumerate(targets):
            if target.source in {ReadKind.FIRST, ReadKind.SECOND}:
                error_msg = (
                    f'A {kind} law of {spec.owner} sets {target.entry.name} at an end of every edge, which several '
                    'edges share. Set the edge member, or reduce over the neighbours in the laws of the nodes.'
                )
                raise ValidationError(error_msg)
            if target.source is ReadKind.SINGLE and spec.rows > 1 and not once:
                error_msg = f'A {kind} law of {spec.owner} targets a single element from a group, which has no meaning.'
                raise ValidationError(error_msg)
            keys = KernelsPass._assignment_keys(target)
            laws = assigned[keys[0]] + assigned[keys[1]]
            if laws and not KernelsPass._combines([*laws, law]):
                error_msg = (
                    f'A {kind} law of {spec.owner} sets {target.entry.name} a second time. Each takes one law, unless '
                    'every update law of the state is an increment or moves the state from its own source value.'
                )
                raise ValidationError(error_msg)
            if laws:
                combined.add(position)
            for key in keys[1:]:
                assigned[key].append(law)
        return combined

    @staticmethod
    def _assignment_keys(target: ReferenceSpec) -> tuple[tuple[object, ...], ...]:
        """Return the keys of the laws an assignment of a target meets, the first two, and holds, all but the first.

        The laws of every member of a group are held under the component, and the laws of one member under the component
        and the member and under the laws of single members, so each meets the other.
        """
        name = target.entry.name
        if target.source is ReadKind.MEMBER or target.entry.members is None:
            return ((name, target.column, 'one'), (name, target.column))
        member, column = divmod(target.column, target.entry.columns)
        return ((name, column), (name, column, member), (name, column, 'one'))

    @staticmethod
    def _is_once(spec: KernelSpec, law: BaseLaw) -> bool:
        """Check whether a law of a group's interaction sets single elements from reductions alone, so it runs once."""
        if (
            spec.rows <= 1
            or spec.pairs is not None
            or not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw)
        ):
            return False
        if not all(reference.source is ReadKind.SINGLE for reference in spec._target_references(law._get_targets()[0])):
            return False
        expressions = [item for item in components_of(law.value) if isinstance(item, Expression)]
        sources = {spec.references[item].source for item in symbols_of(expressions) if item in spec.references}
        return ReadKind.REDUCED in sources and ReadKind.MEMBER not in sources

    @staticmethod
    def _single_references(spec: KernelSpec) -> dict[Expression, ReferenceSpec]:
        """Return the references of a spec without its group members, for laws that run once."""
        return {
            item: reference
            for item, reference in spec.references.items()
            if reference.source is not ReadKind.MEMBER and reference.entry.members is None
        }

    @staticmethod
    def _law_outputs(
        spec: KernelSpec,
        law: RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw,
        targets: list[ReferenceSpec],
        declarations: DeclarationsSpec,
        lowered: dict[int, list[OutputSpec]],
    ) -> list[OutputSpec]:
        """Return the outputs of a law for its target components, with the named Wiener process of a noise law.

        A law that a pass of a formalism lowered takes the outputs of that pass.
        """
        if id(law) in lowered:
            return lowered[id(law)]
        if isinstance(law, NoiseLaw) and law.conserved:
            error_msg = (
                f'A conserved noise law of {spec.owner} has no lowering. Declare its state on a `Field` of the fields '
                'formalism, or remove `conserved=True`.'
            )
            raise ValidationError(error_msg)
        outputs = KernelsPass._outputs_of(law.value, targets, f'{type(law).__name__} of {spec.owner}')
        if isinstance(law, NoiseLaw) and law.process is not None:
            keys = NoisePass._process_keys(spec, law, declarations)
            outputs = [
                replace(item, process=KernelsPass._noise_process(key, declarations.wieners))
                for item, key in zip(outputs, keys, strict=True)
            ]
        return outputs

    @staticmethod
    def _lower_laws(
        specs: Laws,
        declarations: DeclarationsSpec,
        lowered_outputs: dict[int, list[OutputSpec]],
    ) -> LoweredKernelsSpec:
        """Return the kernels of every law kind, with the law of every output, the update laws of each phase apart."""
        states, parameters = LayoutSpec(tuple(declarations.states)), LayoutSpec(tuple(declarations.parameters))
        kernels: defaultdict[type, list[LawKernel]] = defaultdict(list)
        jump_changes: dict[int, list[tuple[ReferenceSpec, float]]] = {}
        laws_of: dict[int, tuple[KernelSpec, BaseLaw]] = {}
        assigned: dict[tuple[type, str | None], defaultdict[tuple[object, ...], list[BaseLaw]]] = defaultdict(
            lambda: defaultdict(list),
        )
        for spec, laws in specs:
            outputs: dict[tuple[type, bool, str | None], list[OutputSpec]] = defaultdict(list)
            for law in laws:
                if isinstance(law, _KERNEL_FREE_LAWS):
                    continue
                if isinstance(law, JumpLaw):
                    output = KernelsPass._outputs_of(law.rate, [None], f'JumpLaw of {spec.owner}')[0]
                    jump_changes[id(output)] = [
                        (spec._target_references(target)[0], float(increment))
                        for target, increment in (law.changes or {}).items()
                    ]
                    laws_of[id(output)] = (spec, law)
                    outputs[(JumpLaw, False, None)].append(output)
                    continue
                if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
                    error_msg = f'A {type(law).__name__} of {spec.owner} reached lowering before it was rewritten.'
                    raise ValidationError(error_msg)
                targets = spec._target_references(law._get_targets()[0])
                once = KernelsPass._is_once(spec, law)
                reduced = isinstance(law, UpdateLaw) and spec.reduce is not None
                phase = law.phase if isinstance(law, UpdateLaw) else None
                combined = (
                    KernelsPass._check_assignments(spec, law, targets, assigned[(type(law), phase)], once=once)
                    if isinstance(law, _ASSIGNED_KINDS) and not reduced
                    else set()
                )
                law_outputs = KernelsPass._law_outputs(spec, law, targets, declarations, lowered_outputs)
                law_outputs = [
                    replace(item, combined=True) if position in combined else item
                    for position, item in enumerate(law_outputs)
                ]
                if reduced:
                    law_outputs = [replace(item, reduce=spec.reduce, counts=spec.counts) for item in law_outputs]
                laws_of.update({id(item): (spec, law) for item in law_outputs})
                outputs[(KernelsPass._kind_of(law), once, phase)].extend(law_outputs)
            for (kind, once, _), kind_outputs in outputs.items():
                kernels[kind].append(
                    LawKernel(
                        spec.owner,
                        declarations.time,
                        KernelsPass._single_references(spec) if once else spec.references,
                        1 if once else spec.rows,
                        states,
                        layout_parameters=parameters,
                        pairs=spec.pairs,
                        outputs=kind_outputs,
                    )
                    .check()
                    .prepare(),
                )
        return LoweredKernelsSpec(kernels, jump_changes, laws_of)

    @staticmethod
    def _covariance_factor(covariance: object) -> npt.NDArray[np.float64]:
        """Return a factor `L` with `L L^T = C` of a covariance, by Cholesky or, for a singular one, by eigenvalues."""
        matrix = np.asarray(covariance, dtype=float)
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            values, vectors = np.linalg.eigh(matrix)
            return vectors * np.sqrt(np.maximum(values, 0.0))

    @staticmethod
    def _noise_process(key: ProcessKey | None, wieners: Mapping[str, WienerProcessSpec]) -> NoiseProcessSpec | None:
        """Return the named Wiener process of a process key, with the covariance factor of a `WienerProcessSpec`."""
        if key is None:
            return None
        kind, owner, item = key
        if kind == 'local':
            return NoiseProcessSpec(key)
        wiener = wieners[owner]
        factor = None if wiener.covariance is None else KernelsPass._covariance_factor(wiener.covariance)
        return NoiseProcessSpec(owner, item, wiener.dimension, factor)

    @staticmethod
    def _kind_of(law: BaseLaw) -> type:
        """Return the class of a law that decides its form, with every noise law counted as a noise law.

        Args:
            law:
                The law.

        Returns:
            kind:
                The class.
        """
        return NoiseLaw if isinstance(law, NoiseLaw) else type(law)

    @staticmethod
    def _is_sympy(value: object) -> bool:
        """Check whether a value comes from SymPy, without importing it."""
        return type(value).__module__.split('.')[0] == 'sympy'

    @staticmethod
    def _sympy_error(value: object, owner: str) -> ValidationError:
        """Return the error of a SymPy value in a law."""
        error_msg = (
            f'A {owner} has the SymPy value {value}. Sage is the symbolic layer, so write the expression with the '
            'symbols of the namespace `s` and Sage functions, or convert it with `expression._sage_()`.'
        )
        return ValidationError(error_msg)

    @staticmethod
    def _to_expression(component: object, owner: str) -> Expression:
        """Return a component as a Sage expression."""
        if KernelsPass._is_sympy(component):
            raise KernelsPass._sympy_error(component, owner)
        if isinstance(component, Expression):
            return component
        if isinstance(component, numbers.Number | np.number):
            return SR(float(component) if isinstance(component, np.floating) else component)
        error_msg = f'A {owner} has the component {component!r}, which is neither a Sage expression nor a number.'
        raise ValidationError(error_msg)

    @staticmethod
    def _to_components(value: object, owner: str) -> list[object] | None:
        """Return the components of a symbolic value, or `None` for a callable."""
        if KernelsPass._is_sympy(value):
            raise KernelsPass._sympy_error(value, owner)
        if isinstance(value, FreeModuleElement):
            return list(value)
        if isinstance(value, Expression | numbers.Number | np.number):
            return [value]
        if isinstance(value, list | tuple | np.ndarray):
            return list(np.ravel(np.asarray(value, dtype=object)))
        if callable(value):
            return None
        error_msg = (
            f'A {owner} has the value {value!r}, which is neither a Sage expression, a number, a sequence, nor a '
            'callable.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _symbolic_components(value: object, size: int, owner: str) -> list[Expression] | None:
        """Return the symbolic components of a law value, or `None` when the value is a callable.

        Args:
            value:
                The law value.

            size:
                The number of components of the law target. A scalar value is broadcast to every component.

            owner:
                The law and the path of the component that returned it, such as `RateLaw law of hunt`, used in error
                messages.

        Returns:
            components:
                The symbolic components, or `None` for a callable.

        Raises:
            ValidationError:
                The value is a SymPy expression, is of an unknown type, or has the wrong size.
        """
        components = KernelsPass._to_components(value, owner)
        if components is None:
            return None
        expressions = [KernelsPass._to_expression(component, owner) for component in components]
        if len(expressions) == 1 and size > 1:
            expressions *= size
        if len(expressions) != size:
            error_msg = f'A {owner} has a value with {len(expressions)} components for a target with {size}.'
            raise ValidationError(error_msg)
        return expressions
