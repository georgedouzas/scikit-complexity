"""Lower the laws of every kernel spec into kernels, one per law kind and component."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numbers
from collections import defaultdict
from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.errors import ValidationError
from ....core.laws import BaseLaw, BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.laws.utils import Call
from ....core.specs import WienerProcessSpec
from ...problems import Kernel, Layout, Lowered, NoiseProcess, Output, Reference, Source, components_of, symbols_of
from .._base import BasePass
from .._census_record import ProcessKey
from .._compilation import Laws
from .._declarations import Declarations
from .._kernel_spec import KernelSpec
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
        compilation.kernels = self._lower_laws(
            cast('Laws', compilation.reduced),
            cast('Declarations', compilation.layout),
            compilation.outputs,
        )

    @staticmethod
    def _outputs_of(value: object, targets: list[Reference | None], owner: str) -> list[Output]:
        """Return the outputs of one law value for its target components, with the law and component as owner."""
        if isinstance(value, Call):
            arguments = tuple(
                item
                for argument in value.arguments
                for item in KernelsPass._symbolic_components(argument, 1, owner) or []
            )
            if len(arguments) != len(value.arguments):
                error_msg = f'A Call in a {owner} has an argument that is not a symbolic expression.'
                raise ValidationError(error_msg)
            return [
                Output(target, function=value.function, arguments=arguments, component=index, components=len(targets))  # type: ignore[arg-type]
                for index, target in enumerate(targets)
            ]
        expressions = KernelsPass._symbolic_components(value, len(targets), owner)
        if expressions is None:
            return [
                Output(target, function=value, component=index, components=len(targets))  # type: ignore[arg-type]
                for index, target in enumerate(targets)
            ]
        return [Output(target, expression) for target, expression in zip(targets, expressions, strict=True)]

    @staticmethod
    def _check_assignments(
        spec: KernelSpec,
        law: BaseLaw,
        targets: list[Reference],
        assigned: set[tuple[str, int]],
        *,
        once: bool,
    ) -> None:
        """Raise an error for an update or boundary law that sums over members or assigns a component twice."""
        kind = type(law).__name__
        for target in targets:
            if target.source in {Source.FIRST, Source.SECOND}:
                error_msg = (
                    f'A {kind} law of {spec.owner} sets {target.entry.name} at an end of every edge, which several '
                    'edges share. Set the edge member, or reduce over the neighbours in the laws of the nodes.'
                )
                raise ValidationError(error_msg)
            if target.source is Source.SINGLE and spec.rows > 1 and not once:
                error_msg = f'A {kind} law of {spec.owner} targets a single element from a group, which has no meaning.'
                raise ValidationError(error_msg)
            key = (target.entry.name, target.column)
            if key in assigned:
                error_msg = f'A {kind} law of {spec.owner} sets {target.entry.name} a second time. Each takes one law.'
                raise ValidationError(error_msg)
            assigned.add(key)

    @staticmethod
    def _is_once(spec: KernelSpec, law: BaseLaw) -> bool:
        """Check whether a law of a group's interaction sets single elements from reductions alone, so it runs once."""
        if (
            spec.rows <= 1
            or spec.pairs is not None
            or not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw)
        ):
            return False
        if not all(
            reference.source is Source.SINGLE
            for reference in KernelSpec._target_references(law._get_targets()[0], spec)
        ):
            return False
        expressions = [item for item in components_of(law.value) if isinstance(item, Expression)]
        sources = {spec.references[item].source for item in symbols_of(expressions) if item in spec.references}
        return Source.REDUCED in sources and Source.MEMBER not in sources

    @staticmethod
    def _single_references(spec: KernelSpec) -> dict[Expression, Reference]:
        """Return the references of a spec without its group members, for laws that run once."""
        return {
            item: reference
            for item, reference in spec.references.items()
            if reference.source is not Source.MEMBER and reference.entry.members is None
        }

    @staticmethod
    def _law_outputs(
        spec: KernelSpec,
        law: RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw,
        targets: list[Reference],
        declarations: Declarations,
        lowered: dict[int, list[Output]],
    ) -> list[Output]:
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
        outputs = KernelsPass._outputs_of(law.value, targets, f'{type(law).__name__} of {spec.owner}')  # type: ignore[arg-type]
        if isinstance(law, NoiseLaw) and law.process is not None:
            for item, key in zip(outputs, NoisePass._process_keys(spec, law, declarations), strict=True):
                item.process = KernelsPass._noise_process(key, declarations.wieners)
        return outputs

    @staticmethod
    def _lower_laws(specs: Laws, declarations: Declarations, lowered_outputs: dict[int, list[Output]]) -> Lowered:
        """Return the kernels of every law kind, with the law of every output."""
        states, parameters = Layout(declarations.states), Layout(declarations.parameters)
        lowered = Lowered()
        assigned: dict[type, set[tuple[str, int]]] = defaultdict(set)
        for spec, laws in specs:
            outputs: dict[tuple[type, bool], list[Output]] = defaultdict(list)
            for law in laws:
                if isinstance(law, _KERNEL_FREE_LAWS):
                    continue
                if isinstance(law, JumpLaw):
                    output = KernelsPass._outputs_of(law.rate, [None], f'JumpLaw of {spec.owner}')[0]
                    lowered.jump_changes[id(output)] = [
                        (KernelSpec._target_references(target, spec)[0], float(increment))
                        for target, increment in law.changes.items()
                    ]
                    lowered.laws[id(output)] = (spec, law)
                    outputs[(JumpLaw, False)].append(output)
                    continue
                if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
                    error_msg = f'A {type(law).__name__} of {spec.owner} reached lowering before it was rewritten.'
                    raise ValidationError(error_msg)
                targets = KernelSpec._target_references(law._get_targets()[0], spec)
                once = KernelsPass._is_once(spec, law)
                if isinstance(law, _ASSIGNED_KINDS):
                    KernelsPass._check_assignments(spec, law, targets, assigned[type(law)], once=once)
                law_outputs = KernelsPass._law_outputs(spec, law, targets, declarations, lowered_outputs)
                lowered.laws.update({id(item): (spec, law) for item in law_outputs})
                outputs[(KernelsPass._kind_of(law), once)].extend(law_outputs)
            for (kind, once), kind_outputs in outputs.items():
                lowered.kernels[kind].append(
                    Kernel(
                        spec.owner,
                        declarations.time,
                        KernelsPass._single_references(spec) if once else spec.references,
                        1 if once else spec.rows,
                        states,
                        parameters,
                        spec.pairs,
                        kind_outputs,
                    ).check(),
                )
        return lowered

    @staticmethod
    def _covariance_factor(covariance: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return a factor `L` with `L L^T = C` of a covariance, by Cholesky or, for a singular one, by eigenvalues."""
        matrix = np.asarray(covariance, dtype=float)
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            values, vectors = np.linalg.eigh(matrix)
            return vectors * np.sqrt(np.maximum(values, 0.0))

    @staticmethod
    def _noise_process(key: ProcessKey | None, wieners: Mapping[str, WienerProcessSpec]) -> NoiseProcess | None:
        """Return the named Wiener process of a process key, with the covariance factor of a `WienerProcessSpec`."""
        if key is None:
            return None
        kind, owner, item = key
        if kind == 'local':
            return NoiseProcess(key)
        wiener = wieners[owner]
        factor = None if wiener.covariance is None else KernelsPass._covariance_factor(wiener.covariance)
        return NoiseProcess(owner, item, wiener.dimension, factor)

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
