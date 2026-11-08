"""Pass that adds the stable step, the linear terms, the finite volumes, the fixed nodes, and the observed fields."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from sage.all__sagemath_symbolics import SR, Expression

from ....components.elements import FieldElement
from ....errors import PipelineError
from ....specs import LayoutSpec, ObservableSpec
from ...problems._kernel_spec import KernelSpec
from ...problems._law_kernel import LawKernel
from ...problems._output_spec import OutputSpec
from .._base import BaseCompiler, BasePass
from .._compilation import Compilation
from .._field_lowering import FieldLowering


class FieldProblemPass(BasePass):
    """Pass that adds the stable step, the linear terms, the finite volumes, the fixed nodes, and the observed fields.

    An observable of a field whose expression applies field operators, such as `inverse_helmholtz` or `gradient`, is
    lowered like a law of the field, into a kernel that gives its value at every site.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'limits', 'volume_limits', 'linear', 'volumes', 'constrained', 'layout', 'specs')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the discretizations of the fields to the problem."""
        compilation = self.compilation_
        problem = Compilation._required(compilation.problem, 'problem')
        constrained, limits, volume_limits = compilation.constrained, compilation.limits, compilation.volume_limits
        linear, volumes = compilation.linear, compilation.volumes
        if constrained is None or limits is None or volume_limits is None or linear is None or volumes is None:
            error_msg = 'FieldProblemPass reads the fields of the field functions, which no pass wrote.'
            raise PipelineError(error_msg)
        indices, values = constrained
        initial_state = problem.initial_state.copy()
        initial_state[indices] = values
        compilation.problem = replace(
            problem,
            initial_state=initial_state,
            stable_step=min(limits, default=None),
            diffusive_limits=tuple(volume_limits),
            linear=tuple(linear),
            volumes=volumes,
            constrained=indices,
            observed=self._observed(compilation),
        )

    @staticmethod
    def _observed(compilation: Compilation) -> dict[str, LawKernel]:
        """Return the kernel of every observable of a field whose expression applies field operators, by its path."""
        specs = {
            spec.owner: spec
            for spec, block in Compilation._required(compilation.specs, 'specs')
            if spec.owner == block.node.path and isinstance(block.node.component, FieldElement)
        }
        kernels = {}
        for scope, observable in BaseCompiler._observables_of(compilation.system):
            path = f'{scope}/{observable.label}' if scope else observable.label
            group = f'{scope}/{observable.group}' if scope else observable.group
            spec = specs.get(group or '')
            if spec is not None:
                kernel = FieldProblemPass._observed_kernel(compilation, spec, path, observable)
                if kernel is not None:
                    kernels[path] = kernel
        return kernels

    @staticmethod
    def _observed_kernel(
        compilation: Compilation,
        spec: KernelSpec,
        path: str,
        observable: ObservableSpec,
    ) -> LawKernel | None:
        """Return the kernel of an observable of a field, or `None` when its expression applies no field operator."""
        expression = observable.expression(spec.namespace) if callable(observable.expression) else observable.expression
        if not isinstance(expression, Expression) or not FieldLowering._has_operators(expression):
            return None
        observed = replace(spec, owner=f'observable {path}')
        lowering = FieldLowering(observed).prepare(compilation)
        lowered = lowering._lower(expression)
        _, added = lowering.added_.get(id(observed), (observed, {}))
        declarations = Compilation._required(compilation.layout, 'layout')
        return LawKernel(
            path,
            declarations.time,
            {**observed.references, **added},
            spec.rows,
            LayoutSpec(tuple(declarations.states)),
            layout_parameters=LayoutSpec(tuple(declarations.parameters)),
            outputs=[OutputSpec(None, SR(lowered))],
        ).prepare()
