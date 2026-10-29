"""Lower the kernels of a problem into the numeric functions of its form on the chosen backend."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ....core.laws import JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....errors import BackendWarning
from ....specs import AcceleratedSpec, EntrySpec, LayoutSpec, LoweredFunctionsSpec, RunStateSpec
from ....types import Array, FormKind, ReadKind, VectorFunction
from ...backends import BaseBackend, NumpyBackend
from ...problems import DeclarationsSpec, LoweredKernelsSpec, OutputSpec, ProblemSpec, ReductionScheduleSpec
from ...problems._draw_kernel import DrawKernel
from ...problems._law_kernel import LawKernel
from .._base import BaseCompiler, BasePass
from .._compilation import Compilation
from .._forms import STEPWISE_FORMS


class FunctionsPass(BasePass):
    """Pass that lowers the kernels into the numeric functions and the equations of the form."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('kernels', 'form', 'layout', 'reductions', 'draws')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('functions',)

    def _run(self, inputs: object = None) -> None:
        """Lower the kernels into numeric functions."""
        compilation = self.compilation_
        compilation.functions = self._lower(compilation)

    @staticmethod
    def _sums_over_members(kernel: LawKernel, item: OutputSpec) -> bool:
        """Check whether an output sums over the pairs or the members of a group."""
        if kernel.pairs is not None:
            return True
        return item.target is not None and item.target.source is ReadKind.SINGLE and kernel.rows > 1

    @staticmethod
    def _equations_of(
        kernels: list[LawKernel],
        declarations: DeclarationsSpec,
        form: FormKind,
    ) -> tuple[dict[str, Expression], dict[str, Expression]]:
        """Return the symbolic equations and their symbols, leaving out components with callables or group sums."""
        contributions: dict[tuple[str, int], list[Expression]] = defaultdict(list)
        excluded: set[tuple[str, int]] = set()
        for kernel in kernels:
            for item in kernel.outputs:
                key = (item.target.entry.name, item.target.column)  # type: ignore[union-attr]
                if item.expression is None or FunctionsPass._sums_over_members(kernel, item):
                    excluded.add(key)
                    continue
                contributions[key].append(item.expression)
        equations, symbols = {}, {}
        for block in declarations.blocks.values():
            for variable, entry in block.entries.items():
                if entry not in declarations.states:
                    continue
                components = block.symbols[variable]
                for column, item in enumerate(ProblemSpec._real_components_of(components)):
                    key = (entry.name, column)
                    if key in excluded:
                        continue
                    default = item if form == FormKind.DISCRETE else SR(0)
                    terms = contributions.get(key)
                    name = entry.generic_names[column]
                    equations[name] = sum(terms[1:], terms[0]) if terms else default
                    symbols[name] = item
        return equations, symbols

    @staticmethod
    def _wrapping_angles(boundary: VectorFunction | None, states: LayoutSpec) -> VectorFunction | None:
        """Return a boundary function that also wraps every angle state into `(-pi, pi]`."""
        indices = np.concatenate(
            [np.zeros(0, dtype=np.intp)]
            + [
                np.arange(entry.start, entry.start + entry.size) for entry in states.entries if entry.period is not None
            ],
        )
        if not indices.size:
            return boundary

        def wrapped(t: float, x: Array, p: Array) -> Array:
            """Apply the boundary laws, then wrap the angle states."""
            state = x if boundary is None else boundary(t, x, p)
            state = state.copy()
            state[indices] = EntrySpec._wrap_angle(state[indices])
            return state

        return wrapped

    @staticmethod
    def _chosen_backend(compilation: Compilation, form: FormKind, members: int) -> BaseBackend:
        """Return the backend that lowers a form, NumPy for a stepwise form, warning when it is not the one asked."""
        requested = BaseCompiler._resolve_backend(compilation.backend, members)
        if not FunctionsPass._is_stepwise(form):
            return requested
        chosen = NumpyBackend().check()
        name, asked = chosen._get_name(), requested._get_name()
        if compilation.backend is not None and asked != name:
            message = (
                f'The {form.value} form runs on the {name} backend, because its solvers evaluate the kernels '
                f'step by step on {name} arrays, not on the {asked} backend asked for.'
            )
            warnings.warn(message, BackendWarning, stacklevel=2)
        return chosen

    @staticmethod
    def _backends_of(
        form: FormKind, kernels: Mapping[type, list[LawKernel]], name: str, lowering: LoweredFunctionsSpec,
    ) -> dict[str, str]:
        """Return the backend that runs the form and every law kind with kernels, with the backend's own record."""
        backends = {form.name: name}
        backends.update({cast('type', kind).__name__: name for kind, items in kernels.items() if items})
        backends.update(lowering.backends)
        return backends

    @staticmethod
    def _lower(compilation: Compilation) -> dict[str, object]:
        """Return the numeric functions and equations of a form, the update of a stochastic map taking a generator."""
        form = cast('FormKind', compilation.form)
        lowered = cast('LoweredKernelsSpec', compilation.kernels)
        declarations = cast('DeclarationsSpec', compilation.layout)
        reductions = cast('ReductionScheduleSpec', compilation.reductions)
        draws = compilation.draws
        backend = FunctionsPass._chosen_backend(
            compilation, form, declarations.members,
        )
        kernels, size = lowered.kernels, declarations.ends[True]
        complete = replace(lowered, form=form, size=size, reductions=reductions, draws=draws)
        lowering = cast('LoweredFunctionsSpec', backend.prepare().run(complete))
        functions: dict[str, object] = vars(lowering).copy()
        if draws is not None and functions.get('update') is not None:
            functions['update'] = FunctionsPass._stochastic_update(
                cast('Callable[..., Array]', functions['update']), draws,
            )
        accelerated = cast('AcceleratedSpec | None', functions.get('accelerated'))
        if draws is not None and draws.sampled and accelerated is not None:
            functions['accelerated'] = replace(accelerated, iterate=None)
        functions['backend'] = backend._get_name()
        functions['backends'] = FunctionsPass._backends_of(form, kernels, backend._get_name(), lowering)
        functions['draws'] = draws
        main = {
            FormKind.ODE: kernels[RateLaw],
            FormKind.FRACTIONAL: kernels[RateLaw],
            FormKind.SDE: kernels[RateLaw] + kernels[NoiseLaw],
            FormKind.JUMP: kernels[JumpLaw],
        }.get(form, kernels[UpdateLaw])
        equation_kernels = {FormKind.DISCRETE: kernels[UpdateLaw], FormKind.JUMP: []}.get(form, kernels[RateLaw])
        equations, symbols = FunctionsPass._equations_of(equation_kernels, declarations, form)
        functions.update(
            equations=equations,
            equation_symbols=symbols,
            has_callables=any(item.expression is None for kernel in main for item in kernel.outputs),
            has_group_sums=any(
                FunctionsPass._sums_over_members(kernel, item) for kernel in main for item in kernel.outputs
            ),
        )
        states = LayoutSpec(tuple(declarations.states))
        functions['boundary'] = FunctionsPass._wrapping_angles(
            cast('VectorFunction | None', functions.get('boundary')),
            states,
        )
        return functions

    @staticmethod
    def _is_stepwise(form: FormKind) -> bool:
        """Check whether the solvers of a form evaluate its kernels step by step on the NumPy backend.

        Args:
            form:
                The form.

        Returns:
            stepwise:
                Whether the form is stepwise.
        """
        return form in STEPWISE_FORMS.values()

    @staticmethod
    def _stochastic_update(update: Callable[..., Array], draws: DrawKernel) -> Callable[..., Array]:
        """Return the update of a stochastic map, which draws its values from a generator before every step.

        Args:
            update:
                The update `F(t, x, q)` of the extended parameter vector.

            draws:
                The draws of the map.

        Returns:
            update:
                The update `F(t, x, p, rng)`.
        """

        def stochastic(t: float, x: Array, p: Array, rng: np.random.Generator) -> Array:
            extended = draws.extend(p)
            draws.run(RunStateSpec(t, x, extended, rng))
            return update(t, x, extended)

        return stochastic
