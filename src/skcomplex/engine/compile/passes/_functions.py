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

from ....core.errors import BackendWarning
from ....core.laws import JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.types import Vector
from ...backends import BaseBackend, Lowering, NumpyBackend, resolve_backend
from ...problems import (
    Accelerated,
    Draws,
    Form,
    Kernel,
    Layout,
    Lowered,
    Output,
    Problem,
    ReductionSchedule,
    Source,
    VectorFunction,
    wrap_angle,
)
from .._compilation import Compilation
from .._declarations import Declarations, _member_count
from .._forms import STEPWISE_FORMS
from .._pipeline import BasePass
from .._stochastic import DrawLowering


class FunctionsPass(BasePass):
    """Pass that lowers the kernels into the numeric functions and the equations of the form."""

    reads = ('kernels', 'form', 'layout', 'reductions', 'drawing')
    writes = ('functions',)

    def _run(self, compilation: Compilation) -> None:
        """Lower the kernels into numeric functions.

        Args:
            compilation:
                The compilation state.
        """
        compilation.functions = self._lower(compilation)

    @staticmethod
    def _sums_over_members(kernel: Kernel, item: Output) -> bool:
        """Check whether an output sums over the pairs or the members of a group."""
        if kernel.pairs is not None:
            return True
        return item.target is not None and item.target.source is Source.SINGLE and kernel.rows > 1

    @staticmethod
    def _equations_of(
        kernels: list[Kernel],
        declarations: Declarations,
        form: Form,
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
                for column, item in enumerate(Problem._real_components_of(components)):
                    key = (entry.name, column)
                    if key in excluded:
                        continue
                    default = item if form == Form.DISCRETE else SR(0)
                    terms = contributions.get(key)
                    name = entry.generic_names[column]
                    equations[name] = sum(terms[1:], terms[0]) if terms else default
                    symbols[name] = item
        return equations, symbols

    @staticmethod
    def _wrapping_angles(boundary: VectorFunction | None, states: Layout) -> VectorFunction | None:
        """Return a boundary function that also wraps every angle state into `(-pi, pi]`."""
        indices = np.concatenate(
            [np.zeros(0, dtype=np.intp)]
            + [
                np.arange(entry.start, entry.start + entry.size) for entry in states.entries if entry.period is not None
            ],
        )
        if not indices.size:
            return boundary

        def wrapped(t: float, x: Vector, p: Vector) -> Vector:
            """Apply the boundary laws, then wrap the angle states."""
            state = x if boundary is None else boundary(t, x, p)
            state = state.copy()
            state[indices] = wrap_angle(state[indices])
            return state

        return wrapped

    @staticmethod
    def _chosen_backend(compilation: Compilation, form: Form, members: int) -> BaseBackend:
        """Return the backend that lowers a form, NumPy for a stepwise form, warning when it is not the one asked."""
        requested = resolve_backend(compilation.backend, members)
        if not FunctionsPass._is_stepwise(form):
            return requested
        chosen = NumpyBackend()
        if compilation.backend is not None and requested.name != chosen.name:
            message = (
                f'The {form.value} form runs on the {chosen.name} backend, because its solvers evaluate the kernels '
                f'step by step on {chosen.name} arrays, not on the {requested.name} backend asked for.'
            )
            warnings.warn(message, BackendWarning, stacklevel=2)
        return chosen

    @staticmethod
    def _backends_of(form: Form, kernels: Mapping[type, list[Kernel]], name: str, lowering: Lowering) -> dict[str, str]:
        """Return the backend that runs the form and every law kind with kernels, with the backend's own record."""
        backends = {form.name: name}
        backends.update({cast('type', kind).__name__: name for kind, items in kernels.items() if items})
        backends.update(lowering.backends)
        return backends

    @staticmethod
    def _lower(compilation: Compilation) -> dict[str, object]:
        """Return the numeric functions and equations of a form, the update of a stochastic map taking a generator."""
        form = cast('Form', compilation.form)
        lowered = cast('Lowered', compilation.kernels)
        declarations = cast('Declarations', compilation.layout)
        reductions = cast('ReductionSchedule', compilation.reductions)
        drawing = cast('DrawLowering', compilation.drawing)
        backend = FunctionsPass._chosen_backend(compilation, form, _member_count(list(declarations.blocks.values())))
        kernels, size = lowered.kernels, declarations.ends[True]
        draws = drawing.draws
        lowering = cast(
            'Lowering',
            backend.prepare(form, size).run(kernels, lowered.jump_changes, reductions=reductions, draws=draws),
        )
        functions: dict[str, object] = vars(lowering).copy()
        if draws is not None and functions.get('update') is not None:
            functions['update'] = FunctionsPass._stochastic_update(
                cast('Callable[..., Vector]', functions['update']), draws,
            )
        accelerated = cast('Accelerated | None', functions.get('accelerated'))
        if draws is not None and draws.sampled and accelerated is not None:
            functions['accelerated'] = replace(accelerated, iterate=None)
        functions['backend'] = backend.name
        functions['backends'] = FunctionsPass._backends_of(form, kernels, backend.name, lowering)
        functions['draws'] = draws
        main = {
            Form.ODE: kernels[RateLaw],
            Form.FRACTIONAL: kernels[RateLaw],
            Form.SDE: kernels[RateLaw] + kernels[NoiseLaw],
            Form.JUMP: kernels[JumpLaw],
        }.get(form, kernels[UpdateLaw])
        equation_kernels = {Form.DISCRETE: kernels[UpdateLaw], Form.JUMP: []}.get(form, kernels[RateLaw])
        equations, symbols = FunctionsPass._equations_of(equation_kernels, declarations, form)
        functions.update(
            equations=equations,
            equation_symbols=symbols,
            has_callables=any(item.expression is None for kernel in main for item in kernel.outputs),
            has_group_sums=any(
                FunctionsPass._sums_over_members(kernel, item) for kernel in main for item in kernel.outputs
            ),
        )
        states = Layout(declarations.states)
        functions['boundary'] = FunctionsPass._wrapping_angles(
            cast('VectorFunction | None', functions.get('boundary')),
            states,
        )
        return functions

    @staticmethod
    def _is_stepwise(form: Form) -> bool:
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
    def _stochastic_update(update: Callable[..., Vector], draws: Draws) -> Callable[..., Vector]:
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

        def stochastic(t: float, x: Vector, p: Vector, rng: np.random.Generator) -> Vector:
            extended = draws.extend(p)
            draws.fill(extended, rng, t, x)
            return update(t, x, extended)

        return stochastic
