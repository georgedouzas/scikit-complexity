"""Lower the kernels of a problem into the numeric functions of its form on the chosen backend."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import SR

from ....core.base import BackendWarning, BaseLaw, Expression, Form, Layout, NumericDomain, Vector, wrap_angle
from ....core.laws import JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.problems import (
    Accelerated,
    BaseBackend,
    Kernel,
    Lowered,
    Lowering,
    Output,
    ReductionSchedule,
    Source,
    VectorFunction,
    resolve_backend,
)
from ....core.problems._problem import real_components_of
from .._compilation import Compilation
from .._declarations import Declarations, member_count
from .._forms import is_stepwise
from .._pipeline import BasePass
from .._stochastic import DrawLowering, stochastic_update


def _sums_over_members(kernel: Kernel, item: Output) -> bool:
    """Check whether an output sums over the pairs or the members of a group."""
    if kernel.pairs is not None:
        return True
    return item.target is not None and item.target.source is Source.SINGLE and kernel.rows > 1


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
            if item.expression is None or _sums_over_members(kernel, item):
                excluded.add(key)
                continue
            contributions[key].append(item.expression)
    equations, symbols = {}, {}
    for block in declarations.blocks.values():
        for variable, entry in block.entries.items():
            if entry not in declarations.states:
                continue
            components = block.symbols[variable]
            for column, item in enumerate(real_components_of(components)):
                key = (entry.name, column)
                if key in excluded:
                    continue
                default = item if form == Form.DISCRETE else SR(0)
                terms = contributions.get(key)
                name = entry.generic_names[column]
                equations[name] = sum(terms[1:], terms[0]) if terms else default
                symbols[name] = item
    return equations, symbols


def _wrapping_angles(boundary: VectorFunction | None, states: Layout) -> VectorFunction | None:
    """Return a boundary function that also wraps every angle state into `(-pi, pi]`."""
    indices = np.concatenate(
        [np.zeros(0, dtype=np.intp)]
        + [
            np.arange(entry.start, entry.start + entry.size)
            for entry in states.entries
            if entry.domain is NumericDomain.ANGLE
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


def _chosen_backend(compilation: Compilation, form: Form, members: int) -> BaseBackend:
    """Return the backend that lowers a form, NumPy for a stepwise form, warning when it is not the one asked for."""
    requested = resolve_backend(compilation.backend, members)
    if not is_stepwise(form):
        return requested
    chosen = resolve_backend('numpy', members)
    if compilation.backend != 'auto' and requested.name != chosen.name:
        message = (
            f'The {form.value} form runs on the {chosen.name} backend, because its solvers evaluate the kernels '
            f'step by step on {chosen.name} arrays, not on the {requested.name} backend asked for.'
        )
        warnings.warn(message, BackendWarning, stacklevel=2)
    return chosen


def _backends_of(form: Form, kernels: Mapping[type, list[Kernel]], name: str, lowering: Lowering) -> dict[str, str]:
    """Return the backend that runs the form and every law kind with kernels, with the backend's own record."""
    backends = {form.name: name}
    backends.update({str(cast('type[BaseLaw]', kind).law_kind): name for kind, items in kernels.items() if items})
    backends.update(lowering.backends)
    return backends


def _lower(compilation: Compilation) -> dict[str, object]:
    """Return the numeric functions and equations of a form, with the update of a stochastic map taking a generator."""
    form = cast('Form', compilation.form)
    lowered = cast('Lowered', compilation.kernels)
    declarations = cast('Declarations', compilation.layout)
    reductions = cast('ReductionSchedule', compilation.reductions)
    drawing = cast('DrawLowering', compilation.drawing)
    backend = _chosen_backend(compilation, form, member_count(list(declarations.blocks.values())))
    kernels, size = lowered.kernels, declarations.ends[True]
    draws = drawing.draws
    lowering = cast(
        'Lowering',
        backend.prepare(form, size).run(kernels, lowered.jump_changes, reductions=reductions, draws=draws),
    )
    functions: dict[str, object] = vars(lowering).copy()
    if draws is not None and functions.get('update') is not None:
        functions['update'] = stochastic_update(cast('Callable[..., Vector]', functions['update']), draws)
    accelerated = cast('Accelerated | None', functions.get('accelerated'))
    if draws is not None and draws.sampled and accelerated is not None:
        functions['accelerated'] = replace(accelerated, iterate=None)
    functions['backend'] = backend.name
    functions['backends'] = _backends_of(form, kernels, backend.name, lowering)
    functions['draws'] = draws
    main = {
        Form.ODE: kernels[RateLaw],
        Form.FRACTIONAL: kernels[RateLaw],
        Form.SDE: kernels[RateLaw] + kernels[NoiseLaw],
        Form.JUMP: kernels[JumpLaw],
    }.get(form, kernels[UpdateLaw])
    equation_kernels = {Form.DISCRETE: kernels[UpdateLaw], Form.JUMP: []}.get(form, kernels[RateLaw])
    equations, symbols = _equations_of(equation_kernels, declarations, form)
    functions.update(
        equations=equations,
        equation_symbols=symbols,
        has_callables=any(item.expression is None for kernel in main for item in kernel.outputs),
        has_group_sums=any(_sums_over_members(kernel, item) for kernel in main for item in kernel.outputs),
    )
    states = Layout(declarations.states)
    functions['boundary'] = _wrapping_angles(cast('VectorFunction | None', functions.get('boundary')), states)
    return functions


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
        compilation.functions = _lower(compilation)
