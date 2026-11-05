"""Lower the kernels of a problem into the numeric functions of its form on the chosen backend."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ....core.laws import JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....errors import BackendWarning, ValidationError
from ....specs import (
    DEFAULT_PHASE,
    EntrySpec,
    LayoutSpec,
    LoweredFunctionsSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    RunStateSpec,
)
from ....types import Array, FormKind, Indices, VectorFunction
from ..._utils import symbols_of
from ...backends import BaseBackend, NumpyBackend
from ...problems import (
    DeclarationsSpec,
    ExpressionPrinter,
    KernelSpec,
    LoweredKernelsSpec,
    OutputSpec,
    ProblemSpec,
    ReductionScheduleSpec,
)
from ...problems._draw_kernel import DrawKernel
from ...problems._law_kernel import LawKernel
from ...problems._problem_functions_spec import _ProblemFunctionsSpec
from ...problems._sweep_spec import _SweepSpec
from ...types import ReadKind
from .._algebraic_law import AlgebraicLaw
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
        """Return the symbolic equations and their symbols, without callables, group sums, or combined updates."""
        contributions: dict[tuple[str, int], list[Expression]] = defaultdict(list)
        excluded: set[tuple[str, int]] = set()
        for kernel in kernels:
            for item in kernel.outputs:
                if item.target is None:
                    continue
                key = (item.target.entry.name, item.target.column)
                if item.expression is None or item.combined or FunctionsPass._sums_over_members(kernel, item):
                    excluded.add(key)
                    continue
                contributions[key].append(item.expression)
        equations, symbols = {}, {}
        for block in declarations.blocks.values():
            for variable, entry in block.entries.items():
                if entry not in declarations.states:
                    continue
                components = block.symbols[variable]
                for column, component in enumerate(ProblemSpec._real_components_of(components)):
                    key = (entry.name, column)
                    if key in excluded:
                        continue
                    default = component if form in (FormKind.DISCRETE, FormKind.STATIC) else SR(0)
                    terms = contributions.get(key)
                    name = entry.generic_names[column]
                    equations[name] = sum(terms[1:], terms[0]) if terms else default
                    symbols[name] = component
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
                f'The {form} form runs on the {name} backend, because its solvers evaluate the kernels '
                f'step by step on {name} arrays, not on the {asked} backend asked for.'
            )
            warnings.warn(message, BackendWarning, stacklevel=2)
        return chosen

    @staticmethod
    def _backends_of(
        form: FormKind,
        kernels: Mapping[type, list[LawKernel]],
        name: str,
        lowering: LoweredFunctionsSpec,
    ) -> dict[str, str]:
        """Return the backend that runs the form and every law kind with kernels, with the backend's own record."""
        backends = {form.name: name}
        backends.update({kind.__name__: name for kind, items in kernels.items() if items})
        backends.update(lowering.backends)
        return backends

    @staticmethod
    def _lower(compilation: Compilation) -> _ProblemFunctionsSpec:
        """Return the numeric functions and equations of a form, the update of a stochastic map taking a generator."""
        form = Compilation._required(compilation.form, 'form')
        lowered = Compilation._required(compilation.kernels, 'kernels')
        declarations = Compilation._required(compilation.layout, 'layout')
        reductions = Compilation._required(compilation.reductions, 'reductions')
        draws = compilation.draws
        backend = FunctionsPass._chosen_backend(
            compilation,
            form,
            declarations.members,
        )
        kernels, size = lowered.kernels, declarations.ends[True]
        complete = replace(lowered, form=form, size=size, reductions=reductions, draws=draws)
        lowering = backend.prepare().run(complete).functions_
        update: Callable[..., Array] | None = lowering.update
        phases = FunctionsPass._phases_of(compilation, lowered) if form == FormKind.DISCRETE else None
        sweeps = FunctionsPass._sweeps_of(lowered, declarations.time)
        if phases is not None:
            update = FunctionsPass._phased_update(
                [
                    (phase, FunctionsPass._phase_update(backend, complete, items, declarations.time))
                    for phase, items in phases
                ],
            )
        elif sweeps and update is not None:
            update = FunctionsPass._ordered_update(update, sweeps)
        if draws is not None and update is not None:
            update = FunctionsPass._stochastic_update(update, draws)
        accelerated = lowering.accelerated
        if (
            (draws is not None and (draws.sampled or draws.orders)) or sweeps or phases is not None
        ) and accelerated is not None:
            accelerated = replace(accelerated, iterate=None)
        main = {
            FormKind.ODE: kernels[RateLaw],
            FormKind.FRACTIONAL: kernels[RateLaw],
            FormKind.SDE: kernels[RateLaw] + kernels[NoiseLaw],
            FormKind.JUMP: kernels[JumpLaw],
        }.get(form, kernels[UpdateLaw])
        equation_kernels = {
            FormKind.DISCRETE: kernels[UpdateLaw],
            FormKind.STATIC: kernels[UpdateLaw] or kernels[RateLaw],
            FormKind.JUMP: [],
        }.get(form, kernels[RateLaw])
        equations, symbols = (
            FunctionsPass._equations_of(equation_kernels, declarations, form)
            if phases is None
            else FunctionsPass._phased_equations([items for _, items in phases], declarations, form)
        )
        states = LayoutSpec(tuple(declarations.states))
        return _ProblemFunctionsSpec(
            equations=equations,
            equation_symbols=symbols,
            has_callables=any(item.expression is None for kernel in main for item in kernel.outputs)
            or any(not isinstance(value, float) for changes in lowered.jump_changes.values() for _, value in changes),
            has_group_sums=any(
                FunctionsPass._sums_over_members(kernel, item) for kernel in main for item in kernel.outputs
            ),
            drift=lowering.drift,
            diffusion=lowering.diffusion,
            noise_rows=lowering.noise_rows,
            noise_columns=lowering.noise_columns,
            n_processes=lowering.n_processes,
            noise_blocks=lowering.noise_blocks,
            propensities=lowering.propensities,
            jumps=lowering.jumps,
            update=update,
            boundary=FunctionsPass._wrapping_angles(lowering.boundary, states),
            algebraic=FunctionsPass._algebraic_function(lowered, reductions),
            draws=draws,
            backend=backend._get_name(),
            backends=FunctionsPass._backends_of(form, kernels, backend._get_name(), lowering),
            accelerated=accelerated,
        )

    @staticmethod
    def _algebraic_function(lowered: LoweredKernelsSpec, reductions: ReductionScheduleSpec) -> VectorFunction | None:
        """Return the function that sets the algebraic states of a state vector, or `None` for a system without them."""
        kernels = lowered.kernels.get(AlgebraicLaw)
        if not kernels:
            return None
        return NumpyBackend._wrapped(
            BaseBackend._assigning_function(kernels),
            AlgebraicLaw,
            reductions,
            lowered.kernels,
        )

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
    def _key_function(spec: KernelSpec, order: object, time: Expression) -> Callable[[float, Array, Array], Array]:
        """Return the function of the order of the members of an update law at a state."""
        expression = SR(order)
        arguments = sorted(symbols_of([expression]) - {time}, key=str)
        unknown = [item for item in arguments if item not in spec.references]
        if unknown:
            error_msg = (
                f'The order of an update law of {spec.owner} reads {", ".join(map(str, unknown))}, which the namespace '
                'does not hold. Order the members by their states, parameters, and draws.'
            )
            raise ValidationError(error_msg)
        references = [spec.references[item] for item in arguments]
        function = ExpressionPrinter(time=time, compiled=True).check().prepare(arguments).run([expression]).code_
        if not callable(function):
            error_msg = f'The printer returns no function of the order of an update law of {spec.owner}.'
            raise TypeError(error_msg)
        rows = spec.rows

        def key(t: float, x: Array, q: Array) -> Array:
            values = function(t, *(reference.column_of(x, q) for reference in references))
            if not isinstance(values, list):
                error_msg = f'The function of the order of an update law of {spec.owner} returns no list of values.'
                raise TypeError(error_msg)
            return np.broadcast_to(np.asarray(values[0], dtype=float), (rows,))

        return key

    @staticmethod
    def _sweeps_of(lowered: LoweredKernelsSpec, time: Expression) -> list[_SweepSpec]:
        """Return the update laws whose members update one after the other, with the indices each member sets."""
        found: dict[int, tuple[KernelSpec, UpdateLaw, list[Indices]]] = {}
        for kernel in lowered.kernels.get(UpdateLaw, []):
            for item in kernel.outputs:
                spec, law = lowered.laws[id(item)]
                if not isinstance(law, UpdateLaw) or law.order is None or item.target is None:
                    continue
                if not isinstance(spec, KernelSpec):
                    error_msg = f'The update law {law.label!r} has no kernel spec to order its members by.'
                    raise TypeError(error_msg)
                target = item.target
                indices: Indices = (
                    target.flat_indices
                    if target.source is ReadKind.MEMBER
                    else np.asarray([target.index], dtype=np.intp)
                )
                found.setdefault(id(law), (spec, law, []))[2].append(indices)
        return [
            _SweepSpec(np.stack(columns, axis=1), FunctionsPass._key_function(spec, law.order, time))
            for spec, law, columns in found.values()
        ]

    @staticmethod
    def _ordered_update(update: Callable[..., Array], sweeps: list[_SweepSpec]) -> Callable[..., Array]:
        """Return an update whose ordered laws set their members one after the other, each reading those before it.

        Every other law updates at once from the state at the start of the step.

        Args:
            update:
                The update `F(t, x, q)` of every update law at once.

            sweeps:
                The ordered update laws, with the indices every member sets and the function of their order.

        Returns:
            update:
                The update `F(t, x, q)`.
        """
        written = np.concatenate([sweep.indices.ravel() for sweep in sweeps])

        def ordered(t: float, x: Array, q: Array) -> Array:
            swept = x.copy()
            for sweep in sweeps:
                for row in np.argsort(sweep.key(t, swept, q), kind='stable'):
                    indices = sweep.indices[row]
                    swept[indices] = update(t, swept, q)[indices]
            following = update(t, x, q)
            following[written] = swept[written]
            return following

        return ordered

    @staticmethod
    def _phases_of(
        compilation: Compilation,
        lowered: LoweredKernelsSpec,
    ) -> list[tuple[PhaseSpec, list[LawKernel]]] | None:
        """Return the phases of a step in order with the update kernels of each, or `None` for the default phase alone.

        The phases are those of the `PhasesScheduleSpec` of the discrete time of the system and its nested systems, with
        the default phase first unless a schedule places it, and a phase without update laws is left out.

        Args:
            compilation:
                The compilation of the system.

            lowered:
                The lowered kernels of the system.

        Returns:
            phases:
                The phases in order with the update kernels of each, or `None` for the default phase alone.

        Raises:
            ValidationError:
                An update law runs in a phase the schedule does not declare, or a phase of update laws repeats or has an
                activation other than the default.
        """
        declared = FunctionsPass._declared_phases(compilation.system)
        kernels: dict[str, list[LawKernel]] = defaultdict(list)
        for kernel in lowered.kernels.get(UpdateLaw, []):
            _, law = lowered.laws[id(kernel.outputs[0])]
            kernels[(law.phase if isinstance(law, UpdateLaw) else None) or DEFAULT_PHASE].append(kernel)
        names = [phase.name for phase in declared]
        if set(kernels) <= {DEFAULT_PHASE} and DEFAULT_PHASE not in names:
            return None
        plan = declared if DEFAULT_PHASE in names else [PhaseSpec(DEFAULT_PHASE), *declared]
        unknown = sorted(set(kernels) - {phase.name for phase in plan})
        if unknown:
            error_msg = (
                f'Update laws run in the phases {", ".join(unknown)}, which the system does not declare. Declare them '
                'with `DiscreteTime(schedule=PhasesScheduleSpec(...))`. Declared phases are: '
                f'{", ".join(names) or "none"}.'
            )
            raise ValidationError(error_msg)
        for phase in plan:
            if phase.name in kernels and phase.repeat != 1:
                error_msg = (
                    f'The phase {phase.name!r} repeats {phase.repeat} times, but update laws run once per phase, since '
                    'they draw once per step. Give it one round, or move the update laws to a phase of their own.'
                )
                raise ValidationError(error_msg)
            if phase.name in kernels and phase.activation != PhaseSpec.activation:
                error_msg = (
                    f'The phase {phase.name!r} has the activation {phase.activation!r}, but update laws change all '
                    'members at once, and their phases take only a period and an offset. Remove the activation, or '
                    'order the members with `UpdateLaw(order=...)`.'
                )
                raise ValidationError(error_msg)
        return [(phase, kernels[phase.name]) for phase in plan if phase.name in kernels]

    @staticmethod
    def _declared_phases(system: System) -> list[PhaseSpec]:
        """Return the phases of the schedules of a system and its nested systems, the first of every name kept."""
        time = system.time
        schedule = time.check().prepare().schedule_ if isinstance(time, DiscreteTime) else None
        phases = list(schedule.phases) if isinstance(schedule, PhasesScheduleSpec) else []
        for nested in system.systems:
            phases += FunctionsPass._declared_phases(nested)
        kept: dict[str, PhaseSpec] = {}
        for phase in phases:
            kept.setdefault(phase.name, phase)
        return list(kept.values())

    @staticmethod
    def _phase_update(
        backend: BaseBackend,
        complete: LoweredKernelsSpec,
        kernels: list[LawKernel],
        time: Expression,
    ) -> Callable[..., Array]:
        """Return the update of the kernels of one phase, whose ordered laws set their members one after the other."""
        phase = replace(complete, kernels=defaultdict(list, {UpdateLaw: kernels}))
        update = backend.prepare().run(phase).functions_.update
        if update is None:
            error_msg = 'The backend returns no update of a phase of a discrete map.'
            raise TypeError(error_msg)
        sweeps = FunctionsPass._sweeps_of(phase, time)
        return FunctionsPass._ordered_update(update, sweeps) if sweeps else update

    @staticmethod
    def _phased_update(phases: list[tuple[PhaseSpec, Callable[..., Array]]]) -> Callable[..., Array]:
        """Return an update that runs the phases due in a step in order, each reading the states the one before set.

        A phase runs in the steps `k`, counted from one, for which `k - offset` is a multiple of its period.

        Args:
            phases:
                The phases in order, with the update `F(t, x, q)` of each.

        Returns:
            update:
                The update `F(t, x, q)`.
        """

        def phased(t: float, x: Array, q: Array) -> Array:
            state, step = x.copy(), round(t) + 1
            for phase, update in phases:
                if not (step - phase.offset) % phase.period:
                    state = update(t, state, q)
            return state

        return phased

    @staticmethod
    def _phased_equations(
        phases: list[list[LawKernel]],
        declarations: DeclarationsSpec,
        form: FormKind,
    ) -> tuple[dict[str, Expression], dict[str, Expression]]:
        """Return the symbolic map of phases that run in every step, the map of every phase composed with those before.

        A state whose value after a phase has no symbolic equation, or reads such a state, is left out.
        """
        known, symbols = FunctionsPass._equations_of([], declarations, form)
        names = {component: name for name, component in symbols.items()}
        for kernels in phases:
            equations, _ = FunctionsPass._equations_of(kernels, declarations, form)
            following = {}
            for name, equation in equations.items():
                expression = SR(equation)
                read = [names[item] for item in expression.variables() if item in names]
                if all(item in known for item in read):
                    following[name] = expression.subs({symbols[item]: known[item] for item in read})
            known = following
        return known, {name: symbols[name] for name in known}

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
