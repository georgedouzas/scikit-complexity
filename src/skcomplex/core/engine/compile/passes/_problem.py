"""Assemble the problem of a compilation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....components.laws import BaseLaw, CrossingLaw, JumpLaw, RateLaw
from ....components.spaces import LatticeSpace
from ....components.specs import CompiledElementSpec, CompiledInteractionSpec, NodeSpec
from ....errors import PipelineError, ValidationError
from ....specs import CallSpec, LayoutSpec, VisualSpec
from ....specs._breakpoints_spec import _BreakpointsSpec
from ....specs._delay_spec import _DelaySpec
from ....types import Array, FormKind, Indices, VariableKind
from ..._utils import components_of, declarations_of, symbols_of
from ...problems import ProblemSpec
from ...problems._block_spec import BlockSpec
from ...problems._declarations_spec import DeclarationsSpec
from ...problems._delayed_state_spec import _DelayedStateSpec
from ...problems._event_kernel import EventKernel
from ...problems._expression_printer import ExpressionPrinter
from ...problems._kernel_spec import KernelSpec
from ...problems._random_stream import RandomStream
from ...problems._reduction_schedule_spec import ReductionScheduleSpec
from ...problems._reference_spec import ReferenceSpec
from ...types import ReadKind, ScalarsFunction
from .._base import BasePass
from .._compilation import Compilation
from ._events import RELATIONS, SWITCHES


class ProblemPass(BasePass):
    """A pass that assembles the problem of the general parts of a compilation."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return (
            'layout',
            'specs',
            'reduced',
            'form',
            'functions',
            'jacobian',
            'matrices',
            'reductions',
            'events',
            'delays',
        )

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Assemble the problem."""
        compilation = self.compilation_
        compilation.problem = self._get_problem_of(compilation)

    @staticmethod
    def _get_companions(blocks: list[BlockSpec]) -> dict[str, tuple[str, str, int]]:
        """Return every companion state by name.

        It comes with its kind, the name of its state, and its index.
        """
        companions = {}
        for block in blocks:
            if block.node.interaction or block.node.is_agent:
                continue
            declarations = declarations_of(block.node.component_type, VariableKind.STATE)
            for variable, entry in block.entries.items():
                declaration = declarations.get(variable)
                if declaration is None or (declaration.derivative_of is None and declaration.lag_of is None):
                    continue
                kind, base = (
                    ('derivative', declaration.derivative_of)
                    if declaration.derivative_of
                    else ('lag', declaration.lag_of)
                )
                companions[entry.name] = (kind, f'{block.node.path}.{base}', declaration.index)
        return companions

    @staticmethod
    def _get_compiled_elements(blocks: list[BlockSpec]) -> tuple[CompiledElementSpec, ...]:
        """Return the elements and groups that are not agents with their names.

        The names are the dotted names of their states and parameters.
        """
        return tuple(
            CompiledElementSpec(
                path=block.node.path,
                element=block.node.as_element,
                states={
                    name: f'{block.node.path}.{name}'
                    for name in declarations_of(block.node.component_type, VariableKind.STATE)
                },
                parameters={
                    name: f'{block.node.path}.{name}'
                    for name in declarations_of(block.node.component_type, VariableKind.PARAMETER)
                },
                visual=ProblemPass._get_visual_of(block.node),
                members=block.node.members,
                adaptive={
                    name: declaration
                    for name, declaration in declarations_of(block.node.component_type, VariableKind.STATE).items()
                    if declaration.adaptive
                },
                lattice_shape=ProblemPass._get_lattice_shape(block.node.component),
                roles={
                    name: declaration.role
                    for name, declaration in declarations_of(block.node.component_type, VariableKind.STATE).items()
                    if declaration.role is not None
                },
            )
            for block in blocks
            if not block.node.interaction and not block.node.is_agent
        )

    @staticmethod
    def _get_lattice_shape(component: object) -> tuple[int, ...] | None:
        """Return the number of sites along every dimension of the lattice of an element.

        It is `None` off a lattice.
        """
        space = getattr(component, 'space', None)
        return space.shape_ if isinstance(space, LatticeSpace) else None

    @staticmethod
    def _get_compiled_interactions(
        blocks: list[BlockSpec],
        specs: list[tuple[KernelSpec, BlockSpec]],
    ) -> tuple[CompiledInteractionSpec, ...]:
        """Return the interactions with the paths their roles bind to."""
        return tuple(
            CompiledInteractionSpec(
                path=block.node.path,
                interaction=block.node.as_interaction,
                roles=next(spec.roles for spec, _ in specs if spec.owner == block.node.path),
            )
            for block in blocks
            if block.node.interaction
        )

    @staticmethod
    def _get_stream_of(compilation: Compilation) -> RandomStream:
        """Return the random stream that the kernels of a compilation hold."""
        kernels = compilation.kernels
        return kernels.stream if kernels is not None and kernels.stream is not None else RandomStream()

    @staticmethod
    def _get_unknowns_of(compilation: Compilation, form: FormKind) -> Indices:
        """Return the positions of the states the residuals of a static problem target.

        It is none for other forms.
        """
        rates = (
            Compilation._get_required(compilation.kernels, 'kernels').kernels[RateLaw]
            if form == FormKind.STATIC
            else []
        )
        targets = [item.target.flat_indices for kernel in rates for item in kernel.outputs if item.target is not None]
        return np.unique(np.concatenate([np.zeros(0, dtype=np.intp), *targets]))

    @staticmethod
    def _get_problem_of(compilation: Compilation) -> ProblemSpec:
        """Return the problem of the general parts of a compilation.

        Passes of formalisms add to it.
        """
        declarations = Compilation._get_required(compilation.layout, 'layout')
        blocks = list(declarations.blocks.values())
        form = Compilation._get_required(compilation.form, 'form')
        reduced = Compilation._get_required(compilation.reduced, 'reduced')
        reductions = Compilation._get_required(compilation.reductions, 'reductions')
        time = declarations.time
        functions, delays, events = compilation.functions, compilation.delays, compilation.events
        if functions is None or delays is None or events is None:
            error_msg = 'ProblemPass reads the numeric functions, the delays, and the events, which no pass wrote.'
            raise PipelineError(error_msg)
        return ProblemSpec(
            form=form,
            states=LayoutSpec(tuple(declarations.states)),
            parameters=LayoutSpec(tuple(declarations.parameters)),
            initial_state=np.concatenate([np.zeros(0), *declarations.initial_states]),
            initial_parameters=np.concatenate([np.zeros(0), *declarations.initial_parameters]),
            time=time,
            symbols={
                entry.name: block.symbols[variable] for block in blocks for variable, entry in block.entries.items()
            },
            elements=ProblemPass._get_compiled_elements(blocks),
            interactions=ProblemPass._get_compiled_interactions(
                blocks,
                Compilation._get_required(compilation.specs, 'specs'),
            ),
            system=compilation.system,
            companions=ProblemPass._get_companions(blocks),
            delays=ProblemPass._compile_delays(delays, time),
            events=(
                *ProblemPass._compile_events(events, time, reductions, declarations),
                *(ProblemPass._get_hidden_events(reduced, time, reductions) if form == FormKind.ODE else ()),
            ),
            reductions=reductions.reductions,
            jacobian=compilation.jacobian,
            unknowns=ProblemPass._get_unknowns_of(compilation, form),
            stream=ProblemPass._get_stream_of(compilation),
            breakpoints=ProblemPass._get_breakpoints_in(reduced, events, time),
            history=ProblemPass._get_history(blocks),
            matrices=Compilation._get_required(compilation.matrices, 'matrices'),
            equations=functions.equations,
            equation_symbols=functions.equation_symbols,
            has_callables=functions.has_callables,
            has_group_sums=functions.has_group_sums,
            drift=functions.drift,
            sources=functions.sources,
            diffusion=functions.diffusion,
            noise_rows=functions.noise_rows,
            noise_columns=functions.noise_columns,
            n_processes=functions.n_processes,
            noise_blocks=functions.noise_blocks,
            propensities=functions.propensities,
            jumps=functions.jumps,
            update=functions.update,
            boundary=functions.boundary,
            algebraic=functions.algebraic,
            draws=functions.draws,
            backend=functions.backend,
            backends=functions.backends,
            accelerated=functions.accelerated,
        )

    @staticmethod
    def _get_same_zeros(first: Expression, second: Expression) -> bool:
        """Check that two conditions are the same expression or opposite ones.

        Such conditions switch at the same states.
        """
        return bool(first.is_trivially_equal(second) or (first + second).expand().is_trivially_equal(SR(0)))

    @staticmethod
    def _get_linear_root(expression: Expression, time: Expression) -> float | None:
        """Return where a linear expression of time is zero, or `None`.

        The expression has numeric coefficients.
        """
        slope = expression.diff(time)
        if (
            not slope.is_numeric()
            or slope.is_zero()
            or (expression.variables() and set(expression.variables()) - {time})
        ):
            return None
        return float(-expression.subs({time: 0}) / slope)

    @staticmethod
    def _get_first_value(function: ScalarsFunction) -> Callable[[float, Array, Array], float]:
        """Return a function that gives the single value of a scalar function."""

        def first(t: float, x: Array, p: Array) -> float:
            return function(t, x, p)[0]

        return first

    @staticmethod
    def _get_reduced_guard(
        function: ScalarsFunction,
        reductions: ReductionScheduleSpec,
    ) -> Callable[[float, Array, Array], float]:
        """Return a function that gives the single value of a scalar function.

        It reads the reductions at its state.
        """
        if not reductions.steps:
            return ProblemPass._get_first_value(function)

        def reduced(t: float, x: Array, p: Array) -> float:
            return function(t, x, reductions.extend(t, x, p))[0]

        return reduced

    @staticmethod
    def _get_flat_sources(
        references: Mapping[Expression, ReferenceSpec],
        owner: str,
    ) -> dict[Expression, tuple[bool, int]]:
        """Return the vector and flat index of every single component symbol.

        The symbols are those in a namespace.
        """
        del owner
        return {
            item: (reference.is_state, reference.entry.start + reference.column)
            for item, reference in references.items()
            if reference.entry.members is None
        }

    @staticmethod
    def _reset_indices(reference: ReferenceSpec) -> npt.NDArray[np.intp]:
        """Return the flat indices a reset writes.

        The reset is of the state of a single element or of every group member.
        """
        if reference.entry.members is not None and reference.source in {ReadKind.MEMBER, ReadKind.REDUCED}:
            return reference.flat_indices.astype(np.intp)
        return np.asarray([reference.entry.start + reference.column], dtype=np.intp)

    @staticmethod
    def _get_scalar_function(
        expressions: Sequence[object],
        sources: Mapping[Expression, tuple[bool, int]],
        time: Expression,
        owner: str,
    ) -> ScalarsFunction:
        """Return a function that evaluates scalar expressions.

        It takes time, the state vector, and the parameter vector.
        """
        values = [SR(expression) for expression in expressions]
        arguments = sorted(symbols_of(values) - {time}, key=str)
        unknown = [item for item in arguments if item not in sources]
        if unknown:
            error_msg = (
                f'An event or delay of {owner} uses {", ".join(map(str, unknown))}, which belong to group members. '
                'Events and delays read states and parameters of single elements and interactions.'
            )
            raise ValidationError(error_msg)
        function = ExpressionPrinter(time=time, compiled=True).prepare(arguments).run(values).code_
        if not callable(function):
            error_msg = f'The printer returns no function of the events or delays of {owner}.'
            raise TypeError(error_msg)
        places = [sources[item] for item in arguments]

        def evaluate(t: float, x: Array, p: Array) -> list[float]:
            results = function(t, *((x if is_state else p)[index] for is_state, index in places))
            if not isinstance(results, list):
                error_msg = f'The function of the events or delays of {owner} returns no list of values.'
                raise TypeError(error_msg)
            return [float(value) for value in np.asarray(results, dtype=float)]

        return evaluate

    @staticmethod
    def _reset_places(spec: KernelSpec, targets: Sequence[Expression]) -> list[npt.NDArray[np.intp]]:
        """Return the flat indices every target of a reset writes.

        It first checks that each target is a state.
        """
        places = []
        for item in targets:
            reference = spec.references.get(item)
            if (
                reference is None
                or not reference.is_state
                or reference.source not in {ReadKind.SINGLE, ReadKind.MEMBER}
            ):
                error_msg = (
                    f'An event of {spec.owner} resets a symbol that is not a state of an element or a group member.'
                )
                raise ValidationError(error_msg)
            places.append(ProblemPass._reset_indices(reference))
        return places

    @staticmethod
    def _reset_function(
        spec: KernelSpec,
        event: CrossingLaw,
        time: Expression,
        reductions: ReductionScheduleSpec,
        *,
        members: bool = False,
    ) -> Callable[..., Array] | None:
        """Return a function that sets the states an event resets.

        All are set from the state before the event. The reset of an event of the members of a group takes the member as
        a fourth argument, and sets the states of that member alone.
        """
        reset = event.reset or {}
        targets = [item for target in reset for item in components_of(target)]
        if not targets:
            return None
        values = [
            SR(item)
            for target, value in reset.items()
            for item in (
                components_of(value) if isinstance(value, FreeModuleElement) else [value] * len(components_of(target))
            )
        ]
        places = ProblemPass._reset_places(spec, targets)
        arguments = sorted(symbols_of(values) - {time}, key=str)
        unknown = [item for item in arguments if spec.references.get(item) is None]
        if unknown:
            error_msg = (
                f'An event of {spec.owner} uses {", ".join(map(str, unknown))}, which the namespace does not hold.'
            )
            raise ValidationError(error_msg)
        reads = [
            (spec.references[item].is_state, ProblemPass._reset_indices(spec.references[item])) for item in arguments
        ]
        function = ExpressionPrinter(time=time, compiled=True).prepare(arguments).run(values).code_
        if not callable(function):
            error_msg = f'The printer returns no function of the resets of an event of {spec.owner}.'
            raise TypeError(error_msg)

        def evaluate(t: float, x: Array, p: Array) -> Sequence[object]:
            q = reductions.extend(t, x, p)
            results = function(t, *((x if is_state else q)[index] for is_state, index in reads))
            if not isinstance(results, list):
                error_msg = f'The function of the resets of an event of {spec.owner} returns no list of values.'
                raise TypeError(error_msg)
            return results

        def apply(t: float, x: Array, p: Array) -> Array:
            state = x.copy()
            for place, value in zip(places, evaluate(t, x, p), strict=True):
                state[place] = np.broadcast_to(np.asarray(value, dtype=float), place.shape)
            return state

        def apply_member(t: float, x: Array, p: Array, member: int) -> Array:
            state = x.copy()
            for place, value in zip(places, evaluate(t, x, p), strict=True):
                values = np.ravel(np.asarray(value, dtype=float))
                state[place[member] if place.size > 1 else place] = values[member] if values.size > 1 else values[0]
            return state

        return apply_member if members else apply

    @staticmethod
    def _get_member_guard(
        spec: KernelSpec,
        guard: Expression,
        time: Expression,
        reductions: ReductionScheduleSpec,
    ) -> Callable[[float, Array, Array], Array]:
        """Return a function that gives the value of a guard for every member of a group."""
        arguments = sorted(symbols_of([guard]) - {time}, key=str)
        unknown = [item for item in arguments if spec.references.get(item) is None]
        if unknown:
            error_msg = (
                f'An event of {spec.owner} uses {", ".join(map(str, unknown))}, which the namespace does not hold.'
            )
            raise ValidationError(error_msg)
        reads = [
            (spec.references[item].is_state, ProblemPass._reset_indices(spec.references[item])) for item in arguments
        ]
        function = ExpressionPrinter(time=time, compiled=True).prepare(arguments).run([guard]).code_
        if not callable(function):
            error_msg = f'The printer returns no function of the guard of an event of {spec.owner}.'
            raise TypeError(error_msg)
        rows = spec.rows

        def evaluate(t: float, x: Array, p: Array) -> Array:
            q = reductions.extend(t, x, p)
            results = function(t, *((x if is_state else q)[index] for is_state, index in reads))
            if not isinstance(results, list):
                error_msg = f'The function of the guard of an event of {spec.owner} returns no list of values.'
                raise TypeError(error_msg)
            return np.broadcast_to(np.asarray(results[0], dtype=float), (rows,))

        return evaluate

    @staticmethod
    def _reads_members(spec: KernelSpec, guard: Expression) -> bool:
        """Check whether a guard reads the states or parameters of the members of a group."""
        return any(
            reference is not None and reference.source is ReadKind.MEMBER
            for reference in (spec.references.get(item) for item in symbols_of([guard]))
        )

    @staticmethod
    def _get_state_conditions_of(expressions: Iterable[object], time: Expression) -> list[Expression]:
        """Return the expressions whose sign switches a law.

        They come from conditions and steps that read more than time.
        """
        found: list[Expression] = []
        stack = [SR(expression) for expression in expressions]
        while stack:
            item = stack.pop()
            operation = item.operator()
            if operation is None:
                continue
            stack.extend(item.operands())
            if operation in RELATIONS:
                left, right = item.operands()
                candidate = left - right
            elif getattr(operation, 'name', lambda: '')() in SWITCHES:
                candidate = item.operands()[0]
            else:
                continue
            if set(candidate.variables()) - {time} and not any(
                ProblemPass._get_same_zeros(candidate, other) for other in found
            ):
                found.append(candidate)
        return found

    @staticmethod
    def _get_breakpoints_of(expressions: Iterable[object], time: Expression) -> _BreakpointsSpec:
        """Return the times where conditions on time in expressions change.

        Examples are step, pulse, and series inputs.
        """
        fixed: set[float] = set()
        periodic: set[tuple[float, float]] = set()
        unresolved = False
        stack = [SR(expression) for expression in expressions]
        seen: dict[int, list[Expression]] = defaultdict(list)
        while stack:
            item = stack.pop()
            operation = item.operator()
            if operation is None or not item.has(time):
                continue
            twins = seen[hash(item)]
            if any(item.is_trivially_equal(twin) for twin in twins):
                continue
            twins.append(item)
            if operation in RELATIONS:
                left, right = item.operands()
                difference = left - right
                root = ProblemPass._get_linear_root(difference, time)
                resolved = root is not None or time not in difference.variables()
                if root is not None:
                    fixed.add(root)
                for call in BasePass._get_calls_of(difference, 'mod'):
                    argument, period = call.operands()
                    zero, rest = ProblemPass._get_linear_root(argument, time), difference - call
                    if zero is not None and period.is_numeric() and rest.is_numeric():
                        span = float(period)
                        offset = float(-rest)
                        periodic.update({(zero, span), (zero + offset, span)})
                        resolved = True
                unresolved = unresolved or not resolved
            stack.extend(item.operands())
        return _BreakpointsSpec(tuple(sorted(fixed)), tuple(sorted(periodic)), unresolved=unresolved)

    @staticmethod
    def _get_breakpoints_in(
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
        events: list[tuple[KernelSpec, CrossingLaw]],
        time: Expression,
    ) -> _BreakpointsSpec:
        """Return the breakpoints of time-dependent inputs in every law and event guard.

        Args:
            specs:
                The laws of every kernel spec.

            events:
                The events with the kernel specs of their components.

            time:
                The time symbol.

        Returns:
            breakpoints:
                The breakpoints.
        """
        expressions: list[object] = []
        for _, laws in specs:
            for law in laws:
                value = law.rate if isinstance(law, JumpLaw) else getattr(law, 'value', None)
                expressions += (
                    [item for item in components_of(value) if isinstance(item, Expression)] if value is not None else []
                )
        expressions += [event.guard for _, event in events if not BaseLaw._is_function(event.guard)]
        return ProblemPass._get_breakpoints_of(expressions, time)

    @staticmethod
    def _compile_delays(delays: Sequence[_DelayedStateSpec], time: Expression) -> tuple[_DelaySpec, ...]:
        """Return the delays of a system with their delay functions compiled.

        Args:
            delays:
                The delayed states with their hidden parameters.

            time:
                The time symbol.

        Returns:
            delays:
                The compiled delays.
        """
        compiled = []
        for item in delays:
            spec, hidden = item.spec, item.entry
            tau = ProblemPass._get_scalar_function(
                [item.tau],
                ProblemPass._get_flat_sources(spec.references, spec.owner),
                time,
                spec.owner,
            )
            compiled.append(
                _DelaySpec(
                    parameters=hidden.start + np.arange(hidden.size),
                    states=item.states,
                    tau=ProblemPass._get_first_value(tau),
                    minimum=item.minimum,
                    symbol=item.symbol,
                    state=item.state,
                    delay=SR(item.tau),
                ),
            )
        return tuple(compiled)

    @staticmethod
    def _get_symbolic_value(
        spec: KernelSpec,
        value: object,
        reductions: ReductionScheduleSpec,
        time: Expression,
    ) -> Callable[[float, Array, Array], object]:
        """Return the function of time and both vectors of a symbolic guard or reset.

        The guard or reset sits next to a function.
        """
        expression = SR(value)
        arguments = sorted(symbols_of([expression]) - {time}, key=str)
        function = ExpressionPrinter(time=time, compiled=True).prepare(arguments).run([expression]).code_
        if not callable(function):
            error_msg = f'The printer returns no function of an event of {spec.owner}.'
            raise TypeError(error_msg)
        reads = [
            (spec.references[item].is_state, ProblemPass._reset_indices(spec.references[item])) for item in arguments
        ]

        def evaluate(t: float, x: Array, p: Array) -> object:
            q = reductions.extend(t, x, p)
            results = function(t, *((x if is_state else q)[index] for is_state, index in reads))
            if not isinstance(results, list):
                error_msg = f'The function of an event of {spec.owner} returns no list of values.'
                raise TypeError(error_msg)
            return results[0]

        return evaluate

    @staticmethod
    def _get_value_function(
        spec: KernelSpec,
        value: object,
        reductions: ReductionScheduleSpec,
        declarations: DeclarationsSpec,
    ) -> Callable[[float, Array, Array], object]:
        """Return the function of time and both vectors of one guard or reset value.

        The value is one of an event.
        """
        if not BaseLaw._is_function(value):
            return ProblemPass._get_symbolic_value(spec, value, reductions, declarations.time)
        if isinstance(value, CallSpec) and (value.arguments or value.random or value.vectorized):
            error_msg = (
                f'An event of {spec.owner} has the guard or reset {value!r}. An event takes `f(t, values)` or '
                '`CallSpec(f, relative=True)`, because it reads its values by name and draws nothing.'
            )
            raise ValidationError(error_msg)
        states, parameters = LayoutSpec(tuple(declarations.states)), LayoutSpec(tuple(declarations.parameters))
        relative = isinstance(value, CallSpec) and value.relative
        function = value.function if isinstance(value, CallSpec) else value
        if not callable(function):
            error_msg = f'An event of {spec.owner} calls {function!r}, which is not a function.'
            raise ValidationError(error_msg)
        names = spec._get_relative_names((*states.names, *parameters.names))

        def evaluate(t: float, x: Array, p: Array) -> object:
            values = {**states.get_values(x), **parameters.get_values(p)}
            return function(t, {name: values[path] for name, path in names.items()} if relative else values)

        return evaluate

    @staticmethod
    def _get_function_event(
        spec: KernelSpec,
        event: CrossingLaw,
        label: str,
        reductions: ReductionScheduleSpec,
        declarations: DeclarationsSpec,
    ) -> EventKernel:
        """Return an event whose guard or reset is a function.

        Over a group, there is one guard value per member.
        """
        guard = ProblemPass._get_value_function(spec, event.guard, reductions, declarations)
        reset = event.reset or {}
        targets = list(reset)
        places = ProblemPass._reset_places(spec, targets)
        values = [ProblemPass._get_value_function(spec, reset[item], reductions, declarations) for item in targets]
        rows = spec.rows if spec.rows > 1 else 0

        def guard_values(t: float, x: Array, p: Array) -> Array | float:
            found = np.asarray(guard(t, x, p), dtype=float)
            return np.broadcast_to(found, (rows,)) if rows else float(found)

        def apply(t: float, x: Array, p: Array, member: int | None = None) -> Array:
            state = x.copy()
            for place, value in zip(places, values, strict=True):
                found = np.broadcast_to(np.asarray(value(t, x, p), dtype=float), place.shape)
                if member is None or place.size == 1:
                    state[place] = found
                else:
                    state[place[member]] = found[member]
            return state

        return EventKernel(
            label,
            guard_values,
            apply if targets else None,
            int(event.direction),
            bool(event.terminal),
            members=rows,
        ).prepare()

    @staticmethod
    def _compile_events(
        events: list[tuple[KernelSpec, CrossingLaw]],
        time: Expression,
        reductions: ReductionScheduleSpec,
        declarations: DeclarationsSpec,
    ) -> tuple[EventKernel, ...]:
        """Return the events of a system compiled into guard and reset functions.

        The functions are of the state vector.

        Args:
            events:
                The events with the kernel specs of their components.

            time:
                The time symbol.

            reductions:
                The reductions the guards and resets read.

            declarations:
                The layout of the variables that functions of guards and resets read.

        Returns:
            events:
                The compiled events.
        """
        compiled = []
        counts = Counter(spec.owner for spec, _ in events)
        positions: Counter[str] = Counter()
        for spec, event in events:
            position = positions[spec.owner]
            positions[spec.owner] += 1
            label = event.label or (spec.owner if counts[spec.owner] == 1 else f'{spec.owner}:{position}')
            if BaseLaw._is_function(event.guard) or any(map(BaseLaw._is_function, (event.reset or {}).values())):
                compiled.append(ProblemPass._get_function_event(spec, event, label, reductions, declarations))
                continue
            if ProblemPass._reads_members(spec, SR(event.guard)):
                compiled.append(
                    EventKernel(
                        label,
                        ProblemPass._get_member_guard(spec, SR(event.guard), time, reductions),
                        ProblemPass._reset_function(spec, event, time, reductions, members=True),
                        int(event.direction),
                        bool(event.terminal),
                        members=spec.rows,
                    ).prepare(),
                )
                continue
            sources = ProblemPass._get_flat_sources(spec.references, spec.owner)
            guard = ProblemPass._get_scalar_function([event.guard], sources, time, spec.owner)
            compiled.append(
                EventKernel(
                    label,
                    ProblemPass._get_reduced_guard(guard, reductions),
                    ProblemPass._reset_function(spec, event, time, reductions),
                    int(event.direction),
                    bool(event.terminal),
                ).prepare(),
            )
        return tuple(compiled)

    @staticmethod
    def _get_hidden_events(
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
        time: Expression,
        reductions: ReductionScheduleSpec,
    ) -> tuple[EventKernel, ...]:
        """Return one hidden event per single component condition that switches a flow.

        The condition is on the states of single components and switches a law of a flow.

        Args:
            specs:
                The laws of every kernel spec.

            time:
                The time symbol.

            reductions:
                The reductions the conditions read.

        Returns:
            events:
                The hidden events.
        """
        compiled = []
        for spec, laws in specs:
            sources = ProblemPass._get_flat_sources(spec.references, spec.owner)
            values = [
                value
                for law in laws
                if isinstance(law, BaseLaw) and not isinstance(law, JumpLaw)
                for value in law._get_values(include=('value', 'lhs', 'rhs'))
                if isinstance(value, Expression)
            ]
            for index, condition in enumerate(ProblemPass._get_state_conditions_of(values, time)):
                if not set(condition.variables()) - {time} <= set(sources):
                    continue
                guard = ProblemPass._get_scalar_function([condition], sources, time, spec.owner)
                label = f'{spec.owner}:switch:{index}'
                compiled.append(
                    EventKernel(
                        label,
                        ProblemPass._get_reduced_guard(guard, reductions),
                        None,
                        0,
                        False,
                        hidden=True,
                    ).prepare(),
                )
        return tuple(compiled)

    @staticmethod
    def _get_history(blocks: list[BlockSpec]) -> dict[str, object]:
        """Return the history every state declares, by the name of its layout entry."""
        history = {}
        for block in blocks:
            if block.node.interaction or block.node.is_agent:
                continue
            declarations = declarations_of(block.node.component_type, VariableKind.STATE)
            for variable, entry in block.entries.items():
                declaration = declarations.get(variable)
                if declaration is not None and declaration.history is not None:
                    history[entry.name] = declaration.history
        return history

    @staticmethod
    def _get_visual_of(node: NodeSpec) -> VisualSpec:
        """Return the visual of an element or a group.

        It is its own visual when it sets one, and its default otherwise.
        """
        element = node.as_element
        if element.visual is not None:
            return element.visual
        prepared = 'default_visual_' in vars(element)
        default: VisualSpec = (element if prepared else element.prepare(node.space)).default_visual_
        return default
