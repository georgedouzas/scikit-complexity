"""Generate, compile, and cache Numba kernels and stepper loops."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
import inspect
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numba
import numpy as np
from numba.core.dispatcher import Dispatcher

from ...components.laws import (
    ArrayUpdateLaw,
    BilinearLaw,
    BoundaryLaw,
    CascadeLaw,
    CrossingLaw,
    DissipatorLaw,
    EquationLaw,
    ExtremalLaw,
    FluxLaw,
    GeneratorLaw,
    JumpLaw,
    LocalRuleLaw,
    MatchLaw,
    NoiseLaw,
    RateLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
    TransitionLaw,
    UpdateLaw,
)
from ...components.spaces import (
    CayleySpace,
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
    StaticTime,
)
from ...errors import BackendWarning, ValidationError
from ...specs import AcceleratedSpec, EntrySpec, LoweredFunctionsSpec, NeighborReadSpec, SupportSpec
from ...types import Array, FormKind, NoiseKind, OrderKind, VectorFunction
from ..problems._draw_kernel import DrawKernel
from ..problems._edge_row_search import EdgeRowSearch
from ..problems._edge_search import EdgeSearch
from ..problems._expression_printer import ExpressionPrinter
from ..problems._law_kernel import LawKernel
from ..problems._lowered_kernels_spec import LoweredKernelsSpec
from ..problems._metric_search import MetricSearch
from ..problems._reduction_schedule_spec import ReductionScheduleSpec
from ..problems._reference_spec import ReferenceSpec
from ..problems._source_loader import SourceLoader
from ..problems._steps import step_count
from ..types import ReadKind
from . import _numba_components, _numba_neighbours, _numba_stencils
from ._base import BaseBackend
from ._numba_loops import LOOPS, NO_BOUNDARY
from ._numba_printer import NumbaPrinter

FORM_KINDS: dict[FormKind, list[type]] = {
    FormKind.ODE: [RateLaw],
    FormKind.FRACTIONAL: [RateLaw],
    FormKind.SDE: [RateLaw, NoiseLaw],
    FormKind.JUMP: [JumpLaw],
    FormKind.DISCRETE: [UpdateLaw],
    FormKind.STATIC: [RateLaw, UpdateLaw],
}
EVENT_FORMS = (FormKind.JUMP, FormKind.DISCRETE)
RATE_FORMS = (FormKind.ODE, FormKind.FRACTIONAL, FormKind.SDE, FormKind.STATIC)
HEADER = """import math

import numpy as np
from numba import njit, prange

from skcomplex.core.engine.backends._numba_components import _component_values
from skcomplex.core.engine.backends._numba_neighbours import (
    _build_grid,
    _cell_coordinate,
    _cone_neighbours,
    _edge_neighbours,
    _nearest_neighbours,
    _radius_neighbours,
)
from skcomplex.core.engine.backends._numba_stencils import _neighbor_value
from skcomplex.core.engine.problems._steps import step_count
"""


class NumbaBackend(BaseBackend):
    """A backend that compiles fused kernels and stepper loops with Numba.

    They are cached on disk. Kinds of laws that contain a plain Python callable fall back to the NumPy functions, and
    the problem then has no compiled stepper loops.

    Choose it for large lattices, networks, and agent groups, where compiled loops run much faster than NumPy. It
    appears in many fields, such as long runs of spin lattices, sandpiles, and epidemics on large networks.

    Args:
        cache_dir:
            The directory of the generated kernel sources and Numba's cache, or `None` for `~/.cache/skcomplex`.

        parallel:
            Whether member and pair loops over groups of at least 1000 members run in parallel threads. Kernels that
            the compiled loops of jump processes and discrete maps call once per event always run serially.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
        >>> system = CompositeSystem(
        ...     elements=[GroupElement('herds', element=PopulationElement, size=2, values={'x': [1.0, 2.0]})],
        ...     interactions=[ExponentialGrowthInteraction('growth', target='herds', r=3.0)],
        ... )
        >>> problem = simulate(system, until=0, backend=NumbaBackend()).problem
        >>> problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist()
        [3.0, 6.0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return what the backend writes functions of.

        It is the times, spaces, laws, noise, and orders.
        """
        return SupportSpec(
            time={ContinuousTime, DiscreteTime, StaticTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={
                BoundaryLaw,
                CrossingLaw,
                DissipatorLaw,
                ArrayUpdateLaw,
                EquationLaw,
                ExtremalLaw,
                FluxLaw,
                GeneratorLaw,
                JumpLaw,
                LocalRuleLaw,
                MatchLaw,
                NoiseLaw,
                BilinearLaw,
                RateLaw,
                CascadeLaw,
                RuleLaw,
                SampleLaw,
                TransferLaw,
                TransitionLaw,
                UpdateLaw,
            },
            noise=frozenset(NoiseKind),
            orders=frozenset(OrderKind),
        )

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the backend that problems record."""
        return 'numba'

    def __init__(self, cache_dir: str | Path | None = None, parallel: bool = True) -> None:
        self.cache_dir = cache_dir
        self.parallel = parallel

    def _run(self, inputs: LoweredKernelsSpec | None) -> None:
        """Set the numeric functions of the form of the kernels.

        They are compiled with Numba where every law compiles.

        Args:
            inputs:
                The kernels of a problem.

        Raises:
            ValidationError:
                No kernels are passed, or they have no form.
        """
        if inputs is None or inputs.form is None:
            error_msg = 'A backend runs on the kernels of a problem of one form. Pass a `LoweredKernelsSpec` to `run`.'
            raise ValidationError(error_msg)
        form, size, kernels, draws = inputs.form, inputs.size, inputs.kernels, inputs.draws
        used: dict[type, list[LawKernel]] = {
            BoundaryLaw: kernels.get(BoundaryLaw, []),
            RateLaw: kernels.get(RateLaw, []) if form in RATE_FORMS else [],
            NoiseLaw: kernels.get(NoiseLaw, []) if form == FormKind.SDE else [],
            JumpLaw: kernels.get(JumpLaw, []) if form == FormKind.JUMP else [],
            UpdateLaw: kernels.get(UpdateLaw, []) if form in (FormKind.DISCRETE, FormKind.STATIC) else [],
        }
        reductions = inputs.reductions or ReductionScheduleSpec()
        reduced = bool(reductions.steps)
        fused = self._get_reductions_compile(reductions)
        compiled: dict[type, bool] = {
            kind: fused and all(self._is_compilable(kernel) for kernel in items) for kind, items in used.items()
        }
        if JumpLaw in compiled and any(
            not isinstance(increment, float) for changes in inputs.jump_changes.values() for _, increment in changes
        ):
            compiled[JumpLaw] = False
        backends = self._report_fallbacks(self._get_name(), used, compiled, reductions)
        plan: dict[type, tuple[str, str]] = {
            RateLaw: ('drift', 'add'),
            NoiseLaw: ('diffusion', 'concatenate'),
            JumpLaw: ('propensities', 'concatenate'),
            UpdateLaw: ('update', 'assign'),
            BoundaryLaw: ('boundary', 'assign'),
        }
        needed = list(FORM_KINDS[form])
        if kernels.get(BoundaryLaw):
            needed = [*needed, BoundaryLaw]
        parts = [
            (*plan[kind], used[kind], reduced and any(kernel.reads_reductions for kernel in used[kind]))
            for kind in needed
            if compiled[kind]
        ]
        printer = NumbaPrinter(parallel=self.parallel and form not in EVENT_FORMS)
        lines = printer.prepare(reductions if reduced and fused else None).run(parts).code_
        accelerated_loop = {
            FormKind.ODE: 'rk4',
            FormKind.SDE: 'euler_maruyama',
            FormKind.JUMP: 'gillespie',
            FormKind.DISCRETE: 'iterate' if draws is None else 'iterate_random',
        }.get(form, '')
        has_loop = bool(accelerated_loop) and all(compiled[kind] for kind in needed)
        bounded = 'HAS_BOUNDARY = True\n' if BoundaryLaw in needed else NO_BOUNDARY
        loops = (
            [
                bounded,
                LOOPS[accelerated_loop],
                *([LOOPS['velocity_verlet'], LOOPS['yoshida4']] if form == FormKind.ODE else []),
            ]
            if has_loop
            else []
        )
        text = '\n'.join([HEADER, *lines, *loops]) + '\n'
        module = self._load(text, printer.functions_, self.cache_dir)
        layout = printer.get_layout(self._get_layout_entries(used, reductions))
        orders = [(int(layout[2 * printer.slots_[name]]), order) for name, order in printer.orders_.items()]
        lowering = self._get_lowering(
            form,
            needed,
            used=used,
            compiled=compiled,
            changes=inputs.jump_changes,
            size=size,
            module=module,
            layout=(layout, orders),
            reductions=reductions,
        )
        if has_loop:
            accelerated = self._get_accelerated(accelerated_loop, module, layout, lowering, draws)
            lowering = replace(lowering, accelerated=accelerated)
        self.functions_ = replace(lowering, backends=backends)

    def _get_lowering(
        self,
        form: FormKind,
        needed: list[type],
        *,
        used: Mapping[type, list[LawKernel]],
        compiled: Mapping[type, bool],
        changes: Mapping[int, list[tuple[ReferenceSpec, object]]],
        size: int,
        module: ModuleType,
        layout: tuple[np.ndarray, list[tuple[int, np.ndarray]]],
        reductions: ReductionScheduleSpec,
    ) -> LoweredFunctionsSpec:
        """Return the lowering that binds the compiled functions to the layout.

        It has NumPy fallbacks. The layout comes with the orders drawn at every step and the start of their data in it,
        which a bound function copies into the layout before every call.
        """
        data, orders = layout

        def bind(name: str) -> Callable[[float, Array, Array], Array]:
            function = getattr(module, name)

            def call(t: float, x: Array, p: Array) -> Array:
                for start, order in orders:
                    data[start : start + order.size] = order
                return function(t, x, p, data)

            return call

        numpy: dict[type, Callable[[], VectorFunction]] = {
            RateLaw: lambda: BaseBackend._get_adding_function(used[RateLaw], size),
            NoiseLaw: lambda: BaseBackend._get_concatenating_function(used[NoiseLaw]),
            JumpLaw: lambda: BaseBackend._get_concatenating_function(used[JumpLaw]),
            UpdateLaw: lambda: BaseBackend._get_assigning_function(used[UpdateLaw]),
            BoundaryLaw: lambda: BaseBackend._get_assigning_function(used[BoundaryLaw]),
        }
        names: dict[type, str] = {
            RateLaw: 'drift',
            NoiseLaw: 'diffusion',
            JumpLaw: 'propensities',
            UpdateLaw: 'update',
            BoundaryLaw: 'boundary',
        }
        functions: dict[type, VectorFunction | None] = {
            kind: (
                bind(names[kind])
                if compiled[kind]
                else (
                    reductions.wrap(numpy[kind]())
                    if any(kernel.reads_reductions for kernel in used[kind])
                    else numpy[kind]()
                )
            )
            for kind in needed
        }
        lowering = LoweredFunctionsSpec(
            drift=functions[RateLaw] if form in RATE_FORMS else None,
            diffusion=(
                functions.get(NoiseLaw, BaseBackend._get_concatenating_function([])) if form == FormKind.SDE else None
            ),
            propensities=(
                functions.get(JumpLaw, BaseBackend._get_concatenating_function([])) if form == FormKind.JUMP else None
            ),
            jumps=LawKernel._jump_channels(used[JumpLaw], changes) if form == FormKind.JUMP else (),
            update=functions[UpdateLaw] if form in (FormKind.DISCRETE, FormKind.STATIC) else None,
            boundary=functions.get(BoundaryLaw),
        )
        if form == FormKind.SDE:
            rows, columns, processes, blocks = LawKernel._lower_noise(used[NoiseLaw])
            lowering = replace(
                lowering,
                noise_rows=rows,
                noise_columns=columns,
                n_processes=processes,
                noise_blocks=blocks,
            )
        return lowering

    def _get_accelerated(
        self,
        loop: str,
        module: ModuleType,
        layout: np.ndarray,
        lowering: LoweredFunctionsSpec,
        draws: DrawKernel | None,
    ) -> AcceleratedSpec:
        """Return the compiled stepper loop of a form bound to the layout."""
        function = getattr(module, loop)
        if loop == 'rk4':
            verlet, yoshida4 = module.velocity_verlet, module.yoshida4
            return AcceleratedSpec(
                rk4=lambda t, x, p, until, h: function(t, x, p, layout, until, h),
                yoshida4=lambda t, x, p, until, h, first, second, weights: yoshida4(
                    t,
                    x,
                    p,
                    layout,
                    until,
                    h,
                    first,
                    second,
                    weights,
                ),
                velocity_verlet=lambda t, x, p, until, h, positions, velocities: verlet(
                    t,
                    x,
                    p,
                    layout,
                    until,
                    h,
                    positions,
                    velocities,
                ),
            )
        if loop == 'euler_maruyama':
            noise = self._get_noise_arrays(lowering)
            return AcceleratedSpec(
                euler_maruyama=lambda t, x, p, until, h, rng: function(t, x, p, layout, until, h, rng, *noise),
            )
        if loop == 'gillespie':
            targets, increments = self._get_channel_arrays(lowering)
            return AcceleratedSpec(
                gillespie=lambda t, x, p, until, rng: function(t, x, p, layout, until, rng, targets, increments),
            )
        if loop == 'iterate_random' and draws is not None:
            offset, flags = draws.offset, draws.normal
            return AcceleratedSpec(
                iterate=lambda t, x, p, until, rng: function(t, x, p, layout, until, rng, offset, flags),
            )
        return AcceleratedSpec(iterate=lambda t, x, p, until: function(t, x, p, layout, until))

    @staticmethod
    def _search_entries(search: object) -> dict[str, EntrySpec]:
        """Return the entries a neighbourhood search reads.

        They are a changing layer's edges, or positions and directions.
        """
        if isinstance(search, EdgeRowSearch):
            return {entry.name: entry for entry in (search.source, search.target, search.active)}
        if not isinstance(search, MetricSearch):
            return {}
        found = {search.positions.name: search.positions}
        if search.direction is not None:
            found[search.direction.entry.name] = search.direction.entry
        return found

    @staticmethod
    def _get_layout_entries(
        used: Mapping[type, list[LawKernel]],
        reductions: ReductionScheduleSpec,
    ) -> dict[str, EntrySpec]:
        """Return every entry the kernels and the reductions of a problem read, by name."""
        entries = {
            reference.entry.name: reference.entry
            for items in used.values()
            for kernel in items
            for reference in kernel.references.values()
        }
        for items in used.values():
            for kernel in items:
                search = kernel.pairs
                if isinstance(search, EdgeSearch) and search.source is not None and search.target is not None:
                    entries[search.source.name] = search.source
                    entries[search.target.name] = search.target
                for reference in kernel.references.values():
                    read = reference.neighbor
                    if isinstance(read, NeighborReadSpec) and read.entry is not None:
                        entries[read.entry.name] = read.entry
        for item in reductions.reductions:
            entries[item.entry.name] = item.entry
            entries.update({read.entry.name: read.entry for read in item.reads.values() if read.entry is not None})
            if item.members is not None:
                entries[item.members.name] = item.members
        for neighbourhood in reductions.neighbourhoods:
            entries[neighbourhood.entry.name] = neighbourhood.entry
            entries.update(
                {pair.read.entry.name: pair.read.entry for pair in neighbourhood.reads.values() if pair.read.entry},
            )
            entries.update(NumbaBackend._search_entries(neighbourhood.search))
        return entries

    @staticmethod
    def _is_compilable(kernel: LawKernel) -> bool:
        """Check whether every output of a kernel can be compiled.

        An output that can be compiled has no Python callable.
        """
        lattice = all(
            (
                reference.source is not ReadKind.NEIGHBOR
                or reference.shifted.sizes is not None
                or NumbaPrinter._is_gathered(reference)
            )
            and reference.source is not ReadKind.DERIVED
            for reference in kernel.references.values()
        )
        return lattice and all(
            output.function is None
            or (
                output.arguments is not None
                and isinstance(output.function, Dispatcher)
                and not (output.relative or output.random or output.vectorized)
            )
            for output in kernel.outputs
        )

    @staticmethod
    def _get_reductions_compile(reductions: ReductionScheduleSpec) -> bool:
        """Check whether every reduction compiles with Numba.

        Every reduction must print in Numba's dialect, and no reduction may run shortest paths in NetworkX.
        """
        if any(reduction.function is not None for reduction in reductions.reductions):
            return False
        if any(neighbourhood.kind.startswith('path-') for neighbourhood in reductions.neighbourhoods):
            return False
        try:
            for reduction in reductions.reductions:
                condition = reduction.condition if reduction.condition is not None else 1
                ExpressionPrinter('scalar').prepare().run([reduction.argument, condition])
            for neighbourhood in reductions.neighbourhoods:
                ExpressionPrinter('scalar').prepare().run([neighbourhood.argument, neighbourhood.weight])
        except ValidationError:
            return False
        return True

    @staticmethod
    def _get_noise_arrays(lowering: LoweredFunctionsSpec) -> tuple[object, ...]:
        """Return the arrays of the noise of a lowering.

        They are the rows, columns, process count, and flattened covariance blocks.
        """
        blocks = lowering.noise_blocks
        sizes = np.asarray([block.size for block in blocks], dtype=np.int64)
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(sizes**2)])[:-1].astype(np.int64)
        factors = np.concatenate([np.zeros(0), *(np.ravel(block.factor) for block in blocks)]).astype(np.float64)
        return (
            lowering.noise_rows.astype(np.int64),
            lowering.noise_columns.astype(np.int64),
            int(lowering.n_processes),
            np.asarray([block.start for block in blocks], dtype=np.int64),
            sizes,
            offsets,
            factors,
        )

    @staticmethod
    def _get_channel_arrays(lowering: LoweredFunctionsSpec) -> tuple[np.ndarray, np.ndarray]:
        """Return the state targets and increments of every jump channel.

        They are padded with -1 targets.
        """
        n_channels = sum(channels.size for channels in lowering.jumps)
        width = max((len(channels.increments) for channels in lowering.jumps), default=0)
        targets = np.full((n_channels, width), -1, dtype=np.int64)
        increments = np.zeros((n_channels, width))
        for channels in lowering.jumps:
            rows = slice(channels.offset, channels.offset + channels.size)
            count = len(channels.increments)
            targets[rows, :count] = channels.targets.T
            increments[rows, :count] = channels.increments
        return targets, increments

    @staticmethod
    def _get_unprinted_reduction(reductions: ReductionScheduleSpec) -> str | None:
        """Return the first reduction that does not print in the scalar dialect.

        It returns `None` when every one prints.
        """
        for reduction in reductions.reductions:
            if reduction.function is not None:
                return f'the function reduction {reduction.argument}'
            try:
                condition = reduction.condition if reduction.condition is not None else 1
                ExpressionPrinter('scalar').prepare().run([reduction.argument, condition])
            except ValidationError:
                return f'the {reduction.kind} reduction {reduction.argument}'
        for neighbourhood in reductions.neighbourhoods:
            if neighbourhood.kind.startswith('path-'):
                return f'the shortest paths reduction {neighbourhood.kind}, which NetworkX computes,'
            try:
                ExpressionPrinter('scalar').prepare().run([neighbourhood.argument, neighbourhood.weight])
            except ValidationError:
                return f'the neighbourhood {neighbourhood.kind} reduction {neighbourhood.argument}'
        return None

    @staticmethod
    def _report_fallbacks(
        name: str,
        used: Mapping[type, list[LawKernel]],
        compiled: Mapping[type, bool],
        reductions: ReductionScheduleSpec,
    ) -> dict[str, str]:
        """Warn once about the kinds run in Python, and return the runners.

        It warns about the kinds whose kernels run in Python and returns every kind's runner.

        Args:
            name:
                The name of the compiling backend.

            used:
                The kernels of every law kind the backend lowers.

            compiled:
                Whether the backend compiles the kernels of every law kind.

            reductions:
                The reductions the kernels read.

        Returns:
            backends:
                The backend name for every law kind with kernels, `'python'` for those that run in Python.
        """
        kinds = [kind for kind, items in used.items() if items]
        fallen = [kind.__name__ for kind in kinds if not compiled[kind]]
        if fallen:
            unprinted = NumbaBackend._get_unprinted_reduction(reductions)
            reason = (
                f'{unprinted} does not print in its dialect'
                if unprinted is not None
                else 'a law holds a Python callable or reads values it cannot compile'
            )
            message = f'The {name} backend runs the {", ".join(fallen)} laws in Python, because {reason}.'
            warnings.warn(message, BackendWarning, stacklevel=4)
        return {kind.__name__: name if compiled[kind] else 'python' for kind in kinds}

    @staticmethod
    def _get_helpers_fingerprint() -> str:
        """Return a digest of the compiled helpers and of the Numba version.

        The helpers are those that generated kernels import. Numba caches a kernel with the compiled helpers it calls,
        but checks only the kernel's own source, so a kernel cached against other helpers, such as those of another
        version of the library, would run stale code. The digest keys the cached kernels on the helpers too, and on the
        module of the step count that the stepper loops call.
        """
        modules = (_numba_components, _numba_neighbours, _numba_stencils, sys.modules[step_count.__module__])
        sources = [inspect.getsource(module).encode() for module in modules]
        return hashlib.sha256(b''.join([numba.__version__.encode(), *sources])).hexdigest()

    @staticmethod
    def _load(text: str, functions: Mapping[str, object], cache_dir: str | Path | None) -> ModuleType:
        """Return the module of generated kernel source.

        It is imported from a file in the cache directory. Source without injected functions is cached by Numba. Source
        with injected functions, which Numba cannot cache, is compiled uncached. The file starts with the fingerprint of
        the compiled helpers, so kernels compiled against other helpers are cached apart.

        Args:
            text:
                The generated source.

            functions:
                The functions the source reads by name, injected into its module.

            cache_dir:
                The directory of the generated files, or `None` for `~/.cache/skcomplex/kernels`.

        Returns:
            module:
                The module of the source.
        """
        cached = text.replace('cache=CACHE', f'cache={not functions}')
        source = f'# Compiled helpers: {HELPERS_FINGERPRINT}\n{cached}'
        return SourceLoader('skcomplex_kernels', functions).prepare(cache_dir).run(source).module_


HELPERS_FINGERPRINT = NumbaBackend._get_helpers_fingerprint()
