"""Generate, compile, and cache Numba kernels and stepper loops from the kernels of a problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
import importlib.util
import inspect
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import numba
import numpy as np

from ...core.base import (
    NUMERIC_DOMAINS,
    ComponentKind,
    DomainKind,
    Entry,
    Form,
    LawKind,
    NoiseKind,
    OrderKind,
    SpaceKind,
    Supports,
    TimeKind,
    Vector,
)
from ...core.laws import BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ...core.problems import (
    Accelerated,
    BaseBackend,
    Draws,
    EdgeSearch,
    Kernel,
    Lowering,
    ReductionSchedule,
    Reference,
    Source,
    reads_reductions,
    register_backend,
)
from ...core.problems._backends import (
    AUTO_MEMBERS,
    adding_function,
    assigning_function,
    concatenating_function,
    report_fallbacks,
)
from ...core.problems._kernels import jump_channels, lower_noise
from ...core.problems._neighbourhoods import EdgeRows, MetricSearch
from ..solvers import step_count
from . import _numba_neighbours, _numba_stencils
from ._numba_kernels import _edge_kernel, _entry_function, _is_compilable, _member_kernel, _output_count, _pair_kernel
from ._numba_loops import LOOPS, NO_BOUNDARY, _channel_arrays, _noise_arrays
from ._numba_reductions import _reduce_function, _reductions_compile
from ._numba_source import _Source

DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'skcomplex' / 'kernels'
FORM_KINDS: dict[Form, list[type]] = {
    Form.ODE: [RateLaw],
    Form.FRACTIONAL: [RateLaw],
    Form.SDE: [RateLaw, NoiseLaw],
    Form.JUMP: [JumpLaw],
    Form.DISCRETE: [UpdateLaw],
}
EVENT_FORMS = (Form.JUMP, Form.DISCRETE)
HEADER = """import math

import numpy as np
from numba import njit, prange

from skcomplex.engine.backends._numba_neighbours import (
    build_grid,
    cell_coordinate,
    cone_neighbours,
    edge_neighbours,
    nearest_neighbours,
    radius_neighbours,
)
from skcomplex.engine.backends._numba_stencils import neighbor_value
from skcomplex.engine.solvers import step_count
"""


def _helpers_fingerprint() -> str:
    """Return a digest of the compiled helpers that generated kernels import and of the Numba version.

    Numba caches a kernel with the compiled helpers it calls, but checks only the kernel's own source, so a kernel
    cached against other helpers, such as those of another version of the library, would run stale code. The digest keys
    the cached kernels on the helpers too, and on the module of the step count that the stepper loops call.
    """
    modules = (_numba_neighbours, _numba_stencils, sys.modules[step_count.__module__])
    sources = [inspect.getsource(module).encode() for module in modules]
    return hashlib.sha256(b''.join([numba.__version__.encode(), *sources])).hexdigest()


def _load(text: str, functions: Mapping[str, object], cache_dir: Path) -> ModuleType:
    """Return the module of generated source, from a cached file when it has no injected functions."""
    digest = hashlib.sha256((HELPERS_FINGERPRINT + text).encode()).hexdigest()[:24]
    name = f'skcomplex_kernels_{digest}'
    if functions:
        module = ModuleType(name)
        module.__dict__.update(functions)
        code = compile(text.replace('cache=CACHE', 'cache=False'), name, 'exec')
        exec(code, module.__dict__)  # noqa: S102  # nosec B102: runs the kernel source generated above
        return module
    if name in sys.modules:
        return sys.modules[name]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{name}.py'
    if not path.exists():
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(text.replace('cache=CACHE', 'cache=True'))
        temporary.replace(path)
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)  # type: ignore[arg-type]
    sys.modules[name] = module
    specification.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _search_entries(search: object) -> dict[str, Entry]:
    """Return the entries a neighbourhood search reads, the edge rows of a changing layer or positions and headings."""
    if isinstance(search, EdgeRows):
        return {entry.name: entry for entry in (search.source, search.target, search.active)}
    if not isinstance(search, MetricSearch):
        return {}
    found = {search.positions.name: search.positions}
    if search.heading is not None:
        found[search.heading.entry.name] = search.heading.entry
    return found


def _layout_entries(used: Mapping[type, list[Kernel]], reductions: ReductionSchedule) -> dict[str, Entry]:
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
        entries.update(_search_entries(neighbourhood.search))
    return entries


HELPERS_FINGERPRINT = _helpers_fingerprint()


class NumbaBackend(BaseBackend):
    """Backend that generates fused Numba kernels and compiled stepper loops, cached on disk.

    Kinds of laws that contain a plain Python callable fall back to the NumPy functions, and the problem then has no
    compiled stepper loops.

    Args:
        cache_dir:
            The directory of the generated kernel sources and Numba's cache, or `None` for `~/.cache/skcomplex`.

        parallel:
            Whether member and pair loops over groups of at least 1000 members run in parallel threads. Kernels that
            the compiled loops of jump processes and discrete maps call once per event always run serially.

    Examples:
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationComponent
        >>> system = System(
        ...     components=[PopulationComponent.group('herds', 2, n=[1.0, 2.0])],
        ...     interactions=[ExponentialGrowthInteraction('growth', target='herds', r=3.0)],
        ... )
        >>> problem = compile_system(system, backend='numba')
        >>> problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist()
        [3.0, 6.0]
    """

    supports: ClassVar[Supports] = Supports(
        time=set(TimeKind),
        space=set(SpaceKind.registered()),
        laws=set(LawKind.registered()),
        domains=NUMERIC_DOMAINS | {DomainKind.CATEGORICAL},
        components=set(ComponentKind),
        noise=set(NoiseKind),
        orders=set(OrderKind),
    )

    name: ClassVar[str] = 'numba'

    def __init__(self, cache_dir: str | Path | None = None, parallel: bool = True) -> None:
        self.cache_dir = cache_dir
        self.parallel = parallel

    def _is_parallel(self, kernel: Kernel, form: Form) -> bool:
        """Check whether a kernel runs in parallel threads: outside event loops, over groups of many members."""
        if not self.parallel or form in EVENT_FORMS:
            return False
        rows = [
            reference.entry.rows
            for reference in kernel.references.values()
            if reference.source in {Source.MEMBER, Source.FIRST, Source.SECOND}
        ]
        return max(rows, default=0) >= AUTO_MEMBERS

    def _kind(
        self,
        source: _Source,
        kernels: list[Kernel],
        label: str,
        mode: str,
        *,
        form: Form,
        reduced: bool,
    ) -> tuple[str, str]:
        """Add the kernels of one law kind and return the top-level function name and its output size."""
        names = []
        for index, kernel in enumerate(kernels):
            name = f'_{label}{index}_'
            parallel = self._is_parallel(kernel, form)
            if isinstance(kernel.pairs, EdgeSearch):
                _edge_kernel(source, kernel, name, mode)
            elif kernel.pairs is not None:
                _pair_kernel(source, kernel, name, parallel)
            else:
                _member_kernel(source, kernel, name, mode, parallel)
            names.append(name)
        size = ' + '.join(_output_count(source, kernel) for kernel in kernels) or '0'
        _entry_function(source, label, names, mode, size, reduced=reduced)
        return label, size

    def _run(
        self,
        kernels: Mapping[type, list[Kernel]],
        changes: Mapping[int, list[tuple[Reference, float]]],
        *,
        reductions: ReductionSchedule | None = None,
        draws: Draws | None = None,
    ) -> Lowering:
        """Return the numeric functions of the prepared form compiled with Numba.

        Args:
            kernels:
                The kernels of every law kind, in evaluation order.

            changes:
                The changes of every jump output, keyed by the output's identity.

            reductions:
                The reductions the kernels read, fused into one compiled function that every entry function runs
                first, or `None`.

            draws:
                The random values a stochastic map draws at every step into the parameter vector extended after its
                parameters, which the compiled loop takes from the generator, or `None` for a deterministic problem.

        Returns:
            lowering:
                The numeric functions.
        """
        form, size = self.form_, self.size_
        used = {
            BoundaryLaw: kernels.get(BoundaryLaw, []),
            RateLaw: kernels.get(RateLaw, []) if form in (Form.ODE, Form.FRACTIONAL, Form.SDE) else [],
            NoiseLaw: kernels.get(NoiseLaw, []) if form == Form.SDE else [],
            JumpLaw: kernels.get(JumpLaw, []) if form == Form.JUMP else [],
            UpdateLaw: kernels.get(UpdateLaw, []) if form == Form.DISCRETE else [],
        }
        reductions = reductions or ReductionSchedule()
        reduced = bool(reductions.steps)
        fused = _reductions_compile(reductions)
        compiled = {kind: fused and all(_is_compilable(kernel) for kernel in items) for kind, items in used.items()}
        backends = report_fallbacks(self.name, used, compiled, reductions)
        source = _Source()
        if reduced and fused:
            _reduce_function(source, reductions, self.parallel and form not in EVENT_FORMS)
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
        for kind in needed:
            if compiled[kind]:
                self._kind(source, used[kind], *plan[kind], form=form, reduced=reduced and reads_reductions(used[kind]))
        accelerated_loop = {
            Form.ODE: 'rk4',
            Form.SDE: 'euler_maruyama',
            Form.JUMP: 'gillespie',
            Form.DISCRETE: 'iterate' if draws is None else 'iterate_random',
        }.get(form, '')
        has_loop = bool(accelerated_loop) and all(compiled[kind] for kind in needed)
        bounded = 'HAS_BOUNDARY = True\n' if BoundaryLaw in needed else NO_BOUNDARY
        loops = (
            [bounded, LOOPS[accelerated_loop], LOOPS['velocity_verlet'] if form == Form.ODE else ''] if has_loop else []
        )
        text = '\n'.join([HEADER, *source.lines, *loops]) + '\n'
        cache_dir = Path(self.cache_dir) if self.cache_dir is not None else DEFAULT_CACHE_DIR
        module = _load(text, source.functions, cache_dir)
        entries = _layout_entries(used, reductions)
        layout = source.layout(entries)
        loop = accelerated_loop if has_loop else None
        lowering = self._lowering(
            form,
            needed,
            used=used,
            compiled=compiled,
            changes=changes,
            size=size,
            module=module,
            layout=layout,
            loop=loop,
            reductions=reductions,
            draws=draws,
        )
        lowering.backends = backends
        return lowering

    def _lowering(
        self,
        form: Form,
        needed: list[type],
        *,
        used: Mapping[type, list[Kernel]],
        compiled: Mapping[type, bool],
        changes: Mapping[int, list[tuple[Reference, float]]],
        size: int,
        module: ModuleType,
        layout: np.ndarray,
        loop: str | None,
        reductions: ReductionSchedule,
        draws: Draws | None,
    ) -> Lowering:
        """Return the lowering that binds the compiled functions to the layout, with NumPy fallbacks."""

        def bind(name: str) -> Callable[[float, Vector, Vector], Vector]:
            function = getattr(module, name)
            return lambda t, x, p: function(t, x, p, layout)

        numpy = {
            RateLaw: lambda: adding_function(used[RateLaw], size),
            NoiseLaw: lambda: concatenating_function(used[NoiseLaw]),
            JumpLaw: lambda: concatenating_function(used[JumpLaw]),
            UpdateLaw: lambda: assigning_function(used[UpdateLaw]),
            BoundaryLaw: lambda: assigning_function(used[BoundaryLaw]),
        }
        names = {
            RateLaw: 'drift',
            NoiseLaw: 'diffusion',
            JumpLaw: 'propensities',
            UpdateLaw: 'update',
            BoundaryLaw: 'boundary',
        }
        functions: dict[type, Any] = {
            kind: (
                bind(names[kind])
                if compiled[kind]
                else reductions.wrap(numpy[kind]()) if reads_reductions(used[kind]) else numpy[kind]()
            )
            for kind in needed
        }
        lowering = Lowering(boundary=functions.get(BoundaryLaw))
        if form in (Form.ODE, Form.FRACTIONAL, Form.SDE):
            lowering.drift = functions[RateLaw]
        if form == Form.SDE:
            lowering.diffusion = functions.get(NoiseLaw, concatenating_function([]))
            noise = lower_noise(used[NoiseLaw])
            lowering.noise_rows, lowering.noise_columns, lowering.n_processes, lowering.noise_blocks = noise
        if form == Form.JUMP:
            lowering.propensities = functions.get(JumpLaw, concatenating_function([]))
            lowering.jumps = jump_channels(used[JumpLaw], changes)
        if form == Form.DISCRETE:
            lowering.update = functions[UpdateLaw]
        if loop is not None:
            lowering.accelerated = self._accelerated(loop, module, layout, lowering, draws)
        return lowering

    def _accelerated(
        self,
        loop: str,
        module: ModuleType,
        layout: np.ndarray,
        lowering: Lowering,
        draws: Draws | None,
    ) -> Accelerated:
        """Return the compiled stepper loop of a form bound to the layout."""
        function = getattr(module, loop)
        if loop == 'rk4':
            verlet = module.velocity_verlet
            return Accelerated(
                rk4=lambda t, x, p, until, h: function(t, x, p, layout, until, h),
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
            noise = _noise_arrays(lowering)
            return Accelerated(
                euler_maruyama=lambda t, x, p, until, h, rng: function(t, x, p, layout, until, h, rng, *noise),
            )
        if loop == 'gillespie':
            targets, increments = _channel_arrays(lowering)
            return Accelerated(
                gillespie=lambda t, x, p, until, rng: function(t, x, p, layout, until, rng, targets, increments),
            )
        if loop == 'iterate_random' and draws is not None:
            offset, flags = draws.offset, draws.normal
            return Accelerated(iterate=lambda t, x, p, until, rng: function(t, x, p, layout, until, rng, offset, flags))
        return Accelerated(iterate=lambda t, x, p, until: function(t, x, p, layout, until))


register_backend(NumbaBackend)
