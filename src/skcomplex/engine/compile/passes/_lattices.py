"""Compile the laws of lattices: local rules, drives, relaxations, extremal updates, and Monte Carlo sampling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from ....core.errors import ValidationError
from ....core.laws import BaseLaw, DriveLaw, ExtremalLaw, LocalRuleLaw, RelaxationLaw, SampleLaw
from ....core.spaces import CayleySpace, LatticeSpace, NetworkSpace
from ....core.specs import SamplerSpec
from ...backends import resolve_backend
from ...problems import Problem, Source
from .._census import Census
from .._compilation import Compilation, Laws
from .._declarations import Declarations, member_count
from .._issues import Issue
from .._lattice_steps import (
    LatticeContext,
    LatticeState,
    LatticeStep,
    compile_drive,
    compile_extremal,
    compile_monte_carlo,
    compile_relaxation,
    compile_rule,
)
from .._laws import LATTICE_LAWS
from .._pipeline import BasePass
from .._specs import KernelSpec


def _lattice_context(spec: KernelSpec, declarations: Declarations, fused: bool) -> LatticeContext:
    """Return the lattice states, the arguments, and the neighbour symbols of the symbols a component's laws read."""
    states, arguments, others = {}, {}, {}
    for item, reference in spec.references.items():
        entry = reference.entry
        owner, _, variable = entry.name.rpartition('.')
        block = declarations.blocks.get(owner)
        space = block.node.space if block is not None else None
        is_lattice = (
            isinstance(space, LatticeSpace | NetworkSpace | CayleySpace)
            and block is not None
            and block.node.members is not None
            and not block.node.is_agent
        )
        if reference.is_state and is_lattice and not entry.member_shape:
            states[item] = LatticeState(entry.name, entry.start, space)  # type: ignore[arg-type]
        if not entry.member_shape:
            arguments[item] = (variable, reference.is_state, entry.start, entry.size)
        if reference.source is Source.OTHER:
            others[item] = entry.name
    return LatticeContext(spec.owner, states, arguments, fused, others)


def _monte_carlo_step(spec: KernelSpec, law: SampleLaw, context: LatticeContext) -> LatticeStep:
    """Return the Monte Carlo sampling of a sample, with Metropolis acceptance at `temperature` without a sampler."""
    sampler = law.sampler
    if sampler is None:
        if not hasattr(spec.namespace, 'temperature'):
            error_msg = f'{spec.owner} declares a sample without a sampler and no parameter `temperature`.'
            raise ValidationError(error_msg)
        sampler = SamplerSpec(temperature=spec.namespace.temperature)
    return compile_monte_carlo(law.energy, law.move, sampler, context)


def _lattice_step(spec: KernelSpec, law: BaseLaw, context: LatticeContext) -> LatticeStep | None:
    """Return the step of a lattice law, or `None` for a law of another kind."""
    if isinstance(law, LocalRuleLaw):
        return compile_rule(law, context)
    if isinstance(law, DriveLaw):
        return compile_drive(law, context)
    if isinstance(law, RelaxationLaw):
        return compile_relaxation(law, context)
    if isinstance(law, ExtremalLaw):
        return compile_extremal(law, context)
    if isinstance(law, SampleLaw):
        return _monte_carlo_step(spec, law, context)
    return None


def _compile_lattice(
    specs: list[tuple[KernelSpec, list[BaseLaw]]],
    declarations: Declarations,
    fused: bool,
) -> tuple[LatticeStep, ...]:
    """Return the steps of local rules, drives, relaxations, extremal laws, and samples in declaration order."""
    steps: list[LatticeStep] = []
    for spec, laws in specs:
        if not any(isinstance(law, LATTICE_LAWS) for law in laws):
            continue
        context = _lattice_context(spec, declarations, fused)
        found = [_lattice_step(spec, law, context) for law in laws]
        steps += [step for step in found if step is not None]
    return tuple(steps)


def _schedule_of(law: object) -> str | None:
    """Return the schedule of a local rule or of the sampler of a sample, the default sampler's without one."""
    if isinstance(law, LocalRuleLaw):
        return law.schedule
    if isinstance(law, SampleLaw):
        return (law.sampler or SamplerSpec()).schedule
    return None


def schedule_issues(compilation: Compilation) -> list[Issue]:
    """Return the lattice laws whose checkerboard schedule needs a square lattice that their space is not."""
    records = cast('Census', compilation.census).records
    issues = []
    for record in records:
        square = isinstance(record.space, LatticeSpace) and record.space.geometry == 'square'
        if not square and _schedule_of(record.law) == 'checkerboard':
            message = (
                f'The {type(record.law).__name__} of {record.path} uses the checkerboard schedule, which needs a '
                f'square lattice, but the component lives on {record.space!r}.'
            )
            fix = (
                "Use the 'synchronous' or the 'random-sequential' schedule of a local rule, or the "
                "'random-sequential' or the 'colored' schedule of a sampler."
            )
            issues.append(Issue('schedule-unsupported', record.path, message, fix))
    return issues


class LatticeProblemPass(BasePass):
    """Pass that adds the steps of local rules, drives, relaxations, extremal updates, and Monte Carlo sampling."""

    reads = ('problem', 'reduced', 'layout')
    writes = ('problem',)

    def _run(self, compilation: Compilation) -> None:
        """Add the lattice steps to the problem.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        members = member_count(list(declarations.blocks.values()))
        fused = resolve_backend(compilation.backend, members).name == 'numba'
        compilation.problem = replace(
            cast('Problem', compilation.problem),
            lattice=_compile_lattice(cast('Laws', compilation.reduced), declarations, fused),
        )
