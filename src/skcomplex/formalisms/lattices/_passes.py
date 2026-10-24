"""Compile the laws of lattices: local rules, drives, relaxations, extremal updates, and Monte Carlo sampling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from dataclasses import replace
from typing import cast

from ...base import Form, Trait, ValidationError
from ...compile import (
    BlockLaws,
    Census,
    Compilation,
    Declarations,
    Issue,
    KernelSpec,
    Laws,
    Pass,
    Problem,
    Source,
    member_count,
    resolve_backend,
    target_references,
)
from ...laws import Law, has_trait
from ...space import NetworkSpace, SiteSpace
from ._laws import Drive, Energy, Extremal, LocalRule, Move, Relaxation, Sample, Sampler
from ._steps import (
    LatticeContext,
    LatticeState,
    LatticeStep,
    compile_drive,
    compile_extremal,
    compile_monte_carlo,
    compile_relaxation,
    compile_rule,
)


def _lattice_context(spec: KernelSpec, declarations: Declarations, fused: bool) -> LatticeContext:
    """Return the lattice states, the arguments, and the neighbour symbols of the symbols a component's laws read."""
    states, arguments, others = {}, {}, {}
    for item, reference in spec.references.items():
        entry = reference.entry
        owner, _, variable = entry.name.rpartition('.')
        block = declarations.blocks.get(owner)
        space = block.node.space if block is not None else None
        is_lattice = (
            isinstance(space, SiteSpace | NetworkSpace)
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


def _monte_carlo_step(spec: KernelSpec, laws: list[Law], context: LatticeContext) -> LatticeStep | None:
    """Return the Monte Carlo sampling of a component's energy, move, and sampler, or `None` without them."""
    energies = [law for law in laws if isinstance(law, Energy)]
    moves = [law for law in laws if isinstance(law, Move)]
    samplers = [law for law in laws if isinstance(law, Sampler)]
    if not (energies or moves or samplers):
        return None
    if len(energies) != 1 or len(moves) != 1 or len(samplers) > 1:
        error_msg = (
            f'{spec.owner} declares {len(energies)} energies, {len(moves)} moves, and {len(samplers)} samplers. '
            'Monte Carlo sampling takes one energy, one move, and at most one sampler.'
        )
        raise ValidationError(error_msg)
    if samplers:
        sampler = samplers[0]
    elif hasattr(spec.namespace, 'temperature'):
        sampler = Sampler(temperature=spec.namespace.temperature)
    else:
        error_msg = f'{spec.owner} declares no sampler and no parameter `temperature` for the default sampler.'
        raise ValidationError(error_msg)
    return compile_monte_carlo(energies[0], moves[0], sampler, context)


def _compile_lattice(
    specs: list[tuple[KernelSpec, list[Law]]],
    declarations: Declarations,
    fused: bool,
) -> tuple[LatticeStep, ...]:
    """Return the steps of local rules, drives, relaxations, and Monte Carlo sampling in declaration order."""
    steps: list[LatticeStep] = []
    compilers = {
        LocalRule: compile_rule,
        Drive: compile_drive,
        Relaxation: compile_relaxation,
        Extremal: compile_extremal,
    }
    for spec, laws in specs:
        if not any(has_trait(law, Trait.LATTICE) for law in laws):
            continue
        context = _lattice_context(spec, declarations, fused)
        steps += [compilers[type(law)](law, context) for law in laws if type(law) in compilers]  # type: ignore[operator]
        sampling = _monte_carlo_step(spec, laws, context)
        if sampling is not None:
            steps.append(sampling)
    return tuple(steps)


def _lattice_owner(spec: KernelSpec, law: Law, agents: set[str]) -> str | None:
    """Return the element or group whose sites a lattice law updates, or `None` for another law.

    Monte Carlo sampling belongs to the component that declares it.
    """
    del agents
    if isinstance(law, Sample | Energy | Sampler):
        return spec.owner
    if isinstance(law, Relaxation):
        target = next(iter(law.targets), None)
        return spec.owner if target is None else target_references(target, spec)[0].entry.name.rpartition('.')[0]
    if isinstance(law, LocalRule | Extremal):
        return target_references(law.targets[0], spec)[0].entry.name.rpartition('.')[0]
    return None


def _schedule_of(law: object) -> str | None:
    """Return the schedule of a local rule or of the sampler of a sample, the default sampler's without one."""
    if isinstance(law, LocalRule):
        return law.schedule
    if isinstance(law, Sample):
        return (law.sampler or Sampler()).schedule
    return None


def _schedule_issues(compilation: Compilation) -> list[Issue]:
    """Return the lattice laws whose checkerboard schedule needs a regular geometry that their space lacks."""
    records = cast('Census', compilation.census).records
    issues = []
    for record in records:
        lacks: frozenset[str] = getattr(record.space, 'lacks', frozenset())
        if 'regular-geometry' in lacks and _schedule_of(record.law) == 'checkerboard':
            message = (
                f'The {type(record.law).__name__} law of {record.path} uses the checkerboard schedule, which needs the '
                f'regular geometry of a square lattice, but the component lives on {type(record.space).__name__}.'
            )
            fix = (
                "Use the 'synchronous' or the 'random-sequential' schedule of a local rule, or the "
                "'random-sequential' or the 'colored' schedule of a sampler."
            )
            issues.append(Issue('schedule-unsupported', record.path, message, fix))
    return issues


class UnpackSamplesPass(Pass):
    """Pass that replaces every Monte Carlo sample by its energy, move, and sampler."""

    reads = ('collected',)
    writes = ('collected',)

    def run(self, compilation: Compilation) -> None:
        """Replace the samples by their parts.

        Args:
            compilation:
                The compilation state.
        """
        compilation.collected = [
            (spec, [part for law in laws for part in (law.parts if isinstance(law, Sample) else (law,))], block)
            for spec, laws, block in cast('BlockLaws', compilation.collected)
        ]


class LatticeProblemPass(Pass):
    """Pass that adds the steps of local rules, drives, relaxations, extremal updates, and Monte Carlo sampling."""

    reads = ('problem', 'reduced', 'layout')
    writes = ('problem',)

    def run(self, compilation: Compilation) -> None:
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


STEPWISE_FORMS: dict[frozenset[type], Form] = {
    **{
        frozenset(kinds): Form.LATTICE
        for count in (1, 2, 3, 4)
        for kinds in itertools.combinations((LocalRule, Drive, Relaxation, Extremal), count)
    },
    frozenset({Energy, Move}): Form.MONTE_CARLO,
    frozenset({Energy, Move, Sampler}): Form.MONTE_CARLO,
}
OWNER_RULES = (_lattice_owner,)
CHECKS = (_schedule_issues,)
