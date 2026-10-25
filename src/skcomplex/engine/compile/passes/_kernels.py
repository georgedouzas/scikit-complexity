"""Lower the laws of every kernel spec into kernels, one per law kind and component."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from typing import cast

from ....core.base import BaseLaw, Expression, Layout, Trait, ValidationError, components_of, has_trait, symbols_of
from ....core.laws import BoundaryLaw, Call, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.problems import Kernel, Lowered, Output, Reference, Source
from .._census import kind_of
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._lowering import symbolic_components
from .._noise import noise_process, process_keys
from .._pipeline import BasePass
from .._specs import KernelSpec, target_references

ASSIGNED_KINDS = (UpdateLaw, BoundaryLaw)


def _outputs_of(value: object, targets: list[Reference | None], owner: str) -> list[Output]:
    """Return the outputs of one law value for its target components, with the law and component as owner."""
    if isinstance(value, Call):
        arguments = tuple(
            item for argument in value.arguments for item in symbolic_components(argument, 1, owner) or []
        )
        if len(arguments) != len(value.arguments):
            error_msg = f'A Call in a {owner} has an argument that is not a symbolic expression.'
            raise ValidationError(error_msg)
        return [
            Output(target, function=value.function, arguments=arguments, component=index, components=len(targets))  # type: ignore[arg-type]
            for index, target in enumerate(targets)
        ]
    expressions = symbolic_components(value, len(targets), owner)
    if expressions is None:
        return [
            Output(target, function=value, component=index, components=len(targets))  # type: ignore[arg-type]
            for index, target in enumerate(targets)
        ]
    return [Output(target, expression) for target, expression in zip(targets, expressions, strict=True)]


def _check_assignments(
    spec: KernelSpec,
    law: BaseLaw,
    targets: list[Reference],
    assigned: set[tuple[str, int]],
    *,
    once: bool,
) -> None:
    """Raise an error for an update or boundary law that sums over members or assigns a component twice."""
    kind = type(law).__name__
    for target in targets:
        if target.source in {Source.FIRST, Source.SECOND}:
            error_msg = (
                f'A {kind} law of {spec.owner} sets {target.entry.name} at an end of every edge, which several edges '
                'share. Set the edge member, or reduce over the neighbours in the laws of the nodes.'
            )
            raise ValidationError(error_msg)
        if target.source is Source.SINGLE and spec.rows > 1 and not once:
            error_msg = f'A {kind} law of {spec.owner} targets a single element from a group, which has no meaning.'
            raise ValidationError(error_msg)
        key = (target.entry.name, target.column)
        if key in assigned:
            error_msg = f'A {kind} law of {spec.owner} sets {target.entry.name} a second time. Each takes one law.'
            raise ValidationError(error_msg)
        assigned.add(key)


def _is_once(spec: KernelSpec, law: BaseLaw) -> bool:
    """Check whether a law of a group's interaction sets single elements from reductions alone, so it runs once."""
    if spec.rows <= 1 or spec.pairs is not None or not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
        return False
    if not all(reference.source is Source.SINGLE for reference in target_references(law.target, spec)):
        return False
    expressions = [item for item in components_of(law.value) if isinstance(item, Expression)]
    sources = {spec.references[item].source for item in symbols_of(expressions) if item in spec.references}
    return Source.REDUCED in sources and Source.MEMBER not in sources


def _single_references(spec: KernelSpec) -> dict[Expression, Reference]:
    """Return the references of a spec without its group members, for laws that run once."""
    return {
        item: reference
        for item, reference in spec.references.items()
        if reference.source is not Source.MEMBER and reference.entry.members is None
    }


def _law_outputs(
    spec: KernelSpec,
    law: RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw,
    targets: list[Reference],
    declarations: Declarations,
    lowered: dict[int, list[Output]],
) -> list[Output]:
    """Return the outputs of a law for its target components, with the named Wiener process of a noise law.

    A law that a pass of a formalism lowered takes the outputs of that pass.
    """
    if id(law) in lowered:
        return lowered[id(law)]
    if isinstance(law, NoiseLaw) and law.conserved:
        error_msg = (
            f'A conserved noise law of {spec.owner} has no lowering. Declare its state on a `Field` of the fields '
            'formalism, or remove `conserved=True`.'
        )
        raise ValidationError(error_msg)
    outputs = _outputs_of(law.value, targets, f'{type(law).__name__} of {spec.owner}')  # type: ignore[arg-type]
    if isinstance(law, NoiseLaw) and law.process is not None:
        for item, key in zip(outputs, process_keys(spec, law, declarations), strict=True):
            item.process = noise_process(key, declarations.wieners)
    return outputs


def _lower_laws(specs: Laws, declarations: Declarations, lowered_outputs: dict[int, list[Output]]) -> Lowered:
    """Return the kernels of every law kind, with the law of every output."""
    states, parameters = Layout(declarations.states), Layout(declarations.parameters)
    lowered = Lowered()
    assigned: dict[type, set[tuple[str, int]]] = defaultdict(set)
    for spec, laws in specs:
        outputs: dict[tuple[type, bool], list[Output]] = defaultdict(list)
        for law in laws:
            if has_trait(law, Trait.KERNEL_FREE):
                continue
            if isinstance(law, JumpLaw):
                output = _outputs_of(law.rate, [None], f'JumpLaw of {spec.owner}')[0]
                lowered.jump_changes[id(output)] = [
                    (target_references(target, spec)[0], float(increment)) for target, increment in law.changes.items()
                ]
                lowered.laws[id(output)] = (spec, law)
                outputs[(JumpLaw, False)].append(output)
                continue
            if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
                error_msg = f'A {type(law).__name__} of {spec.owner} reached lowering before it was rewritten.'
                raise ValidationError(error_msg)
            targets = target_references(law.target, spec)
            once = _is_once(spec, law)
            if isinstance(law, ASSIGNED_KINDS):
                _check_assignments(spec, law, targets, assigned[type(law)], once=once)
            law_outputs = _law_outputs(spec, law, targets, declarations, lowered_outputs)
            lowered.laws.update({id(item): (spec, law) for item in law_outputs})
            outputs[(kind_of(law), once)].extend(law_outputs)
        for (kind, once), kind_outputs in outputs.items():
            lowered.kernels[kind].append(
                Kernel(
                    spec.owner,
                    declarations.time,
                    _single_references(spec) if once else spec.references,
                    1 if once else spec.rows,
                    states,
                    parameters,
                    spec.pairs,
                    kind_outputs,
                ).check(),
            )
    return lowered


class KernelsPass(BasePass):
    """Pass that lowers the reduced laws into the kernels of every law kind."""

    reads = ('reduced', 'layout', 'outputs')
    writes = ('kernels',)

    def run(self, compilation: Compilation) -> None:
        """Lower the reduced laws into kernels.

        Args:
            compilation:
                The compilation state.
        """
        compilation.kernels = _lower_laws(
            cast('Laws', compilation.reduced),
            cast('Declarations', compilation.layout),
            compilation.outputs,
        )
