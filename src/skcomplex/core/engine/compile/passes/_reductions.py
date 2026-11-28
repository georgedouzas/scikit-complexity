"""Compute every sum, mean, or other reduction of the laws once per evaluation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from ....components.laws import EventLaw
from ...enums import ReadKind
from ...problems._kernel_spec import KernelSpec
from ...problems._neighborhood_kernel import _ReductionSchedule
from .._base import BasePass
from .._compilation import Compilation, _Declarations
from ._reduction_lowering import ReductionLowering


class ReductionsPass(BasePass):
    """A pass that makes every reduction of the laws computed once per evaluation.

    A law may read a sum or a mean over a group, such as the mean phase of the oscillators that pulls every oscillator.
    The pass replaces every reduction by a symbol, so it is computed once and then read by every member.

    Read more in the user guide, under Extend the library.

    See Also:
        ReductionLowering: Rewrites the reductions.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.synchronization import KuramotoSystem
        >>> compilation = PassCompiler().prepare(KuramotoSystem()).run().compilation_
        >>> [(reduction.group, reduction.kind) for reduction in compilation.reductions.reductions]
        [('oscillators', 'count'), ('oscillators', 'sum'), ('oscillators', 'sum')]
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn', 'drawn_apart', 'layout', 'draws', 'events')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('reduced', 'reduced_apart', 'reductions', 'events')

    def _run(self, inputs: object = None) -> None:
        """Lower the reductions of the drawn laws, the laws set apart, and the events."""
        compilation = self.compilation_
        declarations = Compilation._get_required(compilation.layout, 'layout')
        reductions = ReductionLowering().prepare(compilation)
        specs = Compilation._get_required(compilation.drawn, 'drawn')
        apart = Compilation._get_required(compilation.drawn_apart, 'drawn_apart')
        crossings = Compilation._get_required(compilation.events, 'events')
        lowered = reductions.run([*specs, *apart]).lowered_
        events = reductions.run([(spec, [event]) for spec, event in crossings]).lowered_
        compilation.reduced = lowered[: len(specs)]
        compilation.reduced_apart = lowered[len(specs) :]
        compilation.events = [(spec, law) for spec, (law, *_) in events if isinstance(law, EventLaw)]
        reductions._apply_references()
        compilation._replace_specs(
            {
                id(spec): self._get_pruned(spec, declarations)
                for spec, _ in [*compilation.reduced, *compilation.reduced_apart]
            },
        )
        compilation.reductions = _ReductionSchedule(
            tuple(reductions.found_.values()),
            reductions.parameters_,
            tuple(reductions.neighbourhoods_.values()),
            tuple(reductions.sequence_),
        )

    @staticmethod
    def _get_pruned(spec: KernelSpec, declarations: _Declarations) -> KernelSpec:
        """Return a spec without the references of the groups read only in reductions.

        Operands are kept. Kernels then loop over its own group.

        A spec without such references is returned itself.
        """
        if spec.pairs is not None:
            return spec
        owners = {
            reference.entry.name.rpartition('.')[0]
            for reference in spec.references.values()
            if reference.source is ReadKind.MEMBER
        }
        foreign = (owners & declarations.groups) - {spec.group, *spec.aligned}
        if not foreign:
            return spec
        operands = {
            symbol
            for reference in spec.references.values()
            if reference.source is ReadKind.DERIVED
            for symbol in reference.derived.symbols
        }
        references = {
            item: reference
            for item, reference in spec.references.items()
            if reference.source is not ReadKind.MEMBER
            or reference.entry.name.rpartition('.')[0] not in foreign
            or item in operands
        }
        return replace(spec, references=references)
