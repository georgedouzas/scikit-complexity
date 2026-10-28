"""Replace the reductions and neighbourhood reductions of laws by placeholders, compiled once each."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from ....core.laws import CrossingLaw
from ....specs import DeclarationsSpec, KernelSpec, ReductionScheduleSpec
from ....types import ReadKind
from .._base import BasePass
from .._compilation import Laws
from ._reduction_lowering import ReductionLowering


class ReductionsPass(BasePass):
    """Pass that lowers the reductions of the laws, of the laws set apart, and of the events, in that order."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn', 'drawn_apart', 'layout', 'drawing', 'events')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('reduced', 'reduced_apart', 'reductions', 'events')

    def _run(self, inputs: object = None) -> None:
        """Lower the reductions of the drawn laws, the laws set apart, and the events."""
        compilation = self.compilation_
        declarations = cast('DeclarationsSpec', compilation.layout)
        reductions = ReductionLowering().check().prepare(compilation)
        specs, apart = cast('Laws', compilation.drawn), cast('Laws', compilation.drawn_apart)
        lowered = reductions.run([*specs, *apart])
        crossings = cast('list[tuple[KernelSpec, CrossingLaw]]', compilation.events)
        events = reductions.run([(spec, [event]) for spec, event in crossings])
        pruned = {id(spec): self._pruned(spec, declarations) for spec, _ in lowered}
        compilation.events = [(pruned.get(id(spec), spec), cast('CrossingLaw', laws[0])) for spec, laws in events]
        lowered = [(pruned[id(spec)], laws) for spec, laws in lowered]
        compilation.reduced = lowered[: len(specs)]
        compilation.reduced_apart = lowered[len(specs) :]
        compilation.delays = [
            {**delay, 'spec': pruned.get(id(delay['spec']), delay['spec'])} for delay in compilation.delays or []
        ]
        compilation.renewals = [
            replace(renewal, spec=pruned.get(id(renewal.spec), renewal.spec)) for renewal in compilation.renewals or []
        ]
        compilation.reductions = ReductionScheduleSpec(
            tuple(reductions.found_.values()),
            reductions.parameters_,
            tuple(reductions.neighbourhoods_.values()),
            tuple(reductions.sequence_),
        )

    @staticmethod
    def _pruned(spec: KernelSpec, declarations: DeclarationsSpec) -> KernelSpec:
        """Return a spec without the references of the groups it reads only through reductions.

        Kernels then loop over its own group.

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
        references = {
            item: reference
            for item, reference in spec.references.items()
            if reference.source is not ReadKind.MEMBER or reference.entry.name.rpartition('.')[0] not in foreign
        }
        return replace(spec, references=references)
