"""Replace the reductions and neighbourhood reductions of laws by placeholders, compiled once each."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import CrossingLaw
from ...problems import ReductionSchedule, Source
from .._base import BasePass
from .._compilation import Laws
from .._declarations import Declarations
from .._kernel_spec import KernelSpec
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
        declarations = cast('Declarations', compilation.layout)
        reductions = ReductionLowering().check().prepare(compilation)
        specs, apart = cast('Laws', compilation.drawn), cast('Laws', compilation.drawn_apart)
        lowered = reductions.run([*specs, *apart])
        events = reductions.run([(spec, [event]) for spec, event in cast('list[tuple[KernelSpec, CrossingLaw]]', compilation.events)])
        compilation.events = [(spec, cast('CrossingLaw', laws[0])) for spec, laws in events]
        for spec, _ in lowered:
            self._prune_foreign(spec, declarations)
        compilation.reduced = lowered[: len(specs)]
        compilation.reduced_apart = lowered[len(specs) :]
        compilation.reductions = ReductionSchedule(
            tuple(reductions.found_.values()),
            reductions.parameters_,
            tuple(reductions.neighbourhoods_.values()),
            tuple(reductions.sequence_),
        )

    @staticmethod
    def _prune_foreign(spec: KernelSpec, declarations: Declarations) -> None:
        """Remove the references of groups that a spec reads only through reductions, so kernels loop over its group."""
        if spec.pairs is not None:
            return
        owners = {
            reference.entry.name.rpartition('.')[0]
            for reference in spec.references.values()
            if reference.source is Source.MEMBER
        }
        foreign = (owners & declarations.groups) - {spec.group, *spec.aligned}
        if not foreign:
            return
        spec.references = {
            item: reference
            for item, reference in spec.references.items()
            if reference.source is not Source.MEMBER or reference.entry.name.rpartition('.')[0] not in foreign
        }
