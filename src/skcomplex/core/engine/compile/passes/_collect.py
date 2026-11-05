"""Collect the laws every component declares for its kernel spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....components.laws import LinearProgramLaw
from .._base import BasePass
from .._compilation import Compilation
from ._census import CensusPass


class CollectPass(BasePass):
    """Pass that collects the declared laws of every kernel spec, with no laws set apart yet."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('specs',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected', 'apart', 'programs')

    def _run(self, inputs: object = None) -> None:
        """Collect the laws of every kernel spec."""
        compilation = self.compilation_
        specs = Compilation._required(compilation.specs, 'specs')
        declared = [
            (spec, CensusPass._declared_laws(spec, CensusPass._component_of(block.node)), block)
            for spec, block in specs
        ]
        compilation.collected = [
            (spec, [law for law in laws if not isinstance(law, LinearProgramLaw)], block)
            for spec, laws, block in declared
        ]
        compilation.apart = []
        compilation.programs = [
            (spec, [law for law in laws if isinstance(law, LinearProgramLaw)]) for spec, laws, _ in declared
        ]
