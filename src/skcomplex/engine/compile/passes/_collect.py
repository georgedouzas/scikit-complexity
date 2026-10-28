"""Collect the laws every component declares for its kernel spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....specs import BlockSpec, KernelSpec
from .._base import BasePass
from ._census import CensusPass


class CollectPass(BasePass):
    """Pass that collects the declared laws of every kernel spec, with no laws set apart yet."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('specs',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected', 'apart')

    def _run(self, inputs: object = None) -> None:
        """Collect the laws of every kernel spec."""
        compilation = self.compilation_
        specs = cast('list[tuple[KernelSpec, BlockSpec]]', compilation.specs)
        compilation.collected = [
            (spec, CensusPass._declared_laws(spec, CensusPass._component_of(block.node)), block)
            for spec, block in specs
        ]
        compilation.apart = []
