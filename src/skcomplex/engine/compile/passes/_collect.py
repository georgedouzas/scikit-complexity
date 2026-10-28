"""Collect the laws every component declares for its kernel spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._compilation import Compilation
from .._declarations import Block
from .._pipeline import BasePass
from .._specs import KernelSpec
from ._census import CensusPass


class CollectPass(BasePass):
    """Pass that collects the declared laws of every kernel spec, with no laws set apart yet."""

    reads = ('specs',)
    writes = ('collected', 'apart')

    def _run(self, compilation: Compilation) -> None:
        """Collect the laws of every kernel spec.

        Args:
            compilation:
                The compilation state.
        """
        specs = cast('list[tuple[KernelSpec, Block]]', compilation.specs)
        compilation.collected = [
            (spec, CensusPass._declared_laws(spec, CensusPass._component_of(block.node)), block)
            for spec, block in specs
        ]
        compilation.apart = []
