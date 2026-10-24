"""Collect the laws every component declares for its kernel spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._census import declared_laws
from .._compilation import Compilation
from .._declarations import Block
from .._pipeline import Pass
from .._specs import KernelSpec, component_of


class CollectPass(Pass):
    """Pass that collects the declared laws of every kernel spec, with no laws set apart yet."""

    reads = ('specs',)
    writes = ('collected', 'apart')

    def run(self, compilation: Compilation) -> None:
        """Collect the laws of every kernel spec.

        Args:
            compilation:
                The compilation state.
        """
        specs = cast('list[tuple[KernelSpec, Block]]', compilation.specs)
        compilation.collected = [(spec, declared_laws(spec, component_of(block.node)), block) for spec, block in specs]
        compilation.apart = []
