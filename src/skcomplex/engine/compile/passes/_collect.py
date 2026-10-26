"""Collect the laws and the objectives every component declares for its kernel spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._census import declared_laws, declared_objectives
from .._compilation import Compilation
from .._declarations import Block
from .._pipeline import BasePass
from .._specs import KernelSpec, component_of


class CollectPass(BasePass):
    """Pass that collects the declared laws and objectives of every kernel spec, with no laws set apart yet."""

    reads = ('specs',)
    writes = ('collected', 'apart', 'objectives')

    def _run(self, compilation: Compilation) -> None:
        """Collect the laws and the objectives of every kernel spec.

        Args:
            compilation:
                The compilation state.
        """
        specs = cast('list[tuple[KernelSpec, Block]]', compilation.specs)
        compilation.collected = [(spec, declared_laws(spec, component_of(block.node)), block) for spec, block in specs]
        compilation.apart = []
        compilation.objectives = [(spec, declared_objectives(spec, component_of(block.node))) for spec, block in specs]
