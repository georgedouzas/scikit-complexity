"""Collect the laws and the quantities every component declares for its kernel spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.base import BaseElement, BaseInteraction
from .._census import declared_laws, declared_quantities
from .._compilation import Compilation
from .._declarations import Block
from .._pipeline import BasePass
from .._specs import KernelSpec, component_of


class CollectPass(BasePass):
    """Pass that collects the declared laws and quantities of every kernel spec, with no laws set apart yet."""

    reads = ('specs',)
    writes = ('collected', 'apart', 'quantities')

    def _run(self, compilation: Compilation) -> None:
        """Collect the laws and the quantities of every kernel spec.

        Args:
            compilation:
                The compilation state.
        """
        specs = cast('list[tuple[KernelSpec, Block]]', compilation.specs)
        compilation.collected = [(spec, declared_laws(spec, component_of(block.node)), block) for spec, block in specs]
        compilation.apart = []
        compilation.quantities = [
            (spec, declared_quantities(spec, cast('BaseElement | BaseInteraction', component_of(block.node))))
            for spec, block in specs
        ]
