"""Build the kernel spec of every component of a system, the namespace and the rows its laws are lowered with."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._compilation import Compilation
from .._declarations import Declarations, is_interaction
from .._observables import Observer, lowerable_observables
from .._pipeline import BasePass
from .._specs import specs_of
from .._tree import Node


class SpecsPass(BasePass):
    """Pass that builds the kernel specs of every component, one per element set entry of an interaction."""

    reads = ('layout', 'nodes')
    writes = ('specs',)

    def _run(self, compilation: Compilation) -> None:
        """Build the kernel specs of the declared components.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        blocks = list(declarations.blocks.values())
        elements = {block.node.path: block for block in blocks if not is_interaction(block.node)}
        nodes = cast('list[Node]', compilation.nodes)
        observe = Observer(declarations, lowerable_observables(compilation.system, nodes))
        compilation.specs = [
            (spec, block) for block in blocks for spec in specs_of(block, elements, declarations.time, observe.evaluate)
        ]
