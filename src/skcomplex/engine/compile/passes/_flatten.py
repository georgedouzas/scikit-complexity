"""Flatten the system tree into the nodes of its elements, groups, and interactions, once per compilation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from .._compilation import Compilation
from .._pipeline import Pass
from .._tree import flatten_system


class FlattenPass(Pass):
    """Pass that flattens the system tree into its nodes."""

    reads = ('system',)
    writes = ('nodes',)

    def run(self, compilation: Compilation) -> None:
        """Flatten the system into its nodes.

        Args:
            compilation:
                The compilation state.
        """
        compilation.nodes = flatten_system(compilation.system)
