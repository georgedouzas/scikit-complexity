"""Take the census of the laws every component of a system declares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._census import census_of
from .._compilation import Compilation
from .._pipeline import Pass
from .._tree import Node


class CensusPass(Pass):
    """Pass that takes the census of the laws of the system, keeping going past failing components."""

    reads = ('nodes',)
    writes = ('census',)

    def run(self, compilation: Compilation) -> None:
        """Take the census of the laws of the nodes.

        Args:
            compilation:
                The compilation state.
        """
        compilation.census = census_of(compilation.system, cast('list[Node]', compilation.nodes))
