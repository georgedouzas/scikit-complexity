"""Declare the states and parameters of every component of a system in the layout of its problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.base import symbol
from .._compilation import Compilation
from .._declarations import Declarations, check_symbol_names
from .._pipeline import BasePass
from .._tree import Node


class LayoutPass(BasePass):
    """Pass that declares every component in tree order and checks that no two variables share a symbol."""

    reads = ('nodes',)
    writes = ('layout',)

    def _run(self, compilation: Compilation) -> None:
        """Declare the components of the nodes.

        Args:
            compilation:
                The compilation state.
        """
        declarations = Declarations(time=symbol('t'))
        for node in cast('list[Node]', compilation.nodes):
            declarations.declare_component(node)
        check_symbol_names(declarations.blocks)
        compilation.layout = declarations
