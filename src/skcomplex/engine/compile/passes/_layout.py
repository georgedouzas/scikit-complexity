"""Declare the states and parameters of every component of a system in the layout of its problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.errors import LabelError
from ...problems import symbol
from .._base import BasePass
from .._block import Block
from .._declarations import Declarations
from .._tree import Node


class LayoutPass(BasePass):
    """Pass that declares every component in tree order and checks that no two variables share a symbol."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('nodes',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('layout',)

    def _run(self, inputs: object = None) -> None:
        """Declare the components of the nodes."""
        compilation = self.compilation_
        declarations = Declarations(time=symbol('t'))
        for node in cast('list[Node]', compilation.nodes):
            declarations.declare_component(node)
        self._check_symbol_names(declarations.blocks)
        compilation.layout = declarations

    @staticmethod
    def _check_symbol_names(blocks: dict[str, 'Block']) -> None:
        """Raise an error when two variables produce the same symbol name."""
        seen: dict[str, str] = {}
        for block in blocks.values():
            for item, reference in block.references.items():
                name = str(item)
                if name in seen and seen[name] != reference.entry.name:
                    error_msg = (
                        f'{reference.entry.name} and {seen[name]} produce the same symbol {name}. Rename one '
                        'of their labels or variables.'
                    )
                    raise LabelError(error_msg)
                seen[name] = reference.entry.name
