"""Pass that expands the flows of components into laws of other kinds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Any, cast

from ...problems import VariableKind, declarations_of
from .._base import BasePass
from .._compilation import BlockLaws
from .._declarations import Declarations
from .._flows import TransferPass


class ExpandFlowsPass(BasePass):
    """Pass that expands the flows of components into laws of other kinds."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('collected', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected', 'flows')

    def _run(self, inputs: object = None) -> None:
        """Expand the flows of the collected laws."""
        compilation = self.compilation_
        declarations = cast('Declarations', compilation.layout)
        flows = TransferPass(self._quantities_of(declarations.blocks))
        compilation.collected = [
            (spec, flows.expand(spec, list(laws)), block)
            for spec, laws, block in cast('BlockLaws', compilation.collected)
        ]
        compilation.flows = flows

    @staticmethod
    def _quantities_of(blocks: Mapping[str, Any]) -> dict[str, str]:
        """Return the quantity of every state tagged with one, by entry name.

        Args:
            blocks:
                The blocks of the declared components by path.

        Returns:
            quantities:
                The quantity of every tagged state.
        """
        quantities = {}
        for block in blocks.values():
            if block.node.is_agent:
                continue
            for variable, declaration in declarations_of(block.node.component_type, VariableKind.STATE).items():
                if declaration.quantity is not None:
                    quantities[block.entries[variable].name] = declaration.quantity
        return quantities
