"""Define the orders spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..types import VariableKind
from ._base_spec import BaseSpec

if TYPE_CHECKING:
    from ..engine.compile._block import Block


@dataclass(frozen=True)
class _OrdersSpec(BaseSpec):
    """Orders and lags of the states of a system by entry name."""

    orders: dict[str, int]
    lags: dict[str, int]

    @classmethod
    def of(cls, blocks: list[Block]) -> _OrdersSpec:
        """Return the orders above 1 and the lags of every state of the blocks."""
        from ..engine._utils import declarations_of
        orders, lags = {}, {}
        for block in blocks:
            if block.node._is_interaction() or block.node.is_agent:
                continue
            declarations = declarations_of(block.node.component_type, VariableKind.STATE)
            for variable, entry in block.entries.items():
                declaration = declarations.get(variable)
                if declaration is not None and declaration.derivatives:
                    orders[entry.name] = declaration.derivatives + 1
                if declaration is not None and declaration.lags:
                    lags[entry.name] = declaration.lags
        return cls(orders, lags)
