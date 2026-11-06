"""Define the block spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...components.specs import NodeSpec
from ...specs._base_spec import BaseSpec
from ...specs._entry_spec import EntrySpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class BlockSpec(BaseSpec):
    """Variables of a component with the symbols of one member and the components they stand for.

    Args:
        node:
            The component with its place in the system tree.

        symbols:
            The symbol, or the vector of symbols, of every variable of one member by name.

        entries:
            The layout entry of every variable by name.

        references:
            The component of a state or parameter every symbol stands for.

        per:
            The names of the parameters with one value per entry of a variadic role.
    """

    node: NodeSpec
    symbols: Mapping[str, Expression | FreeModuleElement] = field(default_factory=dict)
    entries: Mapping[str, EntrySpec] = field(default_factory=dict)
    references: Mapping[Expression, ReferenceSpec] = field(default_factory=dict)
    per: frozenset[str] = frozenset()

    @property
    def rows(self) -> int:
        """Number of members, 1 for a single element or an interaction."""
        return self.node.members or 1
