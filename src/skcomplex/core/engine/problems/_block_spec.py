"""Define the block spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...base import BaseSpec
from ...components import EntrySpec
from ._node_spec import _NodeSpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class BlockSpec(BaseSpec):
    """A spec of the values of one component and the symbols that stand for them.

    While it compiles, the engine gives every value of an element, group, or interaction a symbol for one member, such
    as `x_population` for the size of a population, and the entry where the value sits in the vectors of the run. Laws
    are written in these symbols and read back through this spec. It holds the symbols of a population, a reactor, and a
    market.

    Read more in the user guide, under Extend the library.

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

    See Also:
        DeclarationsSpec: Holds the block of every component.
        ReferenceSpec: Says which value a symbol stands for.

    Examples:
        >>> from dataclasses import fields
        >>> from skcomplex.core.engine.problems._block_spec import BlockSpec
        >>> [field.name for field in fields(BlockSpec)]
        ['node', 'symbols', 'entries', 'references', 'per']
    """

    node: _NodeSpec
    symbols: Mapping[str, Expression | FreeModuleElement] = field(default_factory=dict)
    entries: Mapping[str, EntrySpec] = field(default_factory=dict)
    references: Mapping[Expression, ReferenceSpec] = field(default_factory=dict)
    per: frozenset[str] = frozenset()

    @property
    def rows(self) -> int:
        """Number of members, 1 for a single element or an interaction."""
        return self.node.members or 1
