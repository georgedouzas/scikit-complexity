"""Variables of a component with the symbols of one member and the components they stand for."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from .._utils import domain_of, symbol, vector
from ..problems import Entry
from ..problems._read_kind import ReadKind
from ..problems._reference import Reference
from ._tree import Node


@dataclass
class Block:
    """Variables of a component with the symbols of one member and the components they stand for."""

    node: Node
    symbols: dict[str, Expression | FreeModuleElement] = field(default_factory=dict)
    entries: dict[str, Entry] = field(default_factory=dict)
    references: dict[Expression, Reference] = field(default_factory=dict)
    per: set[str] = field(default_factory=set)

    @property
    def rows(self) -> int:
        """Number of members, 1 for a single element or an interaction."""
        return self.node.members or 1

    def _namespace(self, mapping: dict[Expression, Expression]) -> SimpleNamespace:
        """Return the namespace of the block's variables in renamed symbols, with tuples for per-entry parameters."""
        symbols: dict[str, Any] = {}
        for name, value in self.symbols.items():
            if not isinstance(value, FreeModuleElement):
                symbols[name] = self._renamed(value, mapping)
            elif name in self.per:
                symbols[name] = tuple(self._renamed(item, mapping) for item in value)
            else:
                symbols[name] = vector(self._renamed(item, mapping) for item in value)
        return SimpleNamespace(**symbols)

    def namespace(self, source: ReadKind, alias: str = '') -> tuple[SimpleNamespace, dict[Expression, Reference]]:
        """Return the namespace of the block's symbols and their references for one way of reading them.

        Args:
            source:
                How the block's components are read.

            alias:
                A suffix that gives the block's symbols distinct names, used for the second member of a pair.

        Returns:
            namespace:
                The namespace and the references of its symbols.
        """
        mapping = {item: symbol(f'{item}{alias}', domain_of(item)) for item in self.references}
        references = {
            mapping[item]: Reference(reference.entry, reference.is_state, reference.column, source)
            for item, reference in self.references.items()
        }
        return self._namespace(mapping), references

    def member_namespace(self, index: int) -> tuple[SimpleNamespace, dict[Expression, Reference]]:
        """Return the namespace of one member of a group, read like a single element, and the references of its symbols.

        Args:
            index:
                The index of the member.

        Returns:
            namespace:
                The namespace and the references of its symbols, whose columns count from the group's first component.
        """
        mapping = {item: symbol(f'{item}__m{index}', domain_of(item)) for item in self.references}
        references = {
            mapping[item]: Reference(
                reference.entry,
                reference.is_state,
                index * reference.entry.columns + reference.column,
                ReadKind.SINGLE,
            )
            for item, reference in self.references.items()
        }
        return self._namespace(mapping), references

    @staticmethod
    def _renamed(value: Expression, mapping: dict[Expression, Expression]) -> Expression:
        """Return the renamed symbol of a variable, or a complex value with its real and imaginary symbols renamed.

        A symbol without a variable, such as the handle of a Wiener process, keeps its name.
        """
        return mapping.get(value, value) if value.operator() is None else value.subs(mapping)
