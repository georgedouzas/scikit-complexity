"""Declare the states and parameters of every component of a system with the symbols its laws read them by."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

import numpy as np
from sage.all__sagemath_symbolics import I

from ...core.base import (
    Entry,
    Expression,
    LabelError,
    NumericDomain,
    SymbolicVector,
    ValidationError,
    Variable,
    VariableKind,
    Vector,
    components_of,
    declarations_of,
    domain_of,
    symbol,
    vector,
)
from ...core.components import ElementComponent, GroupComponent, Wiener
from ...core.laws import Symbols
from ...core.space import NetworkSpace
from ._kernels import Reference, Source
from ._tree import Node, coerce_value, shape_of

ACTIVE = '_active'
ENDS = ('_source', '_target')


def _renamed(value: Expression, mapping: dict[Expression, Expression]) -> Expression:
    """Return the renamed symbol of a variable, or a complex value with its real and imaginary symbols renamed.

    A symbol without a variable, such as the handle of a Wiener process, keeps its name.
    """
    return mapping.get(value, value) if value.operator() is None else value.subs(mapping)


def _per_entry_shape(node: Node, variable: str, declaration: Variable) -> tuple[int, ...]:
    """Return the shape of a parameter with one value per entry of a variadic role, after checking its length."""
    roles = declarations_of(node.component_type, VariableKind.ROLE)
    role = declaration.per
    if role not in roles or not roles[role].variadic:
        error_msg = (
            f'{node.path}.{variable} is declared per entry of {role!r}, which is not a variadic role of '
            f'{node.component_type.__name__}. Declare the role with `role(variadic=True)`.'
        )
        raise ValidationError(error_msg)
    entries = len(getattr(node.component, role))
    value = node.value_of(variable)
    given = len(value) if isinstance(value, list | tuple | np.ndarray) and np.ndim(value) else None
    if given is not None and given != entries:
        error_msg = (
            f'Interaction {node.path} gives its parameter {variable!r} {given} values, but its role {role!r} binds '
            f'{entries} entries. Give one value per entry.'
        )
        raise ValidationError(error_msg)
    return (entries,)


def check_symbol_names(blocks: dict[str, 'Block']) -> None:
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


def is_interaction(node: Node) -> bool:
    """Check whether a node is an interaction."""
    return not isinstance(node.component, ElementComponent | GroupComponent)


def member_count(blocks: list['Block']) -> int:
    """Return the number of members of the groups of a system, leaving out agents, which `'auto'` picks a backend by.

    Args:
        blocks:
            The blocks of the components.

    Returns:
        members:
            The total number of members.
    """
    return sum(block.node.members or 0 for block in blocks if not block.node.is_agent)


@dataclass
class Block:
    """Variables of a component with the symbols of one member and the components they stand for."""

    node: Node
    symbols: dict[str, Expression | SymbolicVector] = field(default_factory=dict)
    entries: dict[str, Entry] = field(default_factory=dict)
    references: dict[Expression, Reference] = field(default_factory=dict)
    per: set[str] = field(default_factory=set)

    @property
    def rows(self) -> int:
        """Number of members, 1 for a single element or an interaction."""
        return self.node.members or 1

    def _namespace(self, mapping: dict[Expression, Expression]) -> Symbols:
        """Return the namespace of the block's variables in renamed symbols, with tuples for per-entry parameters."""
        symbols: dict[str, Any] = {}
        for name, value in self.symbols.items():
            if not isinstance(value, SymbolicVector):
                symbols[name] = _renamed(value, mapping)
            elif name in self.per:
                symbols[name] = tuple(_renamed(item, mapping) for item in value)
            else:
                symbols[name] = vector(_renamed(item, mapping) for item in value)
        return Symbols(**symbols)

    def namespace(self, source: Source, alias: str = '') -> tuple[Symbols, dict[Expression, Reference]]:
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

    def member_namespace(self, index: int) -> tuple[Symbols, dict[Expression, Reference]]:
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
                Source.SINGLE,
            )
            for item, reference in self.references.items()
        }
        return self._namespace(mapping), references


@dataclass
class Declarations:
    """Entries, initial values, and blocks collected from the components."""

    time: Expression
    states: list[Entry] = field(default_factory=list)
    parameters: list[Entry] = field(default_factory=list)
    initial_states: list[Vector] = field(default_factory=list)
    initial_parameters: list[Vector] = field(default_factory=list)
    blocks: dict[str, Block] = field(default_factory=dict)
    ends: dict[bool, int] = field(default_factory=lambda: {True: 0, False: 0})
    wieners: dict[str, Wiener] = field(default_factory=dict)
    handles: dict[Expression, tuple[str, int]] = field(default_factory=dict)

    def _declare_wiener(self, block: Block, wiener: Wiener) -> None:
        """Add the handle `dW` of a Wiener element to its block and index its symbols by process."""
        path = block.node.path
        handle = wiener.declare_handle(path)
        block.symbols['dW'] = handle
        self.wieners[path] = wiener
        for index, item in enumerate(components_of(handle)):
            self.handles[item] = (path, index)

    def _declare(self, block: Block, variable: str, declaration: Variable, is_state: bool) -> None:
        """Add one variable of a component to its block."""
        node = block.node
        entries, values = (self.states, self.initial_states) if is_state else (self.parameters, self.initial_parameters)
        name = f'{node.path}.{variable}'
        if declaration.per is None:
            member_shape = shape_of(name, declaration, node.space)
        else:
            member_shape = _per_entry_shape(node, variable, declaration)
            block.per.add(variable)
        members = None if node.is_agent else node.members
        shape = member_shape if members is None else (members, *member_shape)
        values.append(coerce_value(name, node.value_of(variable), member_shape, declaration.domain, members))
        is_complex = declaration.domain is NumericDomain.COMPLEX
        entry = Entry(name, self.ends[is_state], shape, declaration.domain, members, is_complex)
        entries.append(entry)
        self.ends[is_state] += entry.size
        base = f'{variable}_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}'
        names = [f'{base}_{index}' for index in range(member_shape[0])] if member_shape else [base]
        columns = [f'{item}_{part}' for item in names for part in ('re', 'im')] if is_complex else names
        domain = NumericDomain.REAL if is_complex else declaration.domain
        components = [symbol(item, domain) for item in columns]
        values_of = (
            [real + I * imaginary for real, imaginary in itertools.batched(components, 2)] if is_complex else components
        )
        block.symbols[variable] = vector(values_of) if member_shape else values_of[0]
        block.entries[variable] = entry
        for column, item in enumerate(components):
            block.references[item] = Reference(entry, is_state, column)

    def _declare_hidden(self, block: Block, variable: str, values: Vector) -> None:
        """Add a hidden parameter with one value per member of a group, read by laws as `s.<variable>`."""
        node = block.node
        rows = node.members or 1
        name = f'{node.path}.{variable}'
        entry = Entry(name, self.ends[False], (rows,), NumericDomain.REAL, rows)
        self.parameters.append(entry)
        self.initial_parameters.append(np.asarray(values, dtype=float))
        self.ends[False] += entry.size
        item = symbol(f'{variable.strip("_")}_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}_topology', NumericDomain.REAL)
        block.symbols[variable] = item
        block.entries[variable] = entry
        block.references[item] = Reference(entry, False, 0)

    def _declare_topology(self, block: Block) -> None:
        """Add the activity of every row of a capacity group, and the ends of every edge of a group on edges."""
        node = block.node
        group = cast('GroupComponent', node.component)
        rows, active = node.members or 0, node.active or 0
        self._declare_hidden(block, ACTIVE, (np.arange(rows) < active).astype(float))
        if group.on != 'edges':
            return
        space = cast('NetworkSpace', node.space)
        padding = np.zeros(rows - active, dtype=np.intp)
        for variable, ends in zip(ENDS, space.list_edges(group.layer), strict=True):
            self._declare_hidden(block, variable, np.concatenate([ends, padding]))

    def declare_state(
        self,
        name: str,
        initial: Vector,
        domain: NumericDomain,
        shape: tuple[int, ...] = (),
        members: int | None = None,
    ) -> Entry:
        """Add a hidden state after every state declared so far and return its entry.

        Args:
            name:
                The path of the state.

            initial:
                The flat initial value.

            domain:
                The domain of the state.

            shape:
                The shape of the state.

            members:
                The number of members of a group state, or `None`.

        Returns:
            entry:
                The entry of the state.
        """
        entry = Entry(name, self.ends[True], shape, domain, members)
        self.states.append(entry)
        self.ends[True] += entry.size
        self.initial_states.append(initial)
        return entry

    @cached_property
    def capacity(self) -> dict[str, Block]:
        """Blocks of the variable groups on the sites or edges of networks, by path."""
        return {path: block for path, block in self.blocks.items() if block.node.active is not None}

    def changing_edges(self, space: object) -> dict[str | None, Block]:
        """Return the variable edge group of every layer of a network whose edges change.

        Args:
            space:
                The network space.

        Returns:
            groups:
                The block of the variable edge group by layer.
        """
        return {
            getattr(block.node.component, 'layer', None): block
            for block in self.capacity.values()
            if getattr(block.node.component, 'on', None) == 'edges' and block.node.space is space
        }

    def changing_entries(self, space: object) -> dict[str | None, tuple[Entry, Entry, Entry]]:
        """Return the source, target, and activity entries of the variable edge group of every changing layer.

        Args:
            space:
                The network space.

        Returns:
            entries:
                The entries by layer.
        """
        return {
            layer: (block.entries[ENDS[0]], block.entries[ENDS[1]], block.entries[ACTIVE])
            for layer, block in self.changing_edges(space).items()
        }

    @cached_property
    def groups(self) -> frozenset[str]:
        """Paths of the declared groups, indexed once every component is declared."""
        return frozenset(path for path, block in self.blocks.items() if block.node.members is not None)

    def declare_component(self, node: Node) -> Block:
        """Add the states and parameters of a component and return its block.

        Args:
            node:
                The component with its place in the tree.

        Returns:
            block:
                The block of the component.
        """
        block = Block(node)
        states = {} if node.is_agent else declarations_of(node.component_type, VariableKind.STATE)
        for variable, declaration in states.items():
            self._declare(block, variable, declaration, is_state=True)
        for variable, declaration in declarations_of(node.component_type, VariableKind.PARAMETER).items():
            self._declare(block, variable, declaration, is_state=False)
        if isinstance(node.component, Wiener):
            self._declare_wiener(block, node.component)
        if node.active is not None:
            self._declare_topology(block)
        self.blocks[node.path] = block
        return block
