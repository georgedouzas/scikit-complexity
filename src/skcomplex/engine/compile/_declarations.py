"""Declare the states and parameters of every component of a system with the symbols its laws read them by."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import re
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import CC, RR, ZZ, Expression, I, oo
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ...core.elements import GroupElement
from ...core.errors import SpaceError, ValidationError
from ...core.interactions import BaseInteraction
from ...core.spaces import BaseSpace, NetworkSpace
from ...core.specs import WienerProcessSpec
from ...core.types import Vector
from ..problems import (
    Entry,
    Reference,
    Source,
    Variable,
    VariableKind,
    components_of,
    contains,
    declarations_of,
    domain_of,
    encode,
    symbol,
    vector,
)
from ._tree import _MAX_SHOWN_COMPONENTS, Node

_ACTIVE = '_active'
ENDS = ('_source', '_target')
_MEMBERS = '~members'


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

    def namespace(self, source: Source, alias: str = '') -> tuple[SimpleNamespace, dict[Expression, Reference]]:
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
                Source.SINGLE,
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
    processes: dict[str, Block] = field(default_factory=dict)
    handles: dict[Expression, tuple[str, int]] = field(default_factory=dict)
    agents: dict[Expression, tuple[str, str]] = field(default_factory=dict)

    def _declare_processes(self, node: Node) -> None:
        """Add a block with the handle `dW` of every Wiener process an interaction binds, one symbol per process."""
        for path, wiener in self._noise_processes([node]).items():
            if path in self.processes:
                continue
            block = Block(Node(path, node.scope, wiener, node.space))
            base = f'dW_{re.sub(r"[^0-9A-Za-z]", "_", path)}'
            handle = (
                symbol(base)
                if wiener.dimension == 1
                else vector(symbol(f'{base}_{index}') for index in range(wiener.dimension))
            )
            block.symbols['dW'] = handle
            self.processes[path] = block
            for index, item in enumerate(components_of(handle)):
                self.handles[item] = (path, index)

    def _declare(self, block: Block, variable: str, declaration: Variable, is_state: bool) -> None:
        """Add one variable of a component to its block."""
        node = block.node
        entries, values = (self.states, self.initial_states) if is_state else (self.parameters, self.initial_parameters)
        name = f'{node.path}.{variable}'
        if declaration.per is None:
            member_shape = self._shape_of(name, declaration, node.space)
        else:
            member_shape = self._per_entry_shape(node, variable, declaration)
            block.per.add(variable)
        members = None if node.is_agent else node.members
        shape = member_shape if members is None else (members, *member_shape)
        values.append(
            self._coerce_value(
                name,
                node.value_of(variable),
                member_shape,
                declaration.domain,
                members,
                period=declaration.period,
            ),
        )
        is_complex = declaration.domain is CC
        entry = Entry(name, self.ends[is_state], shape, declaration.domain, members, is_complex, declaration.period)
        entries.append(entry)
        self.ends[is_state] += entry.size
        base = f'{variable}_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}'
        names = [f'{base}_{index}' for index in range(member_shape[0])] if member_shape else [base]
        columns = [f'{item}_{part}' for item in names for part in ('re', 'im')] if is_complex else names
        domain = RR if is_complex else declaration.domain
        components = [symbol(item, domain) for item in columns]
        values_of = (
            [real + I * imaginary for real, imaginary in itertools.batched(components, 2)] if is_complex else components
        )
        block.symbols[variable] = vector(values_of) if member_shape else values_of[0]
        block.entries[variable] = entry
        for column, item in enumerate(components):
            block.references[item] = Reference(entry, is_state, column)

    def _declare_agents(self, block: Block) -> None:
        """Add the symbols that reductions of an agent group read: its members, scalar states, and scalar parameters.

        The states of agents live in the agent stepper, so their symbols have no entries. The symbol of the members
        comes first in the namespace, so a count names the group by it.
        """
        node = block.node
        base = re.sub(r'[^0-9A-Za-z]', '_', node.path)
        members = symbol(f'members_{base}')
        block.symbols = {_MEMBERS: members, **block.symbols}
        self.agents[members] = (node.path, '')
        for variable, declaration in declarations_of(node.component_type, VariableKind.STATE).items():
            if declaration.vector or declaration.domain is CC:
                continue
            item = symbol(f'{variable}_{base}', declaration.domain)
            block.symbols[variable] = item
            self.agents[item] = (node.path, variable)
        for variable in block.entries:
            value = block.symbols[variable]
            if isinstance(value, Expression) and value.is_symbol():
                self.agents[value] = (node.path, variable)

    def _declare_hidden(self, block: Block, variable: str, values: Vector) -> None:
        """Add a hidden parameter with one value per member of a group, read by laws as `s.<variable>`."""
        node = block.node
        rows = node.members or 1
        name = f'{node.path}.{variable}'
        entry = Entry(name, self.ends[False], (rows,), RR, rows)
        self.parameters.append(entry)
        self.initial_parameters.append(np.asarray(values, dtype=float))
        self.ends[False] += entry.size
        item = symbol(f'{variable.strip("_")}_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}_topology', RR)
        block.symbols[variable] = item
        block.entries[variable] = entry
        block.references[item] = Reference(entry, False, 0)

    def _declare_topology(self, block: Block) -> None:
        """Add the activity of every row of a capacity group, and the ends of every edge of a group on edges."""
        node = block.node
        group = cast('GroupElement', node.component)
        rows, active = node.members or 0, node.active or 0
        self._declare_hidden(block, _ACTIVE, (np.arange(rows) < active).astype(float))
        if group.on != 'edges':
            return
        space = cast('NetworkSpace', node.space)
        padding = np.zeros(rows - active, dtype=np.intp)
        for variable, ends in zip(ENDS, space.edges_[group.layer], strict=True):
            self._declare_hidden(block, variable, np.concatenate([ends, padding]))

    def declare_state(
        self,
        name: str,
        initial: Vector,
        domain: object,
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

    @property
    def wieners(self) -> dict[str, WienerProcessSpec]:
        """Wiener processes that the noise roles of interactions bind, by path."""
        return {path: cast('WienerProcessSpec', block.node.component) for path, block in self.processes.items()}

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
            layer: (block.entries[ENDS[0]], block.entries[ENDS[1]], block.entries[_ACTIVE])
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
        if node.is_agent:
            self._declare_agents(block)
        if isinstance(node.component, BaseInteraction):
            self._declare_processes(node)
        if node.active is not None:
            self._declare_topology(block)
        self.blocks[node.path] = block
        return block

    @staticmethod
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

    @staticmethod
    def _shape_of(name: str, declaration: Variable, space: BaseSpace | None) -> tuple[int, ...]:
        """Return the shape of a declared variable in its space.

        Args:
            name:
                The dotted name of the variable.

            declaration:
                The declaration of the variable.

            space:
                The nearest enclosing space.

        Returns:
            shape:
                `()` for a scalar and `(n_dim,)` for a vector.

        Raises:
            SpaceError:
                A vector variable has no enclosing space.

            ValidationError:
                The space dimension is not a positive integer.
        """
        if not declaration.vector:
            return ()
        if space is None:
            error_msg = f'{name} is a vector, so the system that holds it or one of its parents must have a space.'
            raise SpaceError(error_msg)
        if not contains(RealSet.open(0, oo), space.n_dim_) or not contains(ZZ, space.n_dim_):
            error_msg = f'The space of {name} has n_dim = {space.n_dim_!r}, which is not a positive integer.'
            raise ValidationError(error_msg)
        return (int(space.n_dim_),)

    @staticmethod
    def _member_count(blocks: list['Block']) -> int:
        """Return the number of members of the groups of a system, leaving out agents, which chooses a default backend.

        Args:
            blocks:
                The blocks of the components.

        Returns:
            members:
                The total number of members.
        """
        return sum(block.node.members or 0 for block in blocks if not block.node.is_agent)

    @staticmethod
    def _describe(value: object) -> str:
        """Return the repr of a small value, or its shape for an array with many components."""
        if isinstance(value, np.ndarray | list | tuple) and np.size(value) > _MAX_SHOWN_COMPONENTS:
            return f'an array of shape {np.shape(value)}'
        return repr(value)

    @staticmethod
    def _flat_value(
        name: str,
        value: object,
        array: npt.NDArray[Any],
        shape: tuple[int, ...],
        members: int | None,
    ) -> npt.NDArray[Any]:
        """Return an encoded value as flat components of the given shape, member after member for a group."""
        if members is not None:
            return Declarations._coerce_group_value(name, value, array, shape, members)
        if not shape and array.ndim == 0:
            return array.reshape(1)
        if shape and array.ndim == 0:
            return np.full(shape, array.item(), dtype=array.dtype).ravel()
        if array.shape != shape:
            error_msg = (
                f'{name} = {Declarations._describe(value)} has shape {array.shape}, but the variable has shape {shape}.'
            )
            raise ValidationError(error_msg)
        return array.ravel()

    @staticmethod
    def _coerce_group_value(
        name: str,
        value: object,
        array: npt.NDArray[Any],
        shape: tuple[int, ...],
        members: int,
    ) -> npt.NDArray[Any]:
        """Return a group value as flat member-major components, broadcasting shared values."""
        if array.shape in ((), shape):
            return np.broadcast_to(array, (members, *shape)).ravel().copy()
        if array.shape == (members, *shape):
            return array.ravel()
        count = array.shape[0] if array.ndim else 1
        error_msg = (
            f'{name} = {Declarations._describe(value)} has {count} values of shape {array.shape[1:]} for a group of '
            f'{members} members of shape {shape}. Give one value per member or one value shared by every member.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _coerce_value(
        name: str,
        value: object,
        shape: tuple[int, ...],
        domain: object,
        members: int | None = None,
        *,
        period: object = None,
    ) -> npt.NDArray[np.float64]:
        """Return a value as a flat float array of the given shape after validating it.

        Args:
            name:
                The dotted name of the variable.

            value:
                The value given by the user.

            shape:
                The shape of the variable.

            domain:
                The domain of the variable.

            members:
                The size of the group that holds the variable, or `None` for a single element or an interaction. A group
                value is a per-member array with this first dimension, or one value shared by every member.

            period:
                The period the values wrap around, or `None`.

        Returns:
            array:
                The flat components of the value, member after member for a group, with the real and imaginary part of
                every component in a row for the complex domain.

        Raises:
            ValidationError:
                The value is not numeric, has the wrong shape, or is outside the domain.
        """
        if not contains(domain, value):
            error_msg = f'{name} = {Declarations._describe(value)} is not in its domain {domain}.'
            if contains(CC, value):
                error_msg = (
                    f'{name} = {Declarations._describe(value)} has a nonzero imaginary part, which its domain {domain} '
                    f'does not hold. Declare {name.rpartition(".")[2]} with `StateSpec(domain=CC)` to give it complex '
                    'values.'
                )
            raise ValidationError(error_msg)
        flat = Declarations._flat_value(name, value, encode(domain, value, period), shape, members)
        return np.ascontiguousarray(flat).view(float) if domain is CC else flat

    @staticmethod
    def _noise_processes(nodes: list[Node]) -> dict[str, WienerProcessSpec]:
        """Return the Wiener processes that the noise roles of the interactions of a system tree bind, by path.

        Interactions of one system that bind processes of one label share them, at the path of the label in that system.

        Args:
            nodes:
                The nodes of the system tree.

        Returns:
            processes:
                The Wiener processes by path, in the order the interactions first bind them.
        """
        processes: dict[str, WienerProcessSpec] = {}
        for node in nodes:
            if not isinstance(node.component, BaseInteraction):
                continue
            for name, declaration in declarations_of(type(node.component), VariableKind.ROLE).items():
                value = getattr(node.component, name)
                if declaration.noise and isinstance(value, WienerProcessSpec):
                    processes.setdefault(f'{node.scope}{value.label}', value)
        return processes
