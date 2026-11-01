"""Declare the states and parameters of every component of a system in the layout of its problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import re
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import CC, RR, ZZ, Expression, I, oo
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.elements import GroupElement
from ....core.interactions import BaseInteraction
from ....core.spaces import BaseSpace, LatticeSpace, NetworkSpace
from ....core.specs import NodeSpec
from ....errors import LabelError, SpaceError, ValidationError
from ....specs import EntrySpec, VariableSpec, WienerProcessSpec
from ....types import Array, VariableKind
from ..._utils import components_of, declarations_of, default_of, symbol, vector
from ...problems import BlockSpec, DeclarationsSpec, ReferenceSpec
from ...problems._declarations_spec import _ACTIVE, ENDS, MEMBERS
from .._base import BasePass
from .._compilation import Compilation

_MEMBERS = '~members'
_MAX_SHOWN_COMPONENTS = 10


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
        declarations = DeclarationsSpec(time=symbol('t'))
        for node in Compilation._required(compilation.nodes, 'nodes'):
            declarations, _ = self._declare_component(declarations, node)
        self._check_symbol_names(declarations.blocks)
        compilation.layout = declarations

    @staticmethod
    def _check_symbol_names(blocks: Mapping[str, BlockSpec]) -> None:
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

    @staticmethod
    def _declare_component(declarations: DeclarationsSpec, node: NodeSpec) -> tuple[DeclarationsSpec, BlockSpec]:
        """Return the declarations with the states and parameters of a component added, and the block of the component.

        Args:
            declarations:
                The declarations so far.

            node:
                The component with its place in the tree.

        Returns:
            declarations:
                The declarations with the component.

            block:
                The block of the component.
        """
        block = BlockSpec(node)
        states = {} if node.is_agent else declarations_of(node.component_type, VariableKind.STATE)
        for variable, declaration in states.items():
            declarations, block = LayoutPass._declare(declarations, block, variable, declaration, is_state=True)
        for variable, declaration in declarations_of(node.component_type, VariableKind.PARAMETER).items():
            declarations, block = LayoutPass._declare(declarations, block, variable, declaration, is_state=False)
        if node.is_agent:
            declarations, block = LayoutPass._declare_agents(declarations, block)
        if isinstance(node.component, BaseInteraction):
            declarations = LayoutPass._declare_processes(declarations, node)
        if node.active is not None:
            declarations, block = LayoutPass._declare_topology(declarations, block)
        return replace(declarations, blocks={**declarations.blocks, node.path: block}), block

    @staticmethod
    def _declare_processes(declarations: DeclarationsSpec, node: NodeSpec) -> DeclarationsSpec:
        """Return the declarations with a block holding the handle `dW` of every Wiener process an interaction binds."""
        processes, handles = dict(declarations.processes), dict(declarations.handles)
        for path, wiener in LayoutPass._noise_processes([node]).items():
            if path in processes:
                continue
            base = f'dW_{re.sub(r"[^0-9A-Za-z]", "_", path)}'
            handle = (
                symbol(base)
                if wiener.dimension == 1
                else vector(symbol(f'{base}_{index}') for index in range(wiener.dimension))
            )
            process = NodeSpec(path, node.scope, wiener, node.space, WienerProcessSpec, interaction=True)
            processes[path] = BlockSpec(process, {'dW': handle})
            for index, item in enumerate(components_of(handle)):
                handles[item] = (path, index)
        return replace(declarations, processes=processes, handles=handles)

    @staticmethod
    def _with_entry(
        declarations: DeclarationsSpec,
        entry: EntrySpec,
        value: Array,
        is_state: bool,
    ) -> DeclarationsSpec:
        """Return the declarations with an entry and its initial value added to the states or the parameters."""
        if is_state:
            return replace(
                declarations,
                states=(*declarations.states, entry),
                initial_states=(*declarations.initial_states, value),
            )
        return replace(
            declarations,
            parameters=(*declarations.parameters, entry),
            initial_parameters=(*declarations.initial_parameters, value),
        )

    @staticmethod
    def _declare(
        declarations: DeclarationsSpec,
        block: BlockSpec,
        variable: str,
        declaration: VariableSpec,
        is_state: bool,
    ) -> tuple[DeclarationsSpec, BlockSpec]:
        """Return the declarations and the block with one variable of a component added."""
        node = block.node
        name = f'{node.path}.{variable}'
        per = block.per
        if declaration.per is None:
            member_shape = LayoutPass._shape_of(name, declaration, node.space)
        else:
            member_shape = LayoutPass._per_entry_shape(node, variable, declaration)
            per = per | {variable}
        members = None if node.is_agent else node.members
        shape = member_shape if members is None else (members, *member_shape)
        value = LayoutPass._coerce_value(
            name,
            LayoutPass._value_of(node, variable),
            member_shape,
            declaration.domain,
            members,
            period=declaration.period,
        )
        is_complex = declaration.domain is CC
        start = declarations.ends[is_state]
        entry = EntrySpec(name, start, shape, declaration.domain, members, is_complex, declaration.period)
        base = f'{variable}_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}'
        names = [f'{base}_{index}' for index in range(member_shape[0])] if member_shape else [base]
        columns = [f'{item}_{part}' for item in names for part in ('re', 'im')] if is_complex else names
        domain = RR if is_complex else declaration.domain
        components = [symbol(item, domain) for item in columns]
        values_of = (
            [real + I * imaginary for real, imaginary in itertools.batched(components, 2)] if is_complex else components
        )
        block = replace(
            block,
            symbols={**block.symbols, variable: vector(values_of) if member_shape else values_of[0]},
            entries={**block.entries, variable: entry},
            references={
                **block.references,
                **{item: ReferenceSpec(entry, is_state, column) for column, item in enumerate(components)},
            },
            per=per,
        )
        return LayoutPass._with_entry(declarations, entry, value, is_state), block

    @staticmethod
    def _declare_agents(declarations: DeclarationsSpec, block: BlockSpec) -> tuple[DeclarationsSpec, BlockSpec]:
        """Return the declarations and the block with the symbols that reductions of an agent group read.

        The symbols are the members, the scalar states, and the scalar parameters of the group. The states of agents
        live in the agent stepper, so their symbols have no entries. The symbol of the members comes first in the
        namespace, so a count names the group by it.
        """
        node = block.node
        base = re.sub(r'[^0-9A-Za-z]', '_', node.path)
        members = symbol(f'members_{base}')
        symbols: dict[str, Expression | FreeModuleElement] = {_MEMBERS: members, **block.symbols}
        agents = {**declarations.agents, members: (node.path, '')}
        for variable, declaration in declarations_of(node.component_type, VariableKind.STATE).items():
            if declaration.vector or declaration.domain is CC:
                continue
            item = symbol(f'{variable}_{base}', declaration.domain)
            symbols[variable] = item
            agents[item] = (node.path, variable)
        for variable in block.entries:
            value = symbols[variable]
            if isinstance(value, Expression) and value.is_symbol():
                agents[value] = (node.path, variable)
        return replace(declarations, agents=agents), replace(block, symbols=symbols)

    @staticmethod
    def _declare_hidden(
        declarations: DeclarationsSpec,
        block: BlockSpec,
        variable: str,
        values: Array,
    ) -> tuple[DeclarationsSpec, BlockSpec]:
        """Return the declarations and the block with a hidden parameter of one value or one row per member of a group.

        Laws read the parameter as `s.<variable>`, a vector for values with one row per member.
        """
        node = block.node
        rows = node.members or 1
        shape = (rows, *np.shape(values)[1:])
        entry = EntrySpec(f'{node.path}.{variable}', declarations.ends[False], shape, RR, rows)
        base = f'{variable.strip("_")}_{re.sub(r"[^0-9A-Za-z]", "_", node.path)}_topology'
        items = [symbol(f'{base}_{column}', RR) for column in range(shape[1])] if len(shape) > 1 else [symbol(base, RR)]
        block = replace(
            block,
            symbols={**block.symbols, variable: vector(items) if len(shape) > 1 else items[0]},
            entries={**block.entries, variable: entry},
            references={
                **block.references,
                **{item: ReferenceSpec(entry, False, column) for column, item in enumerate(items)},
            },
        )
        return (
            LayoutPass._with_entry(declarations, entry, np.ravel(np.asarray(values, dtype=float)), is_state=False),
            block,
        )

    @staticmethod
    def _declare_topology(declarations: DeclarationsSpec, block: BlockSpec) -> tuple[DeclarationsSpec, BlockSpec]:
        """Return the declarations and the block with the activity of every row of a capacity group.

        A group on edges also gets the ends of every edge, and a group on hyperedges the members of every hyperedge,
        whose hyperedges all have one size.
        """
        node = block.node
        group = node.component
        if not isinstance(group, GroupElement):
            error_msg = f'{node.path} has a capacity layout, but it is not a group.'
            raise ValidationError(error_msg)
        rows, active = node.members or 0, node.active or 0
        declarations, block = LayoutPass._declare_hidden(
            declarations,
            block,
            _ACTIVE,
            (np.arange(rows) < active).astype(float),
        )
        if group.on not in {'edges', 'hyperedges'}:
            return declarations, block
        space = node.space
        if not isinstance(space, NetworkSpace):
            error_msg = f'The group {node.path} lies on the {group.on} of a space that is not a network.'
            raise ValidationError(error_msg)
        if group.on == 'hyperedges':
            return LayoutPass._declare_hidden(declarations, block, MEMBERS, LayoutPass._members(node, space))
        padding = np.zeros(rows - active, dtype=np.intp)
        for variable, ends in zip(ENDS, space.edges_[group.layer], strict=True):
            declarations, block = LayoutPass._declare_hidden(
                declarations,
                block,
                variable,
                np.concatenate([ends, padding]),
            )
        return declarations, block

    @staticmethod
    def _members(node: NodeSpec, space: NetworkSpace) -> Array:
        """Return the member nodes of every row of a variable group on hyperedges, with zeros in the free rows.

        Raises:
            ValidationError:
                The hyperedges of the network have more than one size, or none.
        """
        sizes = sorted({len(hyperedge) for hyperedge in space.hyperedges_})
        if len(sizes) != 1:
            error_msg = (
                f'The variable group {node.path} lies on the hyperedges of a network whose hyperedges have the sizes '
                f'{sizes}. Give the network hyperedges of one size, which every new hyperedge also has.'
            )
            raise ValidationError(error_msg)
        members = np.zeros(((node.members or 0), sizes[0]))
        members[: len(space.hyperedges_)] = np.asarray(space.hyperedges_, dtype=float)
        return members

    @staticmethod
    def _value_of(node: NodeSpec, variable: str) -> object:
        """Return the value of a variable of a component, or the default of the element type for a group without it.

        Args:
            node:
                The component with its place in the tree.

            variable:
                The name of the variable.

        Returns:
            value:
                The value given by the user or the default.
        """
        component = node.component
        if not isinstance(component, GroupElement):
            return getattr(component, variable) if hasattr(component, variable) else default_of(component, variable)
        values = component.values_
        default = default_of(component.element, variable)
        if variable not in values:
            return default
        value = values[variable]
        shape = np.shape(np.asarray(value))
        space = getattr(component, 'space', None)
        if isinstance(space, LatticeSpace) and shape == space.shape_:
            return np.ravel(np.asarray(value))
        if node.active is not None and shape and shape[0] == node.active:
            rows = (node.members or 0) - node.active
            padding = np.broadcast_to(np.asarray(default, dtype=float), (rows, *shape[1:]))
            return np.concatenate([np.asarray(value, dtype=float), padding])
        return value

    @staticmethod
    def _per_entry_shape(node: NodeSpec, variable: str, declaration: VariableSpec) -> tuple[int, ...]:
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
        value = LayoutPass._value_of(node, variable)
        given = len(value) if isinstance(value, list | tuple | np.ndarray) and np.ndim(value) else None
        if given is not None and given != entries:
            error_msg = (
                f'Interaction {node.path} gives its parameter {variable!r} {given} values, but its role {role!r} binds '
                f'{entries} entries. Give one value per entry.'
            )
            raise ValidationError(error_msg)
        return (entries,)

    @staticmethod
    def _shape_of(name: str, declaration: VariableSpec, space: BaseSpace | None) -> tuple[int, ...]:
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
        if not EntrySpec._contains(RealSet.open(0, oo), space.n_dim_) or not EntrySpec._contains(ZZ, space.n_dim_):
            error_msg = f'The space of {name} has n_dim = {space.n_dim_!r}, which is not a positive integer.'
            raise ValidationError(error_msg)
        return (int(space.n_dim_),)

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
        array: npt.NDArray[np.generic],
        shape: tuple[int, ...],
        members: int | None,
    ) -> npt.NDArray[np.generic]:
        """Return an encoded value as flat components of the given shape, member after member for a group."""
        if members is not None:
            return LayoutPass._coerce_group_value(name, value, array, shape, members)
        if not shape and array.ndim == 0:
            return array.reshape(1)
        if shape and array.ndim == 0:
            return np.full(shape, array.item(), dtype=array.dtype).ravel()
        if array.shape != shape:
            error_msg = (
                f'{name} = {LayoutPass._describe(value)} has shape {array.shape}, but the variable has shape {shape}.'
            )
            raise ValidationError(error_msg)
        return array.ravel()

    @staticmethod
    def _coerce_group_value(
        name: str,
        value: object,
        array: npt.NDArray[np.generic],
        shape: tuple[int, ...],
        members: int,
    ) -> npt.NDArray[np.generic]:
        """Return a group value as flat member-major components, broadcasting shared values."""
        if array.shape in ((), shape):
            return np.broadcast_to(array, (members, *shape)).ravel().copy()
        if array.shape == (members, *shape):
            return array.ravel()
        count = array.shape[0] if array.ndim else 1
        error_msg = (
            f'{name} = {LayoutPass._describe(value)} has {count} values of shape {array.shape[1:]} for a group of '
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
        if not EntrySpec._contains(domain, value):
            error_msg = f'{name} = {LayoutPass._describe(value)} is not in its domain {domain}.'
            if EntrySpec._contains(CC, value):
                error_msg = (
                    f'{name} = {LayoutPass._describe(value)} has a nonzero imaginary part, which its domain {domain} '
                    f'does not hold. Declare {name.rpartition(".")[2]} with `StateSpec(domain=CC)` to give it complex '
                    'values.'
                )
            raise ValidationError(error_msg)
        flat = LayoutPass._flat_value(name, value, EntrySpec._encode(domain, value, period), shape, members)
        return np.ascontiguousarray(flat).view(float) if domain is CC else np.asarray(flat, dtype=float)

    @staticmethod
    def _noise_processes(nodes: list[NodeSpec]) -> dict[str, WienerProcessSpec]:
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
