"""List every component of a nested system in one flat list."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import re
from collections.abc import Mapping

import numpy as np

from ....components.elements import BaseElement, Element
from ....components.interactions import BaseInteraction, WienerProcessSpec
from ....components.spaces import BaseSpace, LatticeSpace, NetworkSpace
from ....enums import VariableKind
from ....errors import LabelError, RoleError, SkcomplexError, ValidationError
from ..._utils import binding_of, declarations_of, probe_laws
from ...problems._composite_system import _CompositeSystem
from ...problems._declared_element import _DeclaredElement
from ...problems._group_element import _GroupElement
from ...problems._node_spec import _NodeSpec
from .._base import BasePass
from .._laws import AGENT_LAWS

LABEL_PATTERN = re.compile(r'^[A-Za-z0-9_\-]+$')
TOPOLOGY_PARTS = ('sites', 'edges', 'hyperedges')


class FlattenPass(BasePass):
    """A pass that lists every component of a nested system in one flat list.

    A system may nest other systems, such as villages inside a country, and this pass walks the tree once and lists
    every element, group, and interaction with its full path, such as `village.households`. The passes after it read
    this list in place of the tree.

    Read more in the user guide, under Extend the library.

    See Also:
        CensusPass: Lists the laws of the components.
        PassCompiler: Runs the passes over a compilation.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> compilation = PassCompiler().prepare(LotkaVolterraSystem()).compilation_
        >>> [node.path for node in compilation.nodes]
        ['prey', 'predators', 'growth', 'predation', 'death']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('system',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('nodes',)

    def _run(self, inputs: object = None) -> None:
        """Flatten the system into its nodes."""
        compilation = self.compilation_
        compilation.nodes = self._flatten_system(compilation.system)

    @staticmethod
    def _get_same_network(first: NetworkSpace, second: NetworkSpace) -> bool:
        """Check whether two prepared network spaces are alike.

        Like network clones, they have the same nodes, edges, and layers.
        """
        if list(first.graphs_) != list(second.graphs_):
            return False
        if (first.capacity, first.fill, first.hyperedges_) != (second.capacity, second.fill, second.hyperedges_):
            return False
        return all(
            list(graph.nodes) == list(second.graphs_[layer].nodes)
            and list(graph.edges) == list(second.graphs_[layer].edges)
            and graph.is_directed() == second.graphs_[layer].is_directed()
            for layer, graph in first.graphs_.items()
        )

    @staticmethod
    def _get_same_process(first: WienerProcessSpec, second: WienerProcessSpec) -> bool:
        """Check whether two Wiener processes of one label match.

        They match when they have the same dimension and covariance.
        """
        if first.dimension != second.dimension:
            return False
        return bool(
            np.array_equal(np.asarray(first.covariance, dtype=object), np.asarray(second.covariance, dtype=object)),
        )

    @staticmethod
    def _check_label(label: object, scope: str) -> None:
        """Raise an error for a label that is not a non-empty word."""
        if isinstance(label, str) and LABEL_PATTERN.match(label):
            return
        where = scope or 'the root system'
        error_msg = (
            f'Label {label!r} in {where} is not valid. A label must be a non-empty string of letters, digits, '
            'underscores, and hyphens.'
        )
        raise LabelError(error_msg)

    @staticmethod
    def _check_network_lattice(space: NetworkSpace, size: int | None, where: str) -> None:
        """Raise an error for a lattice group on a network of a wrong size.

        The group has a capacity, or a size other than the number of nodes.
        """
        if space.capacity is not None:
            error_msg = (
                f'{where} has its sites on a network with capacity {space.capacity}, but sites hold states, not '
                'agents. Leave the capacity as None.'
            )
            raise ValidationError(error_msg)
        if size is not None and space.graph.number_of_nodes() != size:
            error_msg = f'{where} has size {size}, but its network has {space.graph.number_of_nodes()} nodes.'
            raise ValidationError(error_msg)

    @staticmethod
    def _check_group(group: _GroupElement, scope: str) -> None:
        """Raise an error for a group with an element type or size that is not valid."""
        where = f'Group {group.label!r} in {scope or "the root system"}'
        space = getattr(group, 'space', None)
        if isinstance(space, NetworkSpace) and not group.variable and group.on in {None, 'sites'}:
            FlattenPass._check_network_lattice(space, group.size, where)
        if group.size is None:
            return
        if isinstance(space, LatticeSpace) and group.on is not None:
            try:
                space.prepare()
            except ValidationError as error:
                error_msg = f'{where} has a lattice that is not valid. {error}'
                raise ValidationError(error_msg) from error
            if space.size_ != group.size:
                error_msg = f'{where} has size {group.size}, but its lattice has {space.size_} sites.'
                raise ValidationError(error_msg)

    @staticmethod
    def _prepare_component(
        component: Element | _GroupElement | BaseInteraction,
        place: BaseSpace | None,
        scope: str,
    ) -> None:
        """Prepare an element, group, or interaction on its place, checking its arguments.

        Its error is raised with the path of the component.
        """
        try:
            component._prepare_on(place)
        except SkcomplexError as error:
            error_msg = f'{scope}{component.label}: {error}'
            raise type(error)(error_msg) from error

    @staticmethod
    def _get_bound_processes(interaction: BaseInteraction, scope: str) -> list[WienerProcessSpec]:
        """Return the Wiener processes that the noise roles of an interaction bind.

        Every role is checked first. A noise or optional role bound to `None` binds nothing, so an interaction may leave
        it unbound.
        """
        processes = []
        for name, declaration in declarations_of(interaction, VariableKind.ROLE).items():
            value = binding_of(interaction, name)
            if (declaration.noise or declaration.optional) and value is None:
                continue
            if declaration.noise != isinstance(value, WienerProcessSpec):
                expected = 'a WienerProcessSpec' if declaration.noise else 'an element or group path'
                error_msg = (
                    f'Interaction {scope}{interaction.label} binds its role {name!r} to {value!r}, but the role takes '
                    f'{expected}. Bind {expected}, and declare a role of Wiener processes with `role(noise=True)`.'
                )
                raise RoleError(error_msg)
            if isinstance(value, WienerProcessSpec):
                processes.append(value)
        return processes

    @staticmethod
    def _check_processes(system: _CompositeSystem, scope: str, labels: set[str]) -> None:
        """Raise an error for conflicting Wiener processes of one label.

        The processes differ, or another child of a system takes the label.
        """
        found: dict[str, WienerProcessSpec] = {}
        for interaction in system.interactions:
            for process in FlattenPass._get_bound_processes(interaction, scope):
                FlattenPass._check_label(process.label, scope)
                known = found.setdefault(process.label, process)
                if not FlattenPass._get_same_process(known, process):
                    error_msg = (
                        f'The interactions of {scope or "the root system"} bind two different Wiener processes '
                        f'labelled {process.label!r}. Bind one process to every interaction that shares it, or give '
                        'them two labels.'
                    )
                    raise LabelError(error_msg)
        clashes = sorted(set(found) & labels)
        if clashes:
            error_msg = (
                f'The Wiener process {clashes[0]!r} has the label of another child of {scope or "the root system"}. '
                'Give it a label of its own.'
            )
            raise LabelError(error_msg)

    @staticmethod
    def _check_child(
        child: Element | _GroupElement | BaseInteraction | _CompositeSystem,
        expected_type: type | tuple[type, ...],
        scope: str,
        place: BaseSpace | None,
    ) -> None:
        """Raise an error for a child of the wrong type or label.

        An element or interaction is prepared on its place, which checks it.
        """
        if not isinstance(child, expected_type):
            names = ' or '.join(
                kind.__name__ for kind in (expected_type if isinstance(expected_type, tuple) else (expected_type,))
            )
            error_msg = f'{child!r} in {scope or "the root system"} is not an instance of {names}.'
            raise ValidationError(error_msg)
        FlattenPass._check_label(child.label, scope)
        if not isinstance(child, _CompositeSystem):
            FlattenPass._prepare_component(child, place, scope)
        if isinstance(child, _GroupElement):
            FlattenPass._check_group(child, scope)

    @staticmethod
    def _check_unique(label: str, labels: set[str], scope: str) -> None:
        """Raise an error for a label another child takes, else add it to the labels."""
        if label in labels:
            error_msg = f'Label {label!r} is used by more than one child of {scope or "the root system"}.'
            raise LabelError(error_msg)
        labels.add(label)

    @staticmethod
    def _check_children(system: _CompositeSystem, scope: str, places: dict[int, BaseSpace | None]) -> None:
        """Raise an error for a child of the wrong type or a duplicated label.

        Every element and interaction is prepared on its place, by the identity of the component, which checks it.
        """
        children: list[tuple[Element | _GroupElement | BaseInteraction | _CompositeSystem, type | tuple[type, ...]]] = [
            *((element, (Element, _GroupElement)) for element in system.elements),
            *((interaction, BaseInteraction) for interaction in system.interactions),
            *((child, _CompositeSystem) for child in system.systems),
        ]
        labels: set[str] = set()
        for child, expected_type in children:
            FlattenPass._check_child(child, expected_type, scope, places.get(id(child)))
            FlattenPass._check_unique(child.label, labels, scope)
        FlattenPass._check_processes(system, scope, labels)

    @staticmethod
    def _get_canonical(space: BaseSpace | None, seen: list[BaseSpace]) -> BaseSpace | None:
        """Return a prepared space, or the first equal network space seen.

        Clones of a network thus share one space. A space is prepared here unless it was prepared before, such as a
        group space whose ball was enumerated once. A space whose arguments are not valid stays unprepared, and
        validation reports it from its check.
        """
        if space is None:
            return None
        for known in seen:
            if known is space:
                return known
        try:
            if not hasattr(space, 'size_'):
                space.prepare()
        except ValidationError:
            seen.append(space)
            return space
        for known in seen:
            if (
                isinstance(known, NetworkSpace)
                and isinstance(space, NetworkSpace)
                and FlattenPass._get_same_network(known, space)
            ):
                return known
        seen.append(space)
        return space

    @staticmethod
    def _get_sized(element: Element | _GroupElement, space: BaseSpace | None) -> Element | _GroupElement:
        """Return a group prepared with its size.

        It is a copy when the size counts the parts of the space it inherits.
        """
        if not isinstance(element, _GroupElement):
            return element
        if element.size is None and element.space is None:
            return copy.copy(element).prepare(space)
        return element.prepare(space)

    @staticmethod
    def _is_topology_group(component: object, space: BaseSpace | None) -> bool:
        """Check whether a component is a variable group on a network.

        It is on the sites, edges, or hyperedges of a network, not of agents. Compiled backends run such a group with a
        capacity layout, whose rows hold the members present and room for new ones, so laws that change the topology of
        the network add and remove its members.

        Args:
            component:
                The element, group, or interaction.

            space:
                The space of the component.

        Returns:
            topology_group:
                Whether the group runs with a capacity layout.
        """
        if not (isinstance(component, _GroupElement) and component.variable and component.on in TOPOLOGY_PARTS):
            return False
        if not isinstance(space, NetworkSpace):
            return False
        return not any(isinstance(law, AGENT_LAWS) for law in probe_laws(component) or [])

    @staticmethod
    def _copy_with_capacity(
        element: Element | _GroupElement,
        space: BaseSpace | None,
    ) -> tuple[Element | _GroupElement, int | None]:
        """Return a variable group on a network with its capacity as its size.

        It also returns its initial size, or the element otherwise.
        """
        if not isinstance(element, _GroupElement) or not FlattenPass._is_topology_group(element, space):
            return element, None
        group = element
        initial = group.size_
        capacity = group.capacity if group.capacity is not None else max(2 * initial, 1)
        if capacity < initial:
            error_msg = (
                f'Group {group.label!r} has the capacity {capacity}, below its {initial} initial members. Give a '
                'capacity of at least its size.'
            )
            raise ValidationError(error_msg)
        sized = copy.copy(group)
        sized.size_ = capacity
        return sized, initial

    @staticmethod
    def _get_element_node(
        element: Element | _GroupElement,
        scope: str,
        space: BaseSpace | None,
        place: BaseSpace | None,
    ) -> _NodeSpec:
        """Return the node of an element or a group of a system with a space, on its place."""
        sized, active = FlattenPass._copy_with_capacity(FlattenPass._get_sized(element, space), place)
        if isinstance(sized, _GroupElement):
            path, members = f'{scope}{element.label}', sized.size_
            return _NodeSpec(path, scope, sized, place, sized.get_declaring_element(), members, active=active)
        declaring = sized if isinstance(sized, _DeclaredElement) else type(sized)
        return _NodeSpec(f'{scope}{element.label}', scope, sized, place, declaring)

    @staticmethod
    def _get_scope_spaces(
        system: _CompositeSystem,
        scope: str = '',
        space: BaseSpace | None = None,
    ) -> dict[str, BaseSpace | None]:
        """Return the space of every system of a tree by its scope.

        It is the own space of the system or the one it inherits.
        """
        space = system.space if system.space is not None else space
        spaces = {scope: space}
        for child in system.systems:
            spaces.update(FlattenPass._get_scope_spaces(child, f'{scope}{child.label}/', space))
        return spaces

    @staticmethod
    def _get_declared_nodes(
        nodes: list[_NodeSpec],
        system: _CompositeSystem,
        declared: Mapping[str, list[BaseElement]],
    ) -> list[_NodeSpec]:
        """Return the nodes of a flat tree with the declared elements in place.

        Each element is replaced by the elements that declare it, such as a group of agents for an element of one
        member. The structure of a tree does not depend on what its elements declare, so the tree is flattened once.

        Args:
            nodes:
                The nodes of the flattened tree.

            system:
                The system tree that holds the declared elements, which gives the space of every scope.

            declared:
                The declared elements by the path of the element they replace.

        Returns:
            nodes:
                The nodes, with the nodes of the declared elements in place of those they replace.

        Raises:
            ValidationError:
                A declared element does not pass its check, named by its path.

            LabelError:
                A declared element takes the label of another child of its system.
        """
        if not declared:
            return nodes
        seen = list({id(node.space): node.space for node in nodes if node.space is not None}.values())
        spaces = FlattenPass._get_scope_spaces(system)
        replaced: list[_NodeSpec] = []
        for node in nodes:
            if node.path not in declared:
                replaced.append(node)
                continue
            space = FlattenPass._get_canonical(spaces[node.scope], seen)
            for element in declared[node.path]:
                place = FlattenPass._get_canonical(getattr(element, 'space', None) or space, seen)
                FlattenPass._check_child(element, (Element, _GroupElement), node.scope, place)
                replaced.append(FlattenPass._get_element_node(element, node.scope, space, place))
        labels: dict[str, set[str]] = {scope: set() for scope in spaces}
        for node in replaced:
            FlattenPass._check_unique(node.path.removeprefix(node.scope), labels[node.scope], node.scope)
        return replaced

    @staticmethod
    def _flatten(
        system: _CompositeSystem,
        scope: str,
        space: BaseSpace | None,
        seen: list[BaseSpace] | None = None,
    ) -> list[_NodeSpec]:
        """Return the nodes of a system tree in tree order."""
        if not isinstance(system, _CompositeSystem):
            error_msg = f'{system!r} is not a system.'
            raise ValidationError(error_msg)
        seen = [] if seen is None else seen
        space = FlattenPass._get_canonical(system.space if system.space is not None else space, seen)
        places = [
            FlattenPass._get_canonical(getattr(element, 'space', None) or space, seen) for element in system.elements
        ]
        FlattenPass._check_children(
            system,
            scope,
            {id(element): place for element, place in zip(system.elements, places, strict=True)},
        )
        nodes = [
            FlattenPass._get_element_node(element, scope, space, place)
            for element, place in zip(system.elements, places, strict=True)
        ]
        nodes += [
            _NodeSpec(
                f'{scope}{interaction.label}',
                scope,
                interaction,
                space,
                interaction if interaction._get_declared() else type(interaction),
                interaction=True,
            )
            for interaction in system.interactions
        ]
        for child in system.systems:
            nodes += FlattenPass._flatten(child, f'{scope}{child.label}/', space, seen)
        return nodes

    @staticmethod
    def _flatten_system(system: _CompositeSystem, scope: str = '', space: BaseSpace | None = None) -> list[_NodeSpec]:
        """Return the elements and interactions of a system tree with their paths.

        Args:
            system:
                The root of the tree.

            scope:
                The path prefix of the system.

            space:
                The space inherited from the parent system.

        Returns:
            nodes:
                The elements and interactions, in tree order, none of them marked as an agent yet.

        Raises:
            ValidationError:
                A child has the wrong type, or a component or interaction does not pass its check, named by its path.

            LabelError:
                A label is not valid or is duplicated among siblings.
        """
        return FlattenPass._flatten(system, scope, space)
