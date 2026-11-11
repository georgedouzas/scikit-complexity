"""Flatten the system tree into the nodes of its parts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import re

import numpy as np

from ....components.elements import DeclaredElement, Element, GroupElement
from ....components.interactions import BaseInteraction
from ....components.spaces import BaseSpace, LatticeSpace, NetworkSpace
from ....components.specs import NodeSpec
from ....components.systems import CompositeSystem
from ....errors import LabelError, RoleError, SkcomplexError, ValidationError
from ....specs import WienerProcessSpec
from ....types import VariableKind
from ..._utils import declarations_of, probe_laws
from .._base import BasePass
from .._laws import AGENT_LAWS

LABEL_PATTERN = re.compile(r'^[A-Za-z0-9_\-]+$')
TOPOLOGY_PARTS = ('sites', 'edges', 'hyperedges')


class FlattenPass(BasePass):
    """A pass that flattens the system tree into its nodes."""

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
    def _check_group(group: GroupElement, scope: str) -> None:
        """Raise an error for a group with an element type or size that is not valid."""
        where = f'Group {group.label!r} in {scope or "the root system"}'
        space = getattr(group, 'space', None)
        if isinstance(space, NetworkSpace) and not group.variable and group.on in {None, 'sites'}:
            FlattenPass._check_network_lattice(space, group.size, where)
        if group.size is None:
            return
        if isinstance(space, LatticeSpace):
            try:
                space.prepare()
            except ValidationError as error:
                error_msg = f'{where} has a lattice that is not valid. {error}'
                raise ValidationError(error_msg) from error
            if space.size_ != group.size:
                error_msg = f'{where} has size {group.size}, but its lattice has {space.size_} sites.'
                raise ValidationError(error_msg)

    @staticmethod
    def _prepare_part(part: Element | GroupElement | BaseInteraction, place: BaseSpace | None, scope: str) -> None:
        """Prepare an element, group, or interaction on its place, checking its arguments.

        Its error is raised with the path of the part.
        """
        try:
            part.prepare(place)
        except SkcomplexError as error:
            error_msg = f'{scope}{part.label}: {error}'
            raise type(error)(error_msg) from error

    @staticmethod
    def _get_bound_processes(interaction: BaseInteraction, scope: str) -> list[WienerProcessSpec]:
        """Return the Wiener processes that the noise roles of an interaction bind.

        Every role is checked first. A noise or optional role bound to `None` binds nothing, so an interaction may leave
        it unbound.
        """
        processes = []
        for name, declaration in declarations_of(type(interaction), VariableKind.ROLE).items():
            value = getattr(interaction, name)
            if (declaration.noise or declaration.optional) and value is None:
                continue
            if declaration.noise != isinstance(value, WienerProcessSpec):
                expected = 'a WienerProcessSpec' if declaration.noise else 'an element or group path'
                error_msg = (
                    f'Interaction {scope}{interaction.label} binds its role {name!r} to {value!r}, but the role takes '
                    f'{expected}. Bind {expected}, and declare a role of Wiener processes with `role(noise=True)`.'
                )
                raise RoleError(error_msg)
            if declaration.noise:
                processes.append(value)
        return processes

    @staticmethod
    def _check_processes(system: CompositeSystem, scope: str, labels: set[str]) -> None:
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
    def _check_children(system: CompositeSystem, scope: str, places: dict[int, BaseSpace | None]) -> None:
        """Raise an error for a child of the wrong type or a duplicated label.

        Every element and interaction is prepared on its place, by the identity of the part, which checks it.
        """
        children: list[tuple[Element | GroupElement | BaseInteraction | CompositeSystem, type | tuple[type, ...]]] = [
            *((element, (Element, GroupElement)) for element in system.elements),
            *((interaction, BaseInteraction) for interaction in system.interactions),
            *((child, CompositeSystem) for child in system.systems),
        ]
        labels: set[str] = set()
        for child, expected_type in children:
            if not isinstance(child, expected_type):
                names = ' or '.join(
                    kind.__name__ for kind in (expected_type if isinstance(expected_type, tuple) else (expected_type,))
                )
                error_msg = f'{child!r} in {scope or "the root system"} is not an instance of {names}.'
                raise ValidationError(error_msg)
            FlattenPass._check_label(child.label, scope)
            if not isinstance(child, CompositeSystem):
                FlattenPass._prepare_part(child, places.get(id(child)), scope)
            if isinstance(child, GroupElement):
                FlattenPass._check_group(child, scope)
            if child.label in labels:
                error_msg = f'Label {child.label!r} is used by more than one child of {scope or "the root system"}.'
                raise LabelError(error_msg)
            labels.add(child.label)
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
    def _get_sized(element: Element | GroupElement, space: BaseSpace | None) -> Element | GroupElement:
        """Return a group prepared with its size.

        It is a copy when the size counts the parts of the space it inherits.
        """
        if not isinstance(element, GroupElement):
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
        if not (isinstance(component, GroupElement) and component.variable and component.on in TOPOLOGY_PARTS):
            return False
        if not isinstance(space, NetworkSpace):
            return False
        return not any(isinstance(law, AGENT_LAWS) for law in probe_laws(component) or [])

    @staticmethod
    def _copy_with_capacity(
        element: Element | GroupElement,
        space: BaseSpace | None,
    ) -> tuple[Element | GroupElement, int | None]:
        """Return a variable group on a network with its capacity as its size.

        It also returns its initial size, or the element otherwise.
        """
        if not isinstance(element, GroupElement) or not FlattenPass._is_topology_group(element, space):
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
    def _flatten(
        system: CompositeSystem,
        scope: str,
        space: BaseSpace | None,
        seen: list[BaseSpace] | None = None,
    ) -> list[NodeSpec]:
        """Return the nodes of a system tree in tree order."""
        if not isinstance(system, CompositeSystem):
            error_msg = f'{system!r} is not an instance of CompositeSystem.'
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
        nodes = []
        for element, place in zip(system.elements, places, strict=True):
            sized, active = FlattenPass._copy_with_capacity(FlattenPass._get_sized(element, space), place)
            if isinstance(sized, GroupElement):
                path, members = f'{scope}{element.label}', sized.size_
                node = NodeSpec(path, scope, sized, place, sized.get_declaring_element(), members, active=active)
            else:
                declaring = sized if isinstance(sized, DeclaredElement) else type(sized)
                node = NodeSpec(f'{scope}{element.label}', scope, sized, place, declaring)
            nodes.append(node)
        nodes += [
            NodeSpec(
                f'{scope}{interaction.label}',
                scope,
                interaction,
                space,
                type(interaction),
                interaction=True,
            )
            for interaction in system.interactions
        ]
        for child in system.systems:
            nodes += FlattenPass._flatten(child, f'{scope}{child.label}/', space, seen)
        return nodes

    @staticmethod
    def _flatten_system(system: CompositeSystem, scope: str = '', space: BaseSpace | None = None) -> list[NodeSpec]:
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
