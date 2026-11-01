"""Flatten the system tree into the nodes of its elements, groups, and interactions, once per compilation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import re

import numpy as np

from ....core.elements import Element, GroupElement
from ....core.interactions import BaseInteraction
from ....core.spaces import BaseSpace, LatticeSpace, NetworkSpace
from ....core.specs import NodeSpec
from ....core.system import System
from ....errors import LabelError, RoleError, SkcomplexError, ValidationError
from ....specs import WienerProcessSpec
from ....types import VariableKind
from ..._utils import declarations_of
from .._base import BasePass
from .._laws import _AGENT_LAWS
from .._probe import probe_laws

_LABEL_PATTERN = re.compile(r'^[A-Za-z0-9_\-]+$')


class FlattenPass(BasePass):
    """Pass that flattens the system tree into its nodes."""

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
    def _same_network(first: NetworkSpace, second: NetworkSpace) -> bool:
        """Check whether two prepared network spaces have the same nodes, edges, and layers, like network clones."""
        if list(first.graphs_) != list(second.graphs_):
            return False
        if (first.capacity, first.fill, list(first.hyperedges)) != (
            second.capacity,
            second.fill,
            list(second.hyperedges),
        ):
            return False
        return all(
            list(graph.nodes) == list(second.graphs_[layer].nodes)
            and list(graph.edges) == list(second.graphs_[layer].edges)
            and graph.is_directed() == second.graphs_[layer].is_directed()
            for layer, graph in first.graphs_.items()
        )

    @staticmethod
    def _same_process(first: WienerProcessSpec, second: WienerProcessSpec) -> bool:
        """Check whether two Wiener processes of one label have the same dimension and covariance."""
        if first.dimension != second.dimension:
            return False
        return bool(
            np.array_equal(np.asarray(first.covariance, dtype=object), np.asarray(second.covariance, dtype=object)),
        )

    @staticmethod
    def _check_label(label: object, scope: str) -> None:
        """Raise an error for a label that is not a non-empty word."""
        if isinstance(label, str) and _LABEL_PATTERN.match(label):
            return
        where = scope or 'the root system'
        error_msg = (
            f'Label {label!r} in {where} is not valid. A label must be a non-empty string of letters, digits, '
            'underscores, and hyphens.'
        )
        raise LabelError(error_msg)

    @staticmethod
    def _check_network_lattice(space: NetworkSpace, size: int | None, where: str) -> None:
        """Raise an error for a lattice group on a network with a capacity or a size other than its number of nodes."""
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
                space.check().prepare()
            except ValidationError as error:
                error_msg = f'{where} has a lattice that is not valid. {error}'
                raise ValidationError(error_msg) from error
            if space.size_ != group.size:
                error_msg = f'{where} has size {group.size}, but its lattice has {space.size_} sites.'
                raise ValidationError(error_msg)

    @staticmethod
    def _check_part(part: Element | GroupElement | BaseInteraction, scope: str) -> None:
        """Check the arguments of an element, group, or interaction, raising its error with the path of the part."""
        try:
            part.check()
        except SkcomplexError as error:
            error_msg = f'{scope}{part.label}: {error}'
            raise type(error)(error_msg) from error

    @staticmethod
    def _bound_processes(interaction: BaseInteraction, scope: str) -> list[WienerProcessSpec]:
        """Return the Wiener processes that the noise roles of an interaction bind, after checking every role.

        A noise or optional role bound to `None` binds nothing, so an interaction may leave it unbound.
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
    def _check_processes(system: System, scope: str, labels: set[str]) -> None:
        """Raise an error for differing Wiener processes of one label, or a label another child of a system takes."""
        found: dict[str, WienerProcessSpec] = {}
        for interaction in system.interactions:
            for process in FlattenPass._bound_processes(interaction, scope):
                FlattenPass._check_label(process.label, scope)
                known = found.setdefault(process.label, process)
                if not FlattenPass._same_process(known, process):
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
    def _check_children(system: System, scope: str) -> None:
        """Raise an error for a child of the wrong type or a duplicated label."""
        children: list[tuple[Element | GroupElement | BaseInteraction | System, type | tuple[type, ...]]] = [
            *((element, (Element, GroupElement)) for element in system.elements),
            *((interaction, BaseInteraction) for interaction in system.interactions),
            *((child, System) for child in system.systems),
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
            if not isinstance(child, System):
                FlattenPass._check_part(child, scope)
            if isinstance(child, GroupElement):
                FlattenPass._check_group(child, scope)
            if child.label in labels:
                error_msg = f'Label {child.label!r} is used by more than one child of {scope or "the root system"}.'
                raise LabelError(error_msg)
            labels.add(child.label)
        FlattenPass._check_processes(system, scope, labels)

    @staticmethod
    def _canonical(space: BaseSpace | None, seen: list[BaseSpace]) -> BaseSpace | None:
        """Return a prepared space, or the first network space seen that equals it, so clones of a network share one.

        A space is prepared here unless it was prepared before, such as a group space whose ball was enumerated once. A
        space whose arguments are not valid stays unprepared, and validation reports it from its check.
        """
        if space is None:
            return None
        for known in seen:
            if known is space:
                return known
        try:
            if not hasattr(space, 'size_'):
                space.check().prepare()
        except ValidationError:
            seen.append(space)
            return space
        for known in seen:
            if (
                isinstance(known, NetworkSpace)
                and isinstance(space, NetworkSpace)
                and FlattenPass._same_network(known, space)
            ):
                return known
        seen.append(space)
        return space

    @staticmethod
    def _sized(element: Element | GroupElement, space: BaseSpace | None) -> Element | GroupElement:
        """Return a group prepared with its size, a copy when the size counts the parts of the space it inherits."""
        if not isinstance(element, GroupElement):
            return element
        if element.size is None and element.space is None:
            return copy.copy(element).prepare(space)
        return element.check().prepare(space)

    @staticmethod
    def _is_topology_group(component: object, space: BaseSpace | None) -> bool:
        """Check whether a component is a variable group on the sites or edges of a network without agent laws.

        Compiled backends run such a group with a capacity layout, whose rows hold the members present and room for new
        ones, so laws that change the topology of the network add and remove its members.

        Args:
            component:
                The element, group, or interaction.

            space:
                The space of the component.

        Returns:
            topology_group:
                Whether the group runs with a capacity layout.
        """
        if not (isinstance(component, GroupElement) and component.variable and component.on in {'sites', 'edges'}):
            return False
        if not isinstance(space, NetworkSpace):
            return False
        return not any(isinstance(law, _AGENT_LAWS) for law in probe_laws(component) or [])

    @staticmethod
    def _with_capacity(
        element: Element | GroupElement,
        space: BaseSpace | None,
    ) -> tuple[Element | GroupElement, int | None]:
        """Return a variable group on a network with its capacity as its size, and its initial size, or the element."""
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
        system: System,
        scope: str,
        space: BaseSpace | None,
        seen: list[BaseSpace] | None = None,
    ) -> list[NodeSpec]:
        """Return the nodes of a system tree in tree order."""
        if not isinstance(system, System):
            error_msg = f'{system!r} is not an instance of System.'
            raise ValidationError(error_msg)
        seen = [] if seen is None else seen
        space = FlattenPass._canonical(system.space if system.space is not None else space, seen)
        places = [FlattenPass._canonical(getattr(element, 'space', None) or space, seen) for element in system.elements]
        FlattenPass._check_children(system, scope)
        nodes = []
        for element, place in zip(system.elements, places, strict=True):
            element.check().prepare(place)
            sized, active = FlattenPass._with_capacity(FlattenPass._sized(element, space), place)
            if isinstance(sized, GroupElement):
                path, members = f'{scope}{element.label}', sized.size_
                node = NodeSpec(path, scope, sized, place, sized.element, members, active=active)
            else:
                node = NodeSpec(f'{scope}{element.label}', scope, sized, place, type(sized))
            nodes.append(node)
        nodes += [
            NodeSpec(
                f'{scope}{interaction.label}',
                scope,
                interaction.check().prepare(),
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
    def _flatten_system(system: System, scope: str = '', space: BaseSpace | None = None) -> list[NodeSpec]:
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
