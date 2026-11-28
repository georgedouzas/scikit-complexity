"""Define the system that composes elements, interactions, and nested systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import copy
import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import ClassVar, TypedDict

import numpy.typing as npt

from ...base import BaseConfig
from ...components import Observable, Tags
from ...components._parameter import _Parameter
from ...components._state import _State
from ...components.base import BaseComponent
from ...components.elements import BaseElement
from ...components.interactions import BaseInteraction
from ...components.spaces import BaseSpace
from ...components.spaces._base import _BaseTime
from ...errors import RoleElementError, TagError


class _RunSettings(TypedDict):
    """A mapping of the tuned run settings of a composite, which runs read."""

    until: float | None
    dt: float | None
    solver: BaseConfig | None
    observables: Sequence[Observable]
    initial_state: Mapping[str, object] | None
    invariants: Mapping[str, float] | None
    schedule: Mapping[str, Callable[[float], float | npt.ArrayLike]] | None
    backend: BaseConfig | None


class _CompositeSystem(BaseComponent[BaseSpace]):
    """A system that composes elements, interactions, and nested systems as data.

    The system lives on a space, in a time, and says nothing about how it runs. An interaction refers to an element by
    its path relative to the system that holds the interaction, so `forest/foxes` names the element `foxes` of the
    nested system `forest`. Groups and nested systems without a space of their own live on the space of the system.

    The tags of a system come from its components: its own time, or the time its components agree on, every space of its
    components, the union of their domains, law kinds, and structures, and a stochastic randomness when any component is
    stochastic. `prepare` rejects an element bound to a role that does not accept the domains of the states the role
    reads.

    Attributes:
        system_:
            The system itself, prepared.
    """

    until: ClassVar[float | None] = None
    dt: ClassVar[float | None] = None
    observables: Sequence[Observable] = ()
    initial_state: ClassVar[Mapping[str, object] | None] = None
    invariants: ClassVar[Mapping[str, float] | None] = None
    schedule: ClassVar[Mapping[str, Callable[[float], float | npt.ArrayLike]] | None] = None
    backend: ClassVar[BaseConfig | None] = None

    system_: _CompositeSystem

    def __init__(
        self,
        label: str = 'system',
        *,
        elements: Sequence[BaseElement] = (),
        interactions: Sequence[BaseInteraction] = (),
        systems: Sequence[_CompositeSystem] = (),
        space: BaseSpace | None = None,
        time: _BaseTime | None = None,
        solver: BaseConfig | None = None,
    ) -> None:
        """Store the label and the components of the system.

        Args:
            label:
                The label of the system, which names it when it is nested in another one.

            elements:
                The elements and groups.

            interactions:
                The interactions between the elements.

            systems:
                The nested systems, composites or system classes such as the ready-made systems of the catalog, each
                with a label of its own.

            space:
                The space the elements live on, which gives vector states their dimension, or `None` for none.

            time:
                The time the system evolves in, `_ContinuousTime`, `_DiscreteTime` with its schedule, or
                `_StaticTime`, or `None` to infer it from the laws.

            solver:
                The solver of the laws of the system, which a nested system keeps as it keeps its time, or `None` for
                the solver of the system that holds it.
        """
        super().__init__(label)
        self.elements = elements
        self.interactions = interactions
        self.systems = systems
        self.space = space
        self.time = time
        self.solver = solver

    def get_run_settings(self) -> _RunSettings:
        """Return the tuned run settings, which runs read for the arguments they omit."""
        return _RunSettings(
            until=self.until,
            dt=self.dt,
            solver=self.solver,
            observables=self.observables,
            initial_state=self.initial_state,
            invariants=self.invariants,
            schedule=self.schedule,
            backend=self.backend,
        )

    def _get_components(self) -> list[BaseComponent]:
        """Return the components of the system.

        They are its space, its time, its elements, its interactions, and its systems.
        """
        around = [component for component in (self.space, self.time) if component is not None]
        return [*around, *self.elements, *self.interactions, *self.systems]

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Prepare the space, the time, the elements, the nested systems, and interactions.

        The space and the time come first, then the elements and the nested systems on the space, then the interactions.

        Args:
            context:
                The space a nested system inherits from the system that holds it, or `None`.

        Every component checks its arguments when it is prepared, so a component that fails may leave the components
        before it prepared, and preparing again replaces their attributes. The system checks its label through the base
        of the components, since it is its own composite.

        Raises:
            RoleElementError:
                A role is bound to an element with a state of a domain the role does not accept.
        """
        self._check_specs()
        space = self.space if self.space is not None else context
        for component in (self.space, self.time):
            if component is not None:
                component.prepare()
        for component in (*self.elements, *self.systems):
            component.prepare(space)
        for component in self.interactions:
            component._prepare_on(space)
        self._check_roles()
        self.system_ = self._get_composite()

    def _nests_classes(self) -> bool:
        """Return whether a nested system, at any depth, is a system class.

        The alternative is a composite.
        """
        return any(not isinstance(system, _CompositeSystem) or system._nests_classes() for system in self.systems)

    def _get_composite(self) -> _CompositeSystem:
        """Return the system with the prepared composite of every nested system in place.

        A nested system class, such as a ready-made system of the catalog, is replaced by the composite it builds from
        its model parameters, so the compilers, the solvers, and the trajectory read the elements, the interactions,
        and the systems of every nested system alike. The system itself is returned when every nested system is its own
        composite already.

        Returns:
            composite:
                The system, or a shallow copy of it whose nested systems are composites.
        """
        systems = [system.system_ for system in self.systems]
        if all(composite is system for composite, system in zip(systems, self.systems, strict=True)):
            return self
        composite = copy.copy(self)
        composite.systems = systems
        composite.system_ = composite
        composite._prepared = True
        return composite

    @staticmethod
    def _get_bound_labels(value: object) -> list[str]:
        """Return the paths of the elements a role binds.

        The paths leave out the index of a group member.

        Args:
            value:
                The value of the role: a path, a sequence of paths or of pairs of a path and a state, or a mapping
                whose keys are such pairs.

        Returns:
            paths:
                The paths of the bound elements.
        """
        entries = [value] if isinstance(value, str) else list(value) if isinstance(value, Sequence | Mapping) else []
        labels = [entry[0] if isinstance(entry, tuple) and entry else entry for entry in entries]
        return [label.partition('[')[0] for label in labels if isinstance(label, str)]

    def _find_element(self, path: str) -> BaseElement | None:
        """Return the element at a path relative to the system.

        The result is `None` when no element of a system has the path. A nested system that is not prepared yet is
        searched as it is.
        """
        *scopes, label = path.split('/')
        system: _CompositeSystem | None = self
        for scope in scopes:
            nested = next((item for item in system.systems if item.label == scope), None) if system else None
            system = nested.system_ if nested is not None and hasattr(nested, 'system_') else nested
        if system is None:
            return None
        return next((element for element in system.elements if element.label == label), None)

    def _check_roles(self) -> None:
        """Raise an error for a role bound to a state of a domain the role does not accept.

        The state belongs to the element the role binds. The states checked are those the role reads, or every state of
        the element when the role reads none. An element with no state, such as a lamp whose level no law changes, has
        its parameters checked instead.

        Raises:
            RoleElementError:
                A role does not accept the domain of a state of its element, with the role, the element, the state, and
                the domain in the message.
        """
        for interaction in self.interactions:
            for role, needs in interaction.get_requirements().roles.items():
                if not needs.domains:
                    continue
                for path in self._get_bound_labels(interaction._get_binding(role)):
                    element = self._find_element(path)
                    if element is None:
                        continue
                    states = element.get_states() or self._get_parameter_states(element)
                    read = {name: states[name] for name in needs.states if name in states} or states
                    for name, spec in read.items():
                        domain = spec._get_domain_tag()
                        if domain is None or domain in needs.domains:
                            continue
                        error_msg = (
                            f'The role {role!r} of {interaction.label!r} accepts the domains '
                            f'{", ".join(needs.domains)}, but the element {path!r} bound to it has the state {name!r} '
                            f'of the domain {domain!r}.'
                        )
                        raise RoleElementError(error_msg)

    @staticmethod
    def _get_parameter_states(element: BaseElement) -> dict[str, _State]:
        """Return the parameters of an element as states of their domains, by name."""
        return {
            name: _State(domain=spec.domain, vector=spec.vector, tensor=spec.tensor)
            for name, spec in element._get_component_specs().items()
            if isinstance(spec, _Parameter)
        }

    @staticmethod
    def _join_tag(name: str, tags: Mapping[str, Tags], joining: frozenset[str]) -> frozenset[str]:
        """Return the values of the time or the space that components agree on.

        An error is raised when they disagree. Components that disagree are joined by the nested systems when each of
        them shares a value with one, as in a co-simulation where interactions of a discrete and of a continuous time
        act on nested systems of each time.

        Args:
            name:
                `'time'` or `'space'`.

            tags:
                The tags of the components by label.

            joining:
                The values of the tag of the nested systems.

        Returns:
            values:
                The values the components share, or the values of the nested systems that join them, empty when no
                component has one.

        Raises:
            TagError:
                Two components have values of the tag that no value joins, both named in the message.
        """
        valued = {label: getattr(item, name) for label, item in tags.items() if getattr(item, name)}
        if not valued:
            return frozenset()
        joined: frozenset[str] = functools.reduce(frozenset.intersection, valued.values())
        if joined:
            return joined
        if all(values & joining for values in valued.values()):
            return joining
        labels = list(valued)
        pairs = ((one, other) for index, one in enumerate(labels) for other in labels[index + 1 :])
        first, second = next(
            ((one, other) for one, other in pairs if not valued[one] & valued[other]),
            (labels[0], labels[1]),
        )
        error_msg = (
            f'The components {first!r} and {second!r} disagree on the tag {name!r}: {first!r} has '
            f'{", ".join(sorted(valued[first]))} and {second!r} has {", ".join(sorted(valued[second]))}. Give the '
            f'system a {name} that joins them, or bind components of one {name}.'
        )
        raise TagError(error_msg)

    def _get_tags(self) -> Tags:
        """Return the tags derived from the components.

        The space is `'none'` for a system without any space.

        Returns:
            tags:
                The tags of the system.

        Raises:
            TagError:
                Two components disagree on the time or the space, and the system has none of its own that joins them.
        """
        tags = self._get_joined_tags()
        return tags if tags.space else replace(tags, space=frozenset({'none'}))

    def _get_joined_tags(self) -> Tags:
        """Return the tags of the components, without a space for a spaceless system.

        Returns:
            tags:
                The tags of the system.

        Raises:
            TagError:
                Two components disagree on the time or the space, and the system has none of its own that joins them.
        """
        components = {element.label: element.get_tags() for element in self.elements} | {
            interaction.label: interaction.get_tags() for interaction in self.interactions
        }
        nested = []
        for system in self.systems:
            tags = system._get_joined_tags()
            nested.append(replace(tags, randomness='stochastic') if system._get_draws() else tags)
        nested_time = frozenset[str]().union(*(item.time for item in nested))
        nested_space = frozenset[str]().union(*(item.space for item in nested))
        time = self.time.get_tags().time if self.time is not None else self._join_tag('time', components, nested_time)
        space = (
            self.space.get_tags().space if self.space is not None else self._join_tag('space', components, nested_space)
        )
        every = [*components.values(), *nested]
        laws = [item.laws for item in every]
        randomness = {item.randomness for item in every}
        stochastic = [item for item in every if item.randomness == 'stochastic']
        return Tags(
            time=time | nested_time,
            domains=frozenset().union(*(item.domains for item in every)),
            space=space.union(nested_space, *(item.space for item in components.values())),
            laws=None if None in laws else frozenset().union(*(item for item in laws if item is not None)),
            randomness=(
                'stochastic' if 'stochastic' in randomness else None if None in randomness else 'deterministic'
            ),
            structure=frozenset().union(*(item.structure for item in every)),
            reparameterized=bool(stochastic) and all(item.reparameterized for item in stochastic),
        )
