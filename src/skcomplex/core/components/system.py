"""Define systems, which compose elements, interactions, and nested systems on a space, in a time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import functools
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

from ..base import BaseConfig
from ..errors import RoleElementError, TagError, ValidationError
from ..specs import ObservableSpec, Tags
from .base import BasePart
from .elements import BaseElement
from .interactions import BaseInteraction
from .spaces import BaseSpace, BaseTime


class BaseSystem(BasePart[BaseSpace, SimpleNamespace]):
    """Base of every system: it builds its parts from its model parameters, prepares them, and derives its tags.

    A subclass takes the parameters of its model as arguments of its own `__init__`, which stores each under its own
    name, as in scikit-learn, so `get_params`, `set_params`, and `clone` work on them. It builds its parts in the hook
    `_build_system`, which returns the generic composite `System` of its elements, interactions, nested systems, space,
    and time, written in terms of the model parameters. `prepare` passes the current model parameters down to the parts
    by building them again, and prepares them as `system_`, so a parameter changed with `set_params` changes the parts
    of the next `prepare`. The tags of a system are derived from its parts and never declared.

    A subclass may give tuned run settings as class variables, which `simulate` reads for the arguments a call leaves
    out: `until`, the end time, `dt`, the time between frames, `solver`, the solver, `observables`, the observables,
    and `initial_state`, the initial values by dotted state name. The compiler also reads `observables` of a `System`
    it compiles, so the tuned observables of a subclass of `System` are computed without being passed again.

    Attributes:
        system_:
            The composite of the parts built from the model parameters at the last `prepare`.

        laws_:
            The laws the system contributes for the namespace of its last run.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import RateInteraction
        >>> from skcomplex.core.components.system import BaseSystem, System
        >>> from skcomplex.core.specs import StateSpec
        >>> class TankElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.x = x
        >>> class DecaySystem(BaseSystem):
        ...     def __init__(self, label='decay', k=1.0):
        ...         super().__init__(label)
        ...         self.k = k
        ...
        ...     def _build_system(self):
        ...         decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -self.k * s.target.x)
        ...         return System(self.label, elements=[TankElement('tank')], interactions=[decay])
        >>> system = DecaySystem(k=0.5).prepare()
        >>> system.get_params()
        {'k': 0.5, 'label': 'decay'}
        >>> sorted(system.get_tags().laws)
        ['rate']
    """

    until: ClassVar[float | None] = None
    dt: ClassVar[float | None] = None
    solver: ClassVar[BaseConfig | None] = None
    observables: Sequence[ObservableSpec] = ()
    initial_state: ClassVar[Mapping[str, object] | None] = None

    system_: System
    laws_: Sequence[BasePart]

    def __init__(self, label: str = 'system') -> None:
        """Store the label of the system.

        Args:
            label:
                The label of the system, which names it when it is nested in another one.
        """
        super().__init__(label)

    @abstractmethod
    def _build_system(self) -> System:
        """Return the composite of the parts of the system, built from its model parameters."""

    def _pass_parameters(self) -> System:
        """Return the parts of the system built from its current model parameters.

        Returns:
            system:
                The composite of the parts.

        Raises:
            ValidationError:
                The hook `_build_system` returns no `System`.
        """
        system = self._build_system()
        if not isinstance(system, System):
            error_msg = f'{type(self).__name__} builds its parts as a System, not as {type(system).__name__}.'
            raise ValidationError(error_msg)
        return system

    def _check(self) -> None:
        """Check the parts built from the model parameters."""
        self._pass_parameters()._validated()

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Pass the model parameters down to the parts, and prepare them as `system_`.

        Args:
            context:
                The space a nested system inherits from the system that holds it, or `None`.
        """
        self.system_ = self._pass_parameters().prepare(context)

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Set `laws_`, the laws of the system for a namespace of symbols.

        Args:
            inputs:
                The namespace of the symbols the laws can use.
        """
        self.laws_ = self._laws_of(inputs)

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws of the prepared parts for a namespace of symbols.

        Args:
            namespace:
                The namespace of the system.

        Returns:
            laws:
                The laws of the parts.
        """
        return self.system_._get_laws(namespace)

    def _get_tags(self) -> Tags:
        """Return the tags derived from the parts built from the current model parameters."""
        return self._pass_parameters()._get_tags()


class System(BaseSystem):
    """Composition of elements and the interactions between them, with nested systems, on a space, in a time.

    A system is defined by its parts and by nothing about how it runs. An interaction refers to an element by its path
    relative to the system that holds the interaction, so `forest/foxes` names the element `foxes` of the nested system
    `forest`. Groups and nested systems without a space of their own live on the space of the system.

    The tags of a system come from its parts: its own time, or else the times its parts agree on, every space of its
    parts and nested systems, or `'none'` without any, the union of the domains, law kinds, and structures of its
    parts, and a stochastic randomness when any part is stochastic. `prepare` rejects an element bound to a role that
    does not accept the domains of the states the role reads.

    Attributes:
        system_:
            The system itself, prepared.

        laws_:
            The laws the system contributes for the namespace of its last run.

    Examples:
        >>> from typing import Annotated
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.components.system import System
        >>> class DecayElement(Element):
        ...     law_classes = (RateLaw,)
        ...
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.x = x
        ...
        ...     def _get_laws(self, namespace):
        ...         return [RateLaw(namespace.x, -namespace.x)]
        >>> system = System(elements=[DecayElement('decay')]).prepare()
        >>> namespace = SimpleNamespace(decay=SimpleNamespace(x=var('x')))
        >>> [type(law).__name__ for law in system.run(namespace).laws_]
        ['RateLaw']
        >>> tags = system.get_tags()
        >>> sorted(tags.time), sorted(tags.space), sorted(tags.laws), tags.randomness
        (['continuous'], ['none'], ['rate'], 'deterministic')
    """

    def __init__(
        self,
        label: str = 'system',
        *,
        elements: Sequence[BaseElement] = (),
        interactions: Sequence[BaseInteraction] = (),
        systems: Sequence[System] = (),
        space: BaseSpace | None = None,
        time: BaseTime | None = None,
    ) -> None:
        """Store the label and the parts of the system.

        Args:
            label:
                The label of the system, which names it when it is nested in another one.

            elements:
                The elements and groups.

            interactions:
                The interactions between the elements.

            systems:
                The nested systems.

            space:
                The space the elements live on, which gives vector states their dimension, or `None` for none.

            time:
                The time the system evolves in, `ContinuousTime`, `DiscreteTime` with its schedule, or `StaticTime`, or
                `None` to infer it from the laws.
        """
        super().__init__(label)
        self.elements = elements
        self.interactions = interactions
        self.systems = systems
        self.space = space
        self.time = time

    def _build_system(self) -> System:
        """Return the system itself, whose parts are its arguments."""
        return self

    def _get_parts(self) -> list[BasePart]:
        """Return the parts of the system: its space, its time, its elements, its interactions, and its systems."""
        around = [part for part in (self.space, self.time) if part is not None]
        return [*around, *self.elements, *self.interactions, *self.systems]

    def _check(self) -> None:
        """Check every part of the system."""
        for part in self._get_parts():
            part._validated()

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Prepare the space and the time, then the elements and the nested systems on the space, then the interactions.

        Args:
            context:
                The space a nested system inherits from the system that holds it, or `None`.

        Raises:
            RoleElementError:
                A role is bound to an element with a state of a domain the role does not accept.
        """
        space = self.space if self.space is not None else context
        for part in (self.space, self.time):
            if part is not None:
                part.prepare()
        for part in (*self.elements, *self.systems):
            part.prepare(space)
        for part in self.interactions:
            part.prepare(space)
        self._check_roles()
        self.system_ = self

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws of every part, each for the namespace of its label.

        Args:
            namespace:
                The namespace of the system, which holds the namespace of every part under its label.

        Returns:
            laws:
                The laws of the space, the elements, the interactions, and the nested systems, in that order.
        """
        return [
            law
            for part in self._get_parts()
            for law in part._get_laws(getattr(namespace, part.label, SimpleNamespace()))
        ]

    @staticmethod
    def _get_bound_labels(value: object) -> list[str]:
        """Return the paths of the elements a role binds, without the index of a group member.

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
        """Return the element at a path relative to the system, or `None` when no element of a system has the path."""
        *scopes, label = path.split('/')
        system: System | None = self
        for scope in scopes:
            system = next((item for item in system.systems if item.label == scope), None) if system else None
        if system is None:
            return None
        return next((element for element in system.elements if element.label == label), None)

    def _check_roles(self) -> None:
        """Raise an error for a role bound to an element with a state of a domain the role does not accept.

        The states checked are those the role reads, or every state of the element when the role reads none.

        Raises:
            RoleElementError:
                A role does not accept the domain of a state of its element, with the role, the element, the state, and
                the domain in the message.
        """
        for interaction in self.interactions:
            for role, needs in interaction.get_requirements().roles.items():
                if not needs.domains:
                    continue
                for path in self._get_bound_labels(getattr(interaction, role, None)):
                    element = self._find_element(path)
                    if element is None:
                        continue
                    states = element.get_states()
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
    def _join_tag(name: str, tags: Mapping[str, Tags], joining: frozenset[str]) -> frozenset[str]:
        """Return the values of the time or the space that parts agree on, raising an error when they disagree.

        Parts that disagree are joined by the nested systems when each of them shares a value with one, as in a
        co-simulation where interactions of a discrete and of a continuous time act on nested systems of each time.

        Args:
            name:
                `'time'` or `'space'`.

            tags:
                The tags of the parts by label.

            joining:
                The values of the tag of the nested systems.

        Returns:
            values:
                The values the parts share, or the values of the nested systems that join them, empty when no part has
                one.

        Raises:
            TagError:
                Two parts have values of the tag that no value joins, both named in the message.
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
            f'The parts {first!r} and {second!r} disagree on the tag {name!r}: {first!r} has '
            f'{", ".join(sorted(valued[first]))} and {second!r} has {", ".join(sorted(valued[second]))}. Give the '
            f'system a {name} that joins them, or bind parts of one {name}.'
        )
        raise TagError(error_msg)

    def _get_tags(self) -> Tags:
        """Return the tags derived from the parts, with the space `'none'` for a system without any space.

        Returns:
            tags:
                The tags of the system.

        Raises:
            TagError:
                Two parts disagree on the time or the space, and the system has none of its own that joins them.
        """
        tags = self._get_joined_tags()
        return tags if tags.space else replace(tags, space=frozenset({'none'}))

    def _get_joined_tags(self) -> Tags:
        """Return the tags derived from the parts, without a space for a system without any.

        Returns:
            tags:
                The tags of the system.

        Raises:
            TagError:
                Two parts disagree on the time or the space, and the system has none of its own that joins them.
        """
        parts = {part.label: part.get_tags() for part in (*self.elements, *self.interactions)}
        nested = [system._pass_parameters()._get_joined_tags() for system in self.systems]
        nested_time = frozenset[str]().union(*(item.time for item in nested))
        nested_space = frozenset[str]().union(*(item.space for item in nested))
        time = self.time.get_tags().time if self.time is not None else self._join_tag('time', parts, nested_time)
        space = self.space.get_tags().space if self.space is not None else self._join_tag('space', parts, nested_space)
        every = [*parts.values(), *nested]
        laws = [item.laws for item in every]
        randomness = {item.randomness for item in every}
        return Tags(
            time=time | nested_time,
            domains=frozenset().union(*(item.domains for item in every)),
            space=space.union(nested_space, *(item.space for item in parts.values())),
            laws=None if None in laws else frozenset().union(*(item for item in laws if item is not None)),
            randomness=(
                'stochastic' if 'stochastic' in randomness else None if None in randomness else 'deterministic'
            ),
            structure=frozenset().union(*(item.structure for item in every)),
        )
