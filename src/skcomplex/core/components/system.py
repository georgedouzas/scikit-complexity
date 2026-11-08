"""Define systems.

A system composes elements, interactions, and nested systems on a space, in a time.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import copy
import functools
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar, TypedDict

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from ..base import BaseConfig
from ..errors import RoleElementError, TagError, ValidationError
from ..specs import ObservableSpec, Tags
from .base import BasePart
from .elements import BaseElement
from .interactions import BaseInteraction
from .spaces import BaseSpace, BaseTime


class RunSettings(TypedDict):
    """Tuned run settings of a system.

    `simulate` uses them for every argument a call leaves out.
    """

    until: float | None
    dt: float | None
    solver: BaseConfig | None
    observables: Sequence[ObservableSpec]
    initial_state: Mapping[str, object] | None
    invariants: Mapping[str, float] | None
    schedule: Mapping[str, Callable[[float], float | npt.ArrayLike]] | None
    compiler: BaseConfig | None
    backend: BaseConfig | None


class BaseSystem(BasePart[BaseSpace, SimpleNamespace]):
    """Base of every system.

    A system builds its parts from its model parameters, prepares them, and derives its tags. A subclass takes the
    parameters of its model as arguments of its own `__init__`, which stores each under its own name, as in
    scikit-learn, so `get_params`, `set_params`, and `clone` work on them. It builds its parts in the hook
    `_build_system`, which returns the generic composite `System` of its elements, interactions, nested systems, space,
    and time, written in terms of the model parameters. `prepare` passes the current model parameters down to the parts
    by building them again when a parameter differs by value from those of the last build, and prepares them as
    `system_`, so a parameter changed with `set_params` changes the parts of the next `prepare`. The tags of a system
    are derived from its parts and never declared.

    A subclass may give tuned run settings as class variables, which `simulate` reads for the arguments a call leaves
    out: `until`, the end time, `dt`, the time between frames, `solver`, the solver, `observables`, the observables,
    `initial_state`, the initial values by dotted state name, `invariants`, the tolerance of every observable that
    must stay constant, `schedule`, the functions of time that drive parameters, `compiler`, the compiler of a form
    the system brings, and `backend`, the backend. The compiler also reads `observables` of a `System`
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
    invariants: ClassVar[Mapping[str, float] | None] = None
    schedule: ClassVar[Mapping[str, Callable[[float], float | npt.ArrayLike]] | None] = None
    compiler: ClassVar[BaseConfig | None] = None
    backend: ClassVar[BaseConfig | None] = None

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
        """Return the composite of the parts of the system.

        The parts are built from its model parameters.
        """

    def _is_equal(self, old: object, new: object) -> bool:
        """Return whether two values of a model parameter are equal by value.

        Estimators are equal when they have the same class and equal parameters, sequences and mappings when their
        items are equal, and arrays when they have the same shape and entries. A value that cannot be compared is
        taken as changed.

        Args:
            old:
                The value the parts were last built from.

            new:
                The current value.

        Returns:
            equal:
                Whether the values are equal.
        """
        if old is new:
            return True
        if type(old) is not type(new):
            return False
        if isinstance(old, BaseEstimator) and isinstance(new, BaseEstimator):
            old, new = old.get_params(deep=False), new.get_params(deep=False)
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            equal = old.keys() == new.keys() and all(self._is_equal(old[key], new[key]) for key in old)
        elif isinstance(old, list | tuple) and isinstance(new, list | tuple):
            equal = len(old) == len(new) and all(self._is_equal(*pair) for pair in zip(old, new, strict=True))
        elif isinstance(old, np.ndarray) and isinstance(new, np.ndarray):
            equal = old.shape == new.shape and bool(np.array_equal(old, new))
        else:
            try:
                equal = bool(old == new)
            except (TypeError, ValueError):
                equal = False
        return equal

    def _pass_parameters(self) -> System:
        """Return the parts of the system built from its current model parameters.

        The parts are built again only when a model parameter differs by value from those of the last build, so a
        system that fits or generates data when it builds its parts does so once per set of parameters, however often
        `get_tags`, `_check`, and `prepare` read them.

        Returns:
            system:
                The composite of the parts.

        Raises:
            ValidationError:
                The hook `_build_system` returns no `System`.
        """
        params = self.get_params(deep=False)
        built: tuple[dict[str, object], System] | None = getattr(self, '_built_from', None)
        if built is not None and self._is_equal(built[0], params):
            return built[1]
        system = self._build_system()
        if not isinstance(system, System):
            error_msg = f'{type(self).__name__} builds its parts as a System, not as {type(system).__name__}.'
            raise ValidationError(error_msg)
        self._built_from = (params, system)
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

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings of the system.

        `simulate` uses them for every argument a call leaves out. The settings are the class attributes `until`, `dt`,
        `solver`, `observables`, `initial_state`, `invariants`, `schedule`, `compiler`, and `backend`. A system whose
        settings depend on its model parameters or on its built parts overrides this method.

        Returns:
            settings:
                The tuned run settings by name.

        Examples:
            >>> from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
            >>> sorted(GrayScottSystem().get_run_settings())
            ['backend', 'compiler', 'dt', 'initial_state', 'invariants', 'observables', 'schedule', 'solver', 'until']
        """
        return RunSettings(
            until=self.until,
            dt=self.dt,
            solver=self.solver,
            observables=self.observables,
            initial_state=self.initial_state,
            invariants=self.invariants,
            schedule=self.schedule,
            compiler=self.compiler,
            backend=self.backend,
        )


class System(BaseSystem):
    """Composition of elements and the interactions between them, with nested systems.

    The system lives on a space, in a time. A system is defined by its parts and by nothing about how it runs. An
    interaction refers to an element by its path relative to the system that holds the interaction, so `forest/foxes`
    names the element `foxes` of the nested system `forest`. Groups and nested systems without a space of their own live
    on the space of the system.

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
        systems: Sequence[BaseSystem] = (),
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
                The nested systems, composites or system classes such as the ready-made systems of the catalog, each
                with a label of its own.

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
        """Return the parts of the system.

        They are its space, its time, its elements, its interactions, and its systems.
        """
        around = [part for part in (self.space, self.time) if part is not None]
        return [*around, *self.elements, *self.interactions, *self.systems]

    def _check(self) -> None:
        """Check every part of the system."""
        for part in self._get_parts():
            part._validated()

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Prepare the space, the time, the elements, the nested systems, and interactions.

        The space and the time come first, then the elements and the nested systems on the space, then the interactions.

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
        self.system_ = self._get_composite()

    def _nests_classes(self) -> bool:
        """Return whether a nested system, at any depth, is a system class.

        The alternative is a composite.
        """
        return any(not isinstance(system, System) or system._nests_classes() for system in self.systems)

    def _get_composite(self) -> System:
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

        The result is `None` when no element of a system has the path.
        """
        *scopes, label = path.split('/')
        system: System | None = self
        for scope in scopes:
            system = next((item.system_ for item in system.systems if item.label == scope), None) if system else None
        if system is None:
            return None
        return next((element for element in system.elements if element.label == label), None)

    def _check_roles(self) -> None:
        """Raise an error for a role bound to a state of a domain the role does not accept.

        The state belongs to the element the role binds. The states checked are those the role reads, or every state of
        the element when the role reads none.

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
        """Return the values of the time or the space that parts agree on.

        An error is raised when they disagree. Parts that disagree are joined by the nested systems when each of them
        shares a value with one, as in a co-simulation where interactions of a discrete and of a continuous time act on
        nested systems of each time.

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
        """Return the tags derived from the parts.

        The space is `'none'` for a system without any space.

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
        """Return the tags derived from the parts, without a space for a spaceless system.

        Returns:
            tags:
                The tags of the system.

        Raises:
            TagError:
                Two parts disagree on the time or the space, and the system has none of its own that joins them.
        """
        parts = {part.label: part.get_tags() for part in (*self.elements, *self.interactions)}
        nested = []
        for system in self.systems:
            tags = system._pass_parameters()._get_joined_tags()
            nested.append(replace(tags, randomness='stochastic') if system._draws() else tags)
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
