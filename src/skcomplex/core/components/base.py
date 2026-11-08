"""Define the base of the parts of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import functools
import inspect
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar, Self

from ..base import BaseConfig
from ..errors import ValidationError
from ..specs import STOCHASTIC_LAWS, DeclaredVariableSpec, ParameterSpec, RoleSpec, Spec, StateSpec, Tags


class BasePart[ContextT, InputT](BaseConfig[ContextT, InputT]):
    """Base of every part of a system.

    The parts of a system are its elements, its interactions, the laws they return, and the system itself. Every part
    has a label, returns the laws it contributes from `_get_laws`, and implements three hooks that the main public
    methods `prepare` and `run` call:

    - `_check()` validates the arguments of the part, after `check` validates the label.
    - `_prepare(context)` prepares the part against its context and sets its attributes that end in `_`.
    - `_run(inputs)` sets what the part contributes: `laws_`, the laws of an element, an interaction, or a system,
      or `value_`, the value of a law.

    A part is generic in the context of `prepare` and the input of `run`, which every base of parts names: a law
    is prepared against the path of its owner, an element against the space of its system.

    A part declares which arguments of its constructor are states, parameters, or roles by annotating them with
    `Annotated` and a `StateSpec`, a `ParameterSpec`, or a `RoleSpec`, and `prepare` validates every declared spec.

    A part returns its capability tags from `get_tags()`, which reads only its declarations and its arguments, so the
    tags are known before `prepare`. Each family derives them in the private hook `_get_tags()`: an element from its
    state specs, a space and a time from their class, a law from its class, an interaction from the laws it returns
    for its arguments, and a system from its parts.

    A part whose run draws random numbers that its law kinds do not show, such as an agent rule that reads
    `system.rng` or a law that holds a draw such as `normal('noise')`, declares it with the class attribute `draws`, and
    a part whose arguments decide it, such as a random order of its agents, returns it from the private hook
    `_draws()`. Its randomness is then stochastic.

    Examples:
        >>> from skcomplex.core.components.base import BasePart
        >>> class DoublingPart(BasePart):
        ...     def __init__(self, label, factor=2):
        ...         super().__init__(label)
        ...         self.factor = factor
        ...
        ...     def _check(self):
        ...         pass
        ...
        ...     def _prepare(self, context=None):
        ...         pass
        ...
        ...     def _get_laws(self, namespace):
        ...         return []
        ...
        ...     def _run(self, value):
        ...         self.doubled_ = self.factor * value
        >>> DoublingPart('doubling').prepare().run(3).doubled_
        6
    """

    draws: ClassVar[bool] = False
    """Whether the run of the part draws random numbers that the kinds of its laws do not show."""

    def __init__(self, label: str) -> None:
        """Store the label of the part.

        Args:
            label:
                The label of the part, unique among the parts of its system.
        """
        self.label = label

    def _check_label(self) -> None:
        """Raise an error for a label that is not a string.

        Raises:
            ValidationError:
                The label is not a string.
        """
        if not isinstance(self.label, str):
            error_msg = f'The label of {type(self).__name__} must be a string, but it is {self.label!r}.'
            raise ValidationError(error_msg)

    @classmethod
    @functools.cache
    def _get_specs(cls) -> dict[str, Spec]:
        """Return the spec of every argument of the constructor annotated with one.

        The specs come in the order of the constructor. The annotations of the constructors of the bases count too, and
        a subclass wins, so a subclass that takes an argument of its base again without annotating it keeps the spec of
        the base. The specs of a class are read once, since compiling a system reads them for every part.

        Returns:
            specs:
                A mapping from argument name to its spec.
        """
        specs: dict[str, Spec] = {}
        for klass in reversed(cls.__mro__):
            init = vars(klass).get('__init__')
            if init is None:
                continue
            for name, argument in inspect.signature(init, eval_str=True).parameters.items():
                for item in getattr(argument.annotation, '__metadata__', ()):
                    if isinstance(item, StateSpec | ParameterSpec | RoleSpec):
                        specs[name] = item
        return {name: specs[name] for name in inspect.signature(cls).parameters if name in specs}

    @classmethod
    def _get_declared_laws(cls) -> tuple[type, ...] | None:
        """Return the law classes the class lists in `law_classes`.

        The result is `None` when the class writes laws it does not list. The nearest class of the method resolution
        order that lists `law_classes` or writes `_get_laws` decides: a list gives the classes, and a method without a
        list leaves them unknown.

        Returns:
            classes:
                The law classes, or `None` when the declarations do not show them.
        """
        for klass in cls.__mro__:
            declared = vars(klass)
            if 'law_classes' in declared:
                classes: tuple[type, ...] = declared['law_classes']
                return classes
            if '_get_laws' in declared:
                return None
        return None

    @staticmethod
    def _get_law_tags(classes: Sequence[type] | None) -> Tags:
        """Return the tags of laws of some classes.

        The tags are their kinds, the times they all run in, and their randomness. The laws are stochastic when one of
        them has a kind that draws, such as noise or jumps, or its class declares that it draws.

        Args:
            classes:
                The law classes, or `None` when they are unknown.

        Returns:
            tags:
                The kinds and the randomness of the laws, `None` for unknown classes, and the times they share.
        """
        if classes is None:
            return Tags(laws=None, randomness=None)
        laws = frozenset(getattr(law, 'law_tag', '') for law in classes) - {''}
        times = [
            frozenset(getattr(time, 'time_tag', '') for time in law.get_requirements().time) - {''}
            for law in classes
            if hasattr(law, 'get_requirements')
        ]
        known = [time for time in times if time]
        common = functools.reduce(frozenset.intersection, known) if known else frozenset()
        drawn = laws & STOCHASTIC_LAWS or any(getattr(law, 'draws', False) for law in classes)
        randomness = 'stochastic' if drawn else 'deterministic'
        return Tags(time=common, laws=laws, randomness=randomness)

    def _get_tags(self) -> Tags:
        """Return the tags the declarations of the part show, none by default."""
        return Tags(laws=None, randomness=None)

    def _draws(self) -> bool:
        """Return whether the part draws random numbers its law kinds do not show.

        The default is `draws`.
        """
        return self.draws

    def get_tags(self) -> Tags:
        """Return the capability tags of the part.

        They come from its declarations and its arguments alone.

        Returns:
            tags:
                The tags of the part, known before it is prepared.

        Examples:
            >>> from typing import Annotated
            >>> from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
            >>> from skcomplex.core.components.elements import Element
            >>> from skcomplex.core.specs import StateSpec
            >>> class CellElement(Element):
            ...     def __init__(self, label, alive: Annotated[int, StateSpec(FiniteEnumeratedSet([0, 1]))] = 0):
            ...         super().__init__(label)
            ...         self.alive = alive
            >>> sorted(CellElement('cell').get_tags().domains)
            ['boolean']

            A part whose rules draw random numbers declares it, so its randomness is stochastic:

            >>> class WalkerElement(CellElement):
            ...     draws = True
            >>> WalkerElement('walker').get_tags().randomness
            'stochastic'
        """
        tags = self._get_tags()
        return replace(tags, randomness='stochastic') if self._draws() else tags

    def _get_declared(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the variables the part declares from data, by name, none by default."""
        return {}

    def _get_part_specs(self) -> dict[str, Spec]:
        """Return the specs of this part.

        They are those of its class, then those it declares from data.
        """
        return {**self._get_specs(), **{name: item.spec for name, item in self._get_declared().items()}}

    @abstractmethod
    def _check(self) -> None:
        """Validate the arguments of the part.

        A named error is raised for an argument the part cannot take.
        """

    @abstractmethod
    def _prepare(self, context: ContextT | None) -> None:
        """Prepare the part against its context and set its attributes that end in `_`."""

    @abstractmethod
    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws the part contributes for a namespace of symbols.

        A law contributes itself.
        """

    def _laws_of(self, inputs: object) -> Sequence[BasePart]:
        """Return the laws of the part for a namespace of symbols.

        Args:
            inputs:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the part.

        Raises:
            ValidationError:
                The input is not a namespace.
        """
        if not isinstance(inputs, SimpleNamespace):
            error_msg = f'{type(self).__name__} returns its laws for a namespace of symbols, but got {inputs!r}.'
            raise ValidationError(error_msg)
        return self._get_laws(inputs)

    def _validated(self) -> Self:
        """Validate the label and the arguments of the part.

        Returns:
            part:
                The part itself.

        Raises:
            ValidationError:
                The label or an argument is invalid, named in the message.
        """
        self._check_label()
        for spec in [*self._get_specs().values(), *self._get_declared().values()]:
            spec._validated()
        return super()._validated()
