"""Define the abstract base of every component of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import functools
import inspect
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

from ..base import BaseConfig
from ..errors import ValidationError
from ..specs import STOCHASTIC_LAWS, DeclaredVariableSpec, ParameterSpec, RoleSpec, Spec, StateSpec, Tags


class BaseComponent[ContextT, InputT](BaseConfig[ContextT, InputT]):
    """An abstract component of a system, with a label, laws, and capability tags.

    The components of a system are its elements, interactions, laws, spaces, times, and the system itself. Every
    component has a label, returns its laws from `_get_laws`, and implements two hooks that the main public methods
    `prepare` and `run` call:

    - `_prepare(context)` checks the arguments, prepares the component against its context, and sets its attributes that
      end in `_`. It starts with `super()._prepare(context)`, so this base checks the label and the declared specs
      first.
    - `_run(inputs)` sets what the component contributes: `laws_`, the laws of an element, an interaction, or a system,
      or `value_`, the value of a law.

    The context of `prepare` depends on the family: a law is prepared against the path of its owner, an element against
    the space of its system. A component declares which arguments of its constructor are states, parameters, or roles by
    annotating them with `StateSpec`, `ParameterSpec`, or `RoleSpec` inside `Annotated`.

    A component returns its capability tags from `get_tags`, which reads only its declarations and arguments, so the
    tags are known before `prepare`. Each family derives them in `_get_tags`. A component whose run draws random numbers
    that its laws do not show, such as an agent rule that reads `system.rng`, sets the class attribute `draws`, or
    returns it from `_draws` when its arguments decide it. A component whose draws are all uniform, normal, or relaxed
    Bernoulli or categorical draws sets `reparameterized`, so its runs on JAX are differentiable.

    Unlike `BaseConfig`, which only stores and checks arguments, it adds laws, a context for `prepare`, and tags. Choose
    it when a new family of components is needed, and choose the base of an existing family, such as `BaseElement`,
    otherwise.

    It appears in every field through its families, such as the populations, infections, and contact networks of an
    epidemic, or the accounts, payments, and markets of an economy.

    Examples:
        >>> from skcomplex.core.components.base import BaseComponent
        >>> class DoublingComponent(BaseComponent):
        ...     def __init__(self, label, factor=2):
        ...         super().__init__(label)
        ...         self.factor = factor
        ...
        ...     def _prepare(self, context=None):
        ...         super()._prepare(context)
        ...         self.factor_ = float(self.factor)
        ...
        ...     def _get_laws(self, namespace):
        ...         return []
        ...
        ...     def _run(self, value):
        ...         self.doubled_ = self.factor_ * value
        >>> DoublingComponent('doubling').prepare().run(3).doubled_
        6.0
    """

    draws: ClassVar[bool] = False
    """Whether the run of the component draws random numbers that the kinds of its laws do not show."""

    reparameterized: ClassVar[bool] = False
    """Whether every draw of the component is a uniform, normal, or relaxed draw that JAX differentiates."""

    def __init__(self, label: str) -> None:
        """Store the label of the component.

        Args:
            label:
                The label of the component, unique among the components of its system.
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
        the base. The specs of a class are read once, since compiling a system reads them for every component.

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
        """Return the tags the declarations of the component show, none by default."""
        return Tags(laws=None, randomness=None)

    def _get_draws(self) -> bool:
        """Return whether the component draws random numbers its law kinds do not show.

        The default is `draws`.
        """
        return self.draws

    def get_tags(self) -> Tags:
        """Return the capability tags of the component.

        They come from its declarations and its arguments alone.

        Returns:
            tags:
                The tags of the component, known before it is prepared.

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

            A component whose rules draw random numbers declares it, so its randomness is stochastic:

            >>> class WalkerElement(CellElement):
            ...     draws = True
            >>> WalkerElement('walker').get_tags().randomness
            'stochastic'
        """
        tags = self._get_tags()
        tags = replace(tags, randomness='stochastic') if self._get_draws() else tags
        return replace(tags, reparameterized=True) if self.reparameterized else tags

    def _get_declared(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the variables the component declares from data, by name, none by default."""
        return {}

    def _get_component_specs(self) -> dict[str, Spec]:
        """Return the specs of this component.

        They are those of its class, then those it declares from data.
        """
        return {**self._get_specs(), **{name: item.spec for name, item in self._get_declared().items()}}

    def _prepare(self, context: ContextT | None) -> None:
        """Check the label and the declared specs of the component.

        A child starts its own `_prepare` with `super()._prepare(context)`, so these checks come first, and then
        checks its own arguments before it sets any attribute that ends in `_`.

        Args:
            context:
                The context of the component, which this base does not read.

        Raises:
            ValidationError:
                The label or a declared spec is invalid, named in the message.
        """
        self._check_label()
        for spec in [*self._get_specs().values(), *self._get_declared().values()]:
            spec._check()

    @abstractmethod
    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the laws the component contributes for a namespace of symbols.

        A law contributes itself.
        """

    def _get_laws_of(self, inputs: object) -> Sequence[BaseComponent]:
        """Return the laws of the component for a namespace of symbols.

        Args:
            inputs:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.

        Raises:
            ValidationError:
                The input is not a namespace.
        """
        if not isinstance(inputs, SimpleNamespace):
            error_msg = f'{type(self).__name__} returns its laws for a namespace of symbols, but got {inputs!r}.'
            raise ValidationError(error_msg)
        return self._get_laws(inputs)
