"""Define the abstract base of every component of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import NoneType, UnionType
from typing import Annotated, ClassVar, TypeAliasType, Union, get_args, get_origin

import numpy as np
from sklearn.base import BaseEstimator

from ..base import _BaseEstimator
from ..errors import ValidationError
from ..types import Role
from ._declared_variable_spec import _DeclaredVariableSpec
from ._parameter_spec import _ParameterSpec
from ._role_spec import RoleSpec, _Spec
from ._state_spec import _StateSpec
from ._tags import STOCHASTIC_LAWS, Tags
from ._tensor_spec import TENSOR_DIMENSIONS

MATRIX_AXES = 2
ROLE_CONTAINERS = (tuple, list, Sequence, Mapping, dict)


class BaseComponent[ContextT](_BaseEstimator[ContextT]):
    """An abstract component of a system, with a label, parameters, and tags.

    The components of a system are its elements, interactions, laws, spaces, and the system itself. Every component has
    a label and stores its parameters under their own names, so `get_params`, `set_params`, and `clone` work on it. Its
    one public method `prepare(context)` calls the hook `_prepare(context)`, which checks the arguments, prepares the
    component against its context, and sets its attributes that end in `_`. The hook starts with
    `super()._prepare(context)`, so this base checks the label and the declared specs first.

    The context of `prepare` depends on the family: a law is prepared against the path of its owner, an element against
    the space of its system, and an interaction against the namespace of the elements its roles bind, from which it
    sets `laws_`. A component declares which arguments of its constructor are states, parameters, or roles by
    annotating them with `StateSpec`, `ParameterSpec`, or `RoleSpec` inside `Annotated`.

    A component returns its capability tags from `get_tags`, which reads only its declarations and arguments, so the
    tags are known before `prepare`. Each family derives them in `_get_tags`. A component whose rules draw random
    numbers that its laws do not show, such as an agent rule that reads `system.rng`, sets the class attribute `draws`,
    or returns it from `_get_draws` when its arguments decide it. A component whose draws are all uniform, normal, or
    relaxed Bernoulli or categorical draws sets `reparameterized`, so its runs on JAX are differentiable.

    Unlike `BaseConfig`, the base of the engine classes, it has no `run`: a system prepares its components and the
    engine runs what they declare. Choose it when a new family of components is needed, and choose the base of an
    existing family, such as `BaseElement`, otherwise.

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
        ...         self.doubled_ = 2.0 * self.factor
        >>> DoublingComponent('doubling', factor=3).prepare().doubled_
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

    @staticmethod
    def _get_shape_of(value: object) -> tuple[bool, int | None, str | None]:
        """Return whether a value is a vector, its size, and its tensor symmetry.

        The size is the number of components of a vector or of rows of a matrix. A number is no vector, a list or an
        array of `n` numbers is a vector of `n` components, and a square matrix of 2 or 3 rows is a full tensor, which
        laws read as a Sage matrix. Any other array is a vector of one component per dimension of the space.
        """
        shape = np.shape(np.asarray(value, dtype=object))
        if not shape:
            return False, None, None
        if len(shape) == 1:
            return True, shape[0], None
        if len(shape) == MATRIX_AXES and shape[0] == shape[1] and shape[0] in TENSOR_DIMENSIONS:
            return False, shape[0], 'full'
        return True, None, None

    @staticmethod
    def _get_matrix_entries(value: object, tensor: str | None) -> object:
        """Return a matrix value as its entries, row after row.

        The matrices sit on the last two axes, so a group gives one matrix per member. Any other value is returned as it
        is.
        """
        if tensor is None:
            return value
        array = np.asarray(value)
        return array.reshape(*array.shape[:-2], array.shape[-2] * array.shape[-1])

    @staticmethod
    def _get_annotated(annotation: object) -> object:
        """Return the `Annotated` form an annotation stands for.

        A short annotation such as `State[float]` stands for the `Annotated` form it aliases, `Role` stands for a role
        of one element, and any other annotation stands for itself. A short annotation inside `Annotated`, such as
        `Annotated[Role, 'boolean']`, stands for the form it aliases with the further metadata added. A role that may
        be `None`, such as `Role | None`, stands for an optional role.
        """
        others = [item for item in get_args(annotation) if item is not NoneType]
        if get_origin(annotation) in (UnionType, Union) and len(others) == 1 < len(get_args(annotation)):
            inner = BaseComponent._get_annotated(others[0])
            if get_origin(inner) is not Annotated:
                return annotation
            base, *metadata = get_args(inner)
            optional = [replace(item, optional=True) if isinstance(item, RoleSpec) else item for item in metadata]
            return Annotated[base | None, *optional]
        if get_origin(annotation) is Annotated:
            inner, *extra = get_args(annotation)
            aliased = BaseComponent._get_annotated(inner)
            if aliased is not inner and get_origin(aliased) is Annotated:
                base, *metadata = get_args(aliased)
                return Annotated[base, *metadata, *extra]
            return annotation
        if annotation is Role:
            return Annotated[str, RoleSpec()]
        alias = get_origin(annotation) or annotation
        return getattr(alias, '__value__', annotation) if isinstance(alias, TypeAliasType) else annotation

    @classmethod
    @functools.cache
    def _get_specs(cls) -> dict[str, _Spec]:
        """Return the spec of every argument of the constructor annotated with one.

        The specs come in the order of the constructor. The annotations of the constructors of the bases count too, and
        a subclass wins, so a subclass that takes an argument of its base again without annotating it keeps the spec of
        the base. The specs of a class are read once, since compiling a system reads them for every component. The
        words after a role, such as `Annotated[Role, 'boolean']`, are the domains the role accepts.

        Returns:
            specs:
                A mapping from argument name to its spec.
        """
        specs: dict[str, _Spec] = {}
        for klass in reversed(cls.__mro__):
            init = vars(klass).get('__init__')
            if init is None:
                continue
            for name, argument in inspect.signature(init, eval_str=True).parameters.items():
                metadata = getattr(cls._get_annotated(argument.annotation), '__metadata__', ())
                for item in metadata:
                    if isinstance(item, _StateSpec | _ParameterSpec | RoleSpec):
                        specs[name] = item
                hinted = None if metadata else cls._get_role_of(argument.annotation)
                if hinted is not None:
                    specs[name] = hinted
                domains = tuple(item for item in metadata if isinstance(item, str))
                role = specs.get(name)
                if domains and isinstance(role, RoleSpec) and role in metadata:
                    specs[name] = replace(role, domains=(*role.domains, *domains))
        return {name: specs[name] for name in inspect.signature(cls).parameters if name in specs}

    @staticmethod
    def _get_role_of(annotation: object) -> RoleSpec | None:
        """Return the role a type hint states, or `None` for a hint of no role.

        `Role` binds one element, `Role | None` one element or none, and a container of roles binds several, such as
        `tuple[Role, ...]` for the reactants of a reaction or a mapping keyed by roles for their amounts.
        """
        annotated = BaseComponent._get_annotated(annotation)
        if get_origin(annotated) is Annotated:
            return next((item for item in get_args(annotated)[1:] if isinstance(item, RoleSpec)), None)
        origin, args = get_origin(annotated), get_args(annotated)
        if origin is Union or origin is UnionType:
            found = [role for arg in args if (role := BaseComponent._get_role_of(arg)) is not None]
            if not found:
                return None
            role = next((role for role in found if role.variadic), found[0])
            return replace(role, optional=NoneType in args)
        if origin in ROLE_CONTAINERS and any(BaseComponent._get_role_of(arg) is not None for arg in args):
            return RoleSpec(variadic=True)
        return None

    @classmethod
    def _get_declared_laws(cls) -> tuple[type, ...] | None:
        """Return the law classes the class lists in `law_classes`.

        The nearest class of the method resolution order that lists `law_classes` decides, and the result is `None`
        when no class lists them.

        Returns:
            classes:
                The law classes, or `None` when the declarations do not show them.
        """
        for klass in cls.__mro__:
            if 'law_classes' in vars(klass):
                classes: tuple[type, ...] = vars(klass)['law_classes']
                return classes
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

    @staticmethod
    def _is_equal(old: object, new: object) -> bool:
        """Return whether two values of a model parameter are equal by value.

        Estimators are equal when they have the same class and equal parameters, sequences and mappings when their
        items are equal, and arrays when they have the same shape and entries. A value that cannot be compared is
        taken as changed.

        Args:
            old:
                The value the components were last built from.

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
            equal = old.keys() == new.keys() and all(BaseComponent._is_equal(old[key], new[key]) for key in old)
        elif isinstance(old, list | tuple) and isinstance(new, list | tuple):
            equal = len(old) == len(new) and all(BaseComponent._is_equal(*pair) for pair in zip(old, new, strict=True))
        elif isinstance(old, np.ndarray) and isinstance(new, np.ndarray):
            equal = old.shape == new.shape and bool(np.array_equal(old, new))
        else:
            try:
                equal = bool(old == new)
            except (TypeError, ValueError):
                equal = False
        return equal

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
            A component whose rules draw random numbers declares it, so its randomness is stochastic:

            >>> from skcomplex.core.components.elements import Element
            >>> class WalkerElement(Element):
            ...     draws = True
            >>> WalkerElement('walker').get_tags().randomness
            'stochastic'
        """
        tags = self._get_tags()
        tags = replace(tags, randomness='stochastic') if self._get_draws() else tags
        return replace(tags, reparameterized=True) if self.reparameterized else tags

    def _get_declared(self) -> Mapping[str, _DeclaredVariableSpec]:
        """Return the variables the component declares from data, by name, none by default."""
        return {}

    def _get_component_specs(self) -> dict[str, _Spec]:
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

    def _prepare_on(self, space: ContextT | None) -> None:
        """Prepare the component on the space of its system, checking its arguments."""
        self.prepare(space)
