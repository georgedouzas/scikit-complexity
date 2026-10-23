"""Define the base of every component: elements, groups, interactions, and systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import inspect
from typing import Any, ClassVar, cast

from sklearn.base import BaseEstimator

from ..base import ArgumentError, LabelError, Variable, declaration_of, declared_fields, default_of

UNSET = (inspect.Parameter.empty, dataclasses.MISSING)


def _is_required(field: Any) -> bool:  # noqa: ANN401
    """Check whether a declared field has no default."""
    return field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING


class Component(BaseEstimator):
    """Base of every part of a system: elements, groups, interactions, and systems.

    A component type declares its variables and roles as class attributes made by `state`, `parameter`, and `role`.
    The constructor takes the label and the values of the declared names, which include the companions of the states
    of higher order or with lags, and stores every value under its own name, the default for a name not given. A type
    with constructor settings that are not declared, such as a radius or a layer, takes them in an `__init__` of its own
    that stores them and passes the label and the remaining values on.

    Args:
        label:
            The label of the component, unique among the children of its system.

        **values:
            The values of the declared states, parameters, and roles, by name.

    Raises:
        ArgumentError:
            A value names a variable the type does not declare, or a declared variable without a default is not given.

        LabelError:
            The type declares a variable whose name is reserved.

    Examples:
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.components import Element
        >>> class Pendulum(Element):
        ...     theta: float = state(0.0, order=2)
        >>> bob = Pendulum('bob', theta=0.1, theta_d1=0.5)
        >>> bob.theta_d1, hasattr(Pendulum, 'theta_d1')
        (0.5, False)
    """

    reserved_names: ClassVar[frozenset[str]] = frozenset({'label'})

    def __init__(self, label: str, **values: Any) -> None:  # noqa: ANN401
        type(self).declarations()
        fields = type(self)._fields()
        unknown = sorted(set(values) - set(fields))
        if unknown:
            declared = ', '.join(fields) or 'none'
            error_msg = (
                f'{type(self).__name__} got {", ".join(map(repr, unknown))}, which it does not declare. Its declared '
                f'variables are {declared}.'
            )
            raise ArgumentError(error_msg)
        missing = [name for name, field in fields.items() if name not in values and _is_required(field)]
        if missing:
            error_msg = f'{type(self).__name__} needs the arguments {", ".join(map(repr, missing))}.'
            raise ArgumentError(error_msg)
        self.label = label
        for name, field in fields.items():
            setattr(self, name, values[name] if name in values else default_of(field))

    @classmethod
    def _fields(cls) -> dict[str, Any]:
        """Return the declared fields of the type, after rejecting reserved names."""
        fields = declared_fields(cls)
        reserved = sorted(set(fields) & cls.reserved_names)
        if reserved:
            error_msg = (
                f'{cls.__name__} declares the variable {reserved[0]!r}, whose name is reserved. Rename it, since '
                f'{", ".join(sorted(cls.reserved_names))} are reserved names.'
            )
            raise LabelError(error_msg)
        return fields

    @classmethod
    def _settings(cls) -> dict[str, inspect.Parameter]:
        """Return the keyword arguments of the constructors along the method resolution order, without `**values`."""
        found: dict[str, inspect.Parameter] = {}
        for klass in cls.__mro__:
            init = vars(klass).get('__init__')
            if init is None:
                continue
            parameters = list(inspect.signature(init).parameters.values())[1:]
            for parameter in parameters:
                if parameter.kind not in {parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL}:
                    found.setdefault(parameter.name, parameter)
            if klass is Component or not any(item.kind is item.VAR_KEYWORD for item in parameters):
                break
        return found

    @classmethod
    def declarations(cls) -> dict[str, Variable]:
        """Return the declarations of the states, parameters, and roles of the type, with the derived companions.

        Returns:
            declarations:
                A mapping from name to declaration, in declaration order.

        Raises:
            LabelError:
                The type declares a variable whose name is reserved.

        Examples:
            >>> from skcomplex.core.base import state
            >>> from skcomplex.core.components import Element
            >>> Lagged = Element.define('Lagged', states={'x': state(0.0, lags=2)})
            >>> list(Lagged.declarations())
            ['x', 'x_lag1', 'x_lag2']
        """
        return {name: cast('Variable', declaration_of(field)) for name, field in cls._fields().items()}

    @classmethod
    def get_default(cls, name: str) -> Any:  # noqa: ANN401
        """Return the default value of a declared name or a setting, a fresh copy for a mutable declared default.

        Args:
            name:
                The declared name or the setting.

        Returns:
            default:
                The default value, `dataclasses.MISSING` for a declared name without one.

        Raises:
            KeyError:
                The type neither declares the name nor takes it as a setting with a default.

        Examples:
            >>> from skcomplex.models.biology.ecology import Population
            >>> Population.get_default('n')
            1.0
        """
        fields = cls._fields()
        if name in fields:
            return default_of(fields[name])
        setting = cls._settings().get(name)
        if setting is None or setting.default is inspect.Parameter.empty:
            error_msg = f'{cls.__name__} has no default for {name!r}.'
            raise KeyError(error_msg)
        return setting.default

    @classmethod
    def get_argument_names(cls) -> list[str]:
        """Return the names the constructor takes besides the label: its settings and the declared names.

        Returns:
            names:
                The names, settings first.

        Examples:
            >>> from skcomplex.core.components import Wiener
            >>> Wiener.get_argument_names()
            ['dimension', 'covariance']
        """
        return [*(name for name in cls._settings() if name != 'label'), *cls._fields()]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the label, the settings, and the values of every declared name.

        Args:
            deep:
                Whether to add the parameters of nested estimators as `<name>__<parameter>`.

        Returns:
            params:
                A mapping from argument name to value, in sorted order.

        Examples:
            >>> from skcomplex.models.biology.ecology import Population
            >>> Population('rabbits', n=40).get_params()['n']
            40
        """
        params: dict[str, Any] = {}
        for name in sorted(['label', *self.get_argument_names()]):
            value = getattr(self, name)
            if deep and hasattr(value, 'get_params') and not isinstance(value, type):
                params.update({f'{name}__{key}': item for key, item in value.get_params().items()})
            params[name] = value
        return params

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the type, the label, and the arguments that differ from their defaults."""
        defaults = {name: item.default for name, item in type(self)._settings().items()}
        defaults.update({name: default_of(field) for name, field in type(self)._fields().items()})
        shown = [
            f'{name}={value!r}'
            for name, value in self.get_params(deep=False).items()
            if name not in defaults
            or any(defaults[name] is item for item in UNSET)
            or repr(value) != repr(defaults[name])
        ]
        text = f'{type(self).__name__}({", ".join(shown)})'
        return text if len(text) <= N_CHAR_MAX else f'{text[: N_CHAR_MAX - 4]}...)'

    def laws(self, s: Any) -> list[Any]:  # noqa: ANN401
        """Return the laws of the component, none by default.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
