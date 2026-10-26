"""Define the base of elements, the parts of a system that hold states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar, Self, cast

from ._types import (
    Variable,
    _argument_names,
    _caller_module,
    _check_declared,
    _declarations,
    _declare,
    _define_type,
    _get_default,
    _get_params,
)
from ._utils import BaseEstimator


class BaseComponent(BaseEstimator):
    """Base of every part of a system: elements and groups.

    A component type declares its variables and roles as class attributes made by `state`, `parameter`, and `role`.
    The constructor takes the label and the values of the declared names, which include the companions of the states
    of higher order or with lags, and stores every value under its own name, the default for a name not given. A type
    with constructor settings that are not declared, such as a radius or a layer, takes them in an `__init__` of its own
    that stores them and passes the label and the remaining values on.

    `check()` rejects a value the type does not declare and a declared variable without a default that is not given,
    `prepare()` sets `declarations_`, and `run(s)` returns the laws of the component for a namespace of symbols, from
    the hook `laws` that a type overrides. A type returns its auxiliaries, invariants, and objectives from the hook
    `quantities`, which takes the same namespace. Validation checks every component of a system.

    Args:
        label:
            The label of the component, unique among the children of its system.

        **values:
            The values of the declared states, parameters, and roles, by name.

    Attributes:
        declarations_:
            The declarations of the states, parameters, and roles of the type, with the derived companions, by name in
            declaration order.

    Examples:
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.components import ElementComponent
        >>> class PendulumComponent(ElementComponent):
        ...     theta: float = state(0.0, order=2)
        >>> bob = PendulumComponent('bob', theta=0.1, theta_d1=0.5).prepare()
        >>> list(bob.declarations_), bob.theta_d1, hasattr(PendulumComponent, 'theta_d1')
        (['theta', 'theta_d1'], 0.5, False)
    """

    reserved_names: ClassVar[frozenset[str]] = frozenset({'label'})

    def __init__(self, label: str, **values: Any) -> None:  # noqa: ANN401
        _declare(self, label, values)

    @classmethod
    def _define(cls, name: str, **declarations: Any) -> type[Self]:  # noqa: ANN401
        """Return a subclass of the type built from declarations, in the module of the code that called `define`."""
        return cast('type[Self]', _define_type(cls, name, _caller_module(), **declarations))

    @classmethod
    def _declarations(cls) -> dict[str, Variable]:
        """Return the declarations of the type, after rejecting the names it reserves."""
        return _declarations(cls)

    def _check(self) -> None:
        """Raise an error for a value the type does not declare, or a declared variable without a default not given.

        Raises:
            ArgumentError:
                A value names a variable the type does not declare, or a declared variable without a default is not
                given.

            LabelError:
                The type declares a variable whose name is reserved.
        """
        _check_declared(self, BaseComponent)

    def _prepare(self) -> None:
        """Set `declarations_`, the declarations of the type."""
        self.declarations_ = type(self)._declarations()

    def _run(self, s: Any) -> list[Any]:  # noqa: ANN401
        """Return the laws of the component for a namespace of symbols, from the hook `laws`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return self.laws(s)

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
            >>> from skcomplex.models.biology.ecology import PopulationComponent
            >>> PopulationComponent.get_default('n')
            1.0
        """
        return _get_default(cls, BaseComponent, name)

    @classmethod
    def get_argument_names(cls) -> list[str]:
        """Return the names the constructor takes besides the label: its settings and the declared names.

        Returns:
            names:
                The names, settings first.

        Examples:
            >>> from skcomplex.formalisms.rewriting import HostGraphComponent
            >>> HostGraphComponent.get_argument_names()
            ['max_size', 'max_matches', 'graph']
        """
        return _argument_names(cls, BaseComponent)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the label, the settings, and the values of every declared name.

        Args:
            deep:
                Whether to add the parameters of nested estimators as `<name>__<parameter>`.

        Returns:
            params:
                A mapping from argument name to value, in sorted order.

        Examples:
            >>> from skcomplex.models.biology.ecology import PopulationComponent
            >>> PopulationComponent('rabbits', n=40).get_params()['n']
            40
        """
        return _get_params(self, BaseComponent, deep=deep)

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

    def quantities(self, s: Any) -> list[Any]:  # noqa: ANN401
        """Return the auxiliaries, invariants, and objectives of the component, none by default.

        Args:
            s:
                The namespace of the symbols the quantities can use.

        Returns:
            quantities:
                The quantities of the component.
        """
        return []
