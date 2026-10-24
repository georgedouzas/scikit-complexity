"""Define open vocabularies of named terms that extensions add to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, NoReturn, Self, cast

from ._errors import RegistryError

TERMS: dict[type, dict[str, Any]] = {}


class Term:
    """Named value of an open vocabulary, created once per name and compared by identity.

    Creating a term with a name already registered in its vocabulary returns the registered term, so every term of one
    name is the same object. Each subclass is a vocabulary of its own, and an extension adds a term by creating it.

    Args:
        name:
            The name of the term in upper case, such as `'ODE'`.

        value:
            The text of the term, such as `'ordinary differential equations'`.

    Raises:
        RegistryError:
            The name is registered in the vocabulary with another value.

    Examples:
        >>> from skcomplex.core.base import Form
        >>> Form('ODE', 'ordinary differential equations') is Form.ODE
        True
        >>> str(Form.ODE)
        'ordinary differential equations'
    """

    __slots__ = ('name', 'value')

    name: str
    value: str

    def __new__(cls, name: str, value: str) -> Self:
        """Return the registered term of the name, or register a new one."""
        members: dict[str, Self] = TERMS.setdefault(cls, {})
        registered = members.get(name)
        if registered is not None and registered.value != value:
            error_msg = (
                f'{cls.__name__} {name} is registered with the value {registered.value!r}, not {value!r}. Give the '
                'registered value, or choose another name.'
            )
            raise RegistryError(error_msg)
        if registered is not None:
            return registered
        term = super().__new__(cls)
        object.__setattr__(term, 'name', name)
        object.__setattr__(term, 'value', value)
        members[name] = term
        return term

    def __setattr__(self, name: str, value: object) -> NoReturn:
        """Reject any change, since a term is frozen."""
        error_msg = f'{self!r} is frozen, so its attribute {name!r} cannot be set.'
        raise AttributeError(error_msg)

    def __reduce__(self) -> tuple[type[Self], tuple[str, str]]:
        """Return the constructor call that restores the registered term."""
        return type(self), (self.name, self.value)

    def __copy__(self) -> Self:
        """Return the term itself, since a term is unique."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        """Return the term itself, since a term is unique."""
        return self

    def __str__(self) -> str:
        """Return the value of the term."""
        return self.value

    def __repr__(self) -> str:
        """Return the vocabulary and the name of the term."""
        return f'{type(self).__name__}.{self.name}'

    @classmethod
    def registered(cls) -> tuple[Self, ...]:
        """Return the terms of the vocabulary in registration order.

        Returns:
            terms:
                The registered terms.

        Examples:
            >>> from skcomplex.core.base import SpaceKind
            >>> SpaceKind.LATTICE in SpaceKind.registered()
            True
        """
        return tuple(cast('dict[str, Self]', TERMS.get(cls, {})).values())
