"""Hold the helpers of the laws: the namespace of the symbols they are written with, and the values they hold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from collections.abc import Collection, Mapping
from typing import Any

from ..base import BaseLaw, SymbolicVector
from ._laws import Call


class Symbols:
    """Namespace of the symbols a law can use.

    An element's namespace holds the symbols of its states and parameters. An interaction's namespace holds the
    symbols of its parameters, one element namespace per role, and the time symbol `t`.

    Args:
        **symbols:
            The symbols and nested namespaces by name.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import Symbols
        >>> s = Symbols(a=symbol('a'), prey=Symbols(n=symbol('n')))
        >>> s.a * s.prey.n
        a*n
    """

    def __init__(self, **symbols: Any) -> None:  # noqa: ANN401
        self.__dict__.update(symbols)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Raise an error that lists the available names."""
        available = ', '.join(sorted(self.__dict__))
        error_msg = f'No symbol named {name!r}. Available names are: {available}.'
        raise AttributeError(error_msg)

    def __repr__(self) -> str:
        """Return the names held by the namespace."""
        return f'Symbols({", ".join(sorted(self.__dict__))})'


def values_of(law: BaseLaw, *, include: Collection[str] | None = None, exclude: Collection[str] = ()) -> list[object]:
    """Return the values of the fields of a law, flattened in field order.

    A called function gives its arguments, a mapping its keys and values in pairs, and a vector, a list, or a tuple its
    items, at any depth. Every other value is returned as it is.

    Args:
        law:
            The law.

        include:
            The names of the fields to read, or `None` for every field.

        exclude:
            The names of the fields to leave out.

    Returns:
        values:
            The flattened values.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import JumpLaw, NoiseLaw, values_of
        >>> s, i = symbol('s'), symbol('i')
        >>> values_of(JumpLaw(0.1 * s * i, {s: -1, i: 1}))
        [0.1*i*s, s, -1, i, 1, None, None, None]
        >>> values_of(NoiseLaw(s, 0.2 * s, process='market'), include=('value', 'process'))
        [0.2*s, 'market']
    """
    names = [
        item.name
        for item in dataclasses.fields(law)
        if (include is None or item.name in include) and item.name not in exclude
    ]
    stack: list[object] = [getattr(law, name) for name in reversed(names)]
    found: list[object] = []
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            stack += reversed([item for pair in value.items() for item in pair])
        elif isinstance(value, SymbolicVector | list | tuple):
            stack += reversed(list(value))
        elif isinstance(value, Call):
            stack += reversed(value.arguments)
        else:
            found.append(value)
    return found
