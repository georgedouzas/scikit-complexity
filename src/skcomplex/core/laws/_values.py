"""Return the values a law holds, flattened into scalars and expressions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from collections.abc import Collection, Mapping

from ..base import SymbolicVector
from ._law import Law
from ._laws import Call


def values_of(law: Law, *, include: Collection[str] | None = None, exclude: Collection[str] = ()) -> list[object]:
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
        >>> from skcomplex.core.laws import Jump, Noise, values_of
        >>> s, i = symbol('s'), symbol('i')
        >>> values_of(Jump(0.1 * s * i, {s: -1, i: 1}))
        [0.1*i*s, s, -1, i, 1, None, None, None]
        >>> values_of(Noise(s, 0.2 * s, process='market'), include=('value', 'process'))
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
