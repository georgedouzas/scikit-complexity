"""Define the namespace of symbols that laws are written with."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any


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
