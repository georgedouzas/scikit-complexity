"""Define the registry of the model catalog."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Hashable
from typing import TypeVar

Entry = TypeVar('Entry')


class Registry[Entry]:
    """Mapping from keys to implementations, open to extensions.

    Args:
        name:
            The name of what the registry holds, used in error messages.

    Examples:
        >>> from skcomplex.engine.utils import Registry
        >>> registry = Registry[type]('visuals')
        >>> registry.register('number', float)
        >>> registry.resolve('number')
        <class 'float'>
    """

    def __init__(self, name: str) -> None:
        """Store the name of the registry and start it empty."""
        self.name = name
        self._entries: dict[Hashable, Entry] = {}
        self._loaders: list[Callable[[Hashable], None]] = []

    @property
    def keys(self) -> list[Hashable]:
        """Registered keys, in registration order."""
        return list(self._entries)

    def register(self, key: Hashable, entry: Entry) -> None:
        """Register an implementation under a key, replacing any previous one.

        Args:
            key:
                The key of the implementation.

            entry:
                The implementation.
        """
        self._entries[key] = entry

    def register_loader(self, loader: Callable[[Hashable], None]) -> None:
        """Register a function that registers the implementation of an unknown key, such as by importing its module.

        Args:
            loader:
                A function of a key, called when the key has no registered implementation, that may register one.
        """
        if loader not in self._loaders:
            self._loaders.append(loader)

    def resolve(self, key: Hashable) -> Entry:
        """Return the implementation registered under a key.

        Args:
            key:
                The key of the implementation.

        Returns:
            entry:
                The implementation.

        Raises:
            KeyError:
                The key has no registered implementation, and no loader registers one.
        """
        for loader in self._loaders:
            if key in self._entries:
                break
            loader(key)
        if key not in self._entries:
            available = ', '.join(sorted(map(str, self._entries)))
            error_msg = f'No {self.name} registered under {key!r}. Registered keys are: {available}.'
            raise KeyError(error_msg)
        return self._entries[key]
