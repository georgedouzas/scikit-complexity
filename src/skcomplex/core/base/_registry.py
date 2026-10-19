"""Register named implementations that extensions can add to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Hashable, Iterator
from typing import TypeVar

Entry = TypeVar('Entry')


class Registry[Entry]:
    """Mapping from keys to implementations, open to extensions.

    Args:
        name:
            The name of what the registry holds, used in error messages.

    Examples:
        >>> from skcomplex.core import Registry
        >>> registry = Registry[type]('visuals')
        >>> registry.register('number', float)
        >>> registry.resolve('number')
        <class 'float'>
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[Hashable, Entry] = {}

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over the registered keys in registration order."""
        return iter(self._entries)

    def register(self, key: Hashable, entry: Entry) -> None:
        """Register an implementation under a key, replacing any previous one.

        Args:
            key:
                The key of the implementation.

            entry:
                The implementation.
        """
        self._entries[key] = entry

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
                The key has no registered implementation.
        """
        if key not in self._entries:
            available = ', '.join(sorted(map(str, self._entries)))
            error_msg = f'No {self.name} registered under {key!r}. Registered keys are: {available}.'
            raise KeyError(error_msg)
        return self._entries[key]
