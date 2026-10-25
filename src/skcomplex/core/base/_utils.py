"""Hold the helpers the bases share: registries, visual declarations, and angle wrapping."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt


def wrap_angle(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return angles wrapped into one turn centred on zero, the half-open interval `(-pi, pi]`.

    Args:
        values:
            An angle or an array of angles in radians.

    Returns:
        angles:
            The angles in `(-pi, pi]`, with `-pi` moved to `pi` and angles already inside left exactly as they are.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.base import wrap_angle
        >>> wrap_angle(np.array([3 * np.pi / 2, -np.pi, 0.5])).round(4).tolist()
        [-1.5708, 3.1416, 0.5]
    """
    angles = np.asarray(values, dtype=float)
    inside = (angles > -np.pi) & (angles <= np.pi)
    return np.where(inside, angles, np.pi - np.mod(np.pi - angles, 2 * np.pi))


Entry = TypeVar('Entry')


class Registry[Entry]:
    """Mapping from keys to implementations, open to extensions.

    Args:
        name:
            The name of what the registry holds, used in error messages.

    Examples:
        >>> from skcomplex.core.base import Registry
        >>> registry = Registry[type]('visuals')
        >>> registry.register('number', float)
        >>> registry.resolve('number')
        <class 'float'>
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[Hashable, Entry] = {}
        self._loaders: list[Callable[[Hashable], None]] = []

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


@dataclass(frozen=True)
class VisualSpec:
    """Name of a registered visual and the options it is built with.

    Args:
        kind:
            The name of a visual registered in the visualization layer.

        options:
            The keyword arguments of the visual.

    Examples:
        >>> from skcomplex.core.base import VisualSpec
        >>> VisualSpec('points', {'position': 'x'}).kind
        'points'
    """

    kind: str = 'scalars'
    options: dict[str, Any] = field(default_factory=dict)
