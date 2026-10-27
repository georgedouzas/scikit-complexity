"""Define the shared helpers: the estimator base of configurable objects and registries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Hashable
from typing import Any, Self, TypeVar

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator as SklearnEstimator


class BaseEstimator(SklearnEstimator):
    """Base of every configurable object: it stores its arguments and speaks three verbs.

    The constructor of a child stores every argument unmodified under its own name. `check` validates the arguments,
    `prepare` prepares the object against what it works on and sets every attribute that ends in `_`, and `run` performs
    the object's one action. Each verb calls a private hook that a child overrides, and the hooks of this base do
    nothing, so every object answers every verb. A last call `run(..., final=True)` also releases what `prepare`
    acquired.

    Examples:
        >>> from skcomplex.engine.utils import BaseEstimator
        >>> class DoublingEstimator(BaseEstimator):
        ...     def __init__(self, factor=2):
        ...         self.factor = factor
        ...
        ...     def _run(self, value):
        ...         return self.factor * value
        >>> DoublingEstimator().prepare().run(3)
        6
    """

    def _check(self) -> None:
        """Raise an error for an argument the object cannot take; accept every argument by default."""

    def _prepare(self, *context: Any, **settings: Any) -> None:  # noqa: ANN401
        """Prepare the object against its context and set its prepared attributes; derive nothing by default."""

    def _run(self, *inputs: Any, **settings: Any) -> Any:  # noqa: ANN401
        """Perform the action of the object; do nothing by default."""
        return None

    def _release(self) -> None:
        """Release what preparing acquired, such as a viewer or a process; release nothing by default."""

    def check(self) -> Self:
        """Validate the arguments of the object.

        Returns:
            self:
                The object.

        Raises:
            ValidationError:
                An argument is invalid, named in the message.
        """
        self._check()
        return self

    def prepare(self, *context: Any, **settings: Any) -> Self:  # noqa: ANN401
        """Check the arguments and prepare the object against its context, setting every attribute that ends in `_`.

        Args:
            *context:
                What the object works on, such as the problem a solver advances.

            **settings:
                Settings of the preparation, such as the end time of a run.

        Returns:
            self:
                The prepared object.
        """
        self.check()
        self._prepare(*context, **settings)
        return self

    def run(self, *inputs: Any, final: bool = False, **settings: Any) -> Any:  # noqa: ANN401
        """Perform the one action of the object on its inputs.

        Args:
            *inputs:
                The inputs of the action, such as the state a solver advances.

            final:
                Whether this is the last call, after which the object releases what preparing acquired.

            **settings:
                Settings of the action.

        Returns:
            result:
                The result of the action.
        """
        result = self._run(*inputs, **settings)
        if final:
            self._release()
        return result


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
        >>> from skcomplex.engine.utils import wrap_angle
        >>> wrap_angle(np.array([3 * np.pi / 2, -np.pi, 0.5])).round(4).tolist()
        [-1.5708, 3.1416, 0.5]
    """
    angles = np.asarray(values, dtype=float)
    inside = (angles > -np.pi) & (angles <= np.pi)
    return np.where(inside, angles, np.pi - np.mod(np.pi - angles, 2 * np.pi))
