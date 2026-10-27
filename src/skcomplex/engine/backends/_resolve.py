"""Choose the backend of a problem: the backend passed, or NumPy or Numba by the size of the system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import AUTO_MEMBERS, BaseBackend
from ._numba import NumbaBackend
from ._numpy import NumpyBackend


def resolve_backend(backend: BaseBackend | None, members: int) -> BaseBackend:
    """Return the backend passed, or the backend that suits the size of a system when none is passed.

    Without a backend, `NumbaBackend` runs systems whose groups hold at least 1000 members in total, and
    `NumpyBackend` the others.

    Args:
        backend:
            The backend instance, or `None`.

        members:
            The total number of members of the system's groups.

    Returns:
        backend:
            The backend passed, or a new backend.

    Examples:
        >>> from skcomplex.engine.backends import resolve_backend
        >>> type(resolve_backend(None, 10)).__name__
        'NumpyBackend'
    """
    if backend is not None:
        return backend
    return NumbaBackend() if members >= AUTO_MEMBERS else NumpyBackend()
