"""Define the bases of where the parts of a system live and of how its time advances."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Self

from sklearn.base import BaseEstimator

from ._kinds import SpaceKind, TimeKind


class BaseSpace(BaseEstimator):
    """Base class of spaces, which give vector states and parameters their dimension.

    Subclasses expose the number of dimensions as `n_dim`, and name in `lacks` the options of their kind they do not
    offer, such as `'regular-geometry'` for the checkerboard schedule of a square lattice and `'image'` for image
    visuals.
    """

    space_kind: ClassVar[SpaceKind]
    lacks: ClassVar[frozenset[str]] = frozenset()

    n_dim: int


class BaseTime(BaseEstimator):
    """Base class of the time of a system, which decides the laws the system may hold.

    Subclasses set the class variable `kind`.
    """

    kind: ClassVar[TimeKind]

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the name of the time."""
        return f'{type(self).__name__}()'


class BaseNeighborhood:
    """Base of the neighbourhoods, the members a law reads around a member of a space.

    A neighbourhood is a value: its arguments are kept as given, and `check` returns the neighbourhood with them
    checked and normalized.
    """

    def _check(self) -> Self:
        """Return the neighbourhood with its arguments checked and normalized, itself by default."""
        return self

    def check(self) -> Self:
        """Return the neighbourhood with its arguments checked and normalized.

        Returns:
            neighborhood:
                The checked neighbourhood.

        Raises:
            ValidationError:
                An argument of the neighbourhood is invalid, such as a negative radius.
        """
        return self._check()
