"""Define the bases of where the parts of a system live and of how its time advances."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt

from ._errors import SpaceError
from ._estimator import BaseEstimator
from ._kinds import SpaceKind, TimeKind


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


class BaseSpace(BaseEstimator):
    """Base class of spaces, where the members of groups live and which give vector states their dimension.

    `fit()` checks the arguments and derives the facts of the space, and `run(site, neighborhood=None)` returns the
    sites around a site. A child sets the fitted attributes in `_fit`, and answers the neighbourhood hooks the engine
    reads: `_index_table` gives the neighbour of every site in every slot of a neighbourhood, `_ball` the neighbourhood
    within a radius, `_neighbor_pairs` the pairs of neighbouring sites, and `_default_neighborhood` the neighbourhood
    that `run` reads when it is given none. A space without sites, such as a continuous one, answers none of them.

    The class variable `space_kind` names the kind of the space that laws require, and `lacks` the options of the kind
    the space does not offer, such as `'regular-geometry'` for the checkerboard schedule of a square lattice and
    `'image'` for image visuals.

    Attributes:
        size_:
            The number of sites, or `None` for a space without sites.

        n_dim_:
            The number of dimensions, the length of the vector states of members of the space.

        shape_:
            The shape of the arrays of the values of the sites, `()` for a space without sites.

        positions_:
            The position of every site, one row per site, or an empty array of `n_dim_` columns without sites.

        periodic_:
            Whether opposite sides of the space join.

        default_visual_:
            The name of the visual of groups on the space when neither the group nor its element type declares one,
            or `None` for the visual of the element type.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.base import BaseSpace, SpaceKind
        >>> class RingSpace(BaseSpace):
        ...     space_kind = SpaceKind.LATTICE
        ...
        ...     def __init__(self, n=4):
        ...         self.n = n
        ...
        ...     def _fit(self):
        ...         self.size_, self.n_dim_, self.shape_ = self.n, 1, (self.n,)
        ...         self.positions_ = np.arange(self.n, dtype=float)[:, None]
        ...         self.periodic_, self.default_visual_ = True, 'sites'
        ...
        ...     def _run(self, site, neighborhood=None):
        ...         return np.array([(site - 1) % self.n, (site + 1) % self.n])
        >>> RingSpace().fit().run(0).tolist()
        [3, 1]
    """

    space_kind: ClassVar[SpaceKind]
    lacks: ClassVar[frozenset[str]] = frozenset()

    size_: int | None
    n_dim_: int
    shape_: tuple[int, ...]
    positions_: npt.NDArray[np.float64]
    periodic_: bool
    default_visual_: str | None

    def _default_neighborhood(self) -> BaseNeighborhood:
        """Return the neighbourhood that `run` reads when it is given none."""
        error_msg = f'{type(self).__name__} has no sites, so it has no neighbourhood.'
        raise SpaceError(error_msg)

    def _index_table(self, neighborhood: Any) -> npt.NDArray[np.intp]:  # noqa: ANN401
        """Return the site of every neighbour of every site, with `size_` for neighbours outside the space.

        Args:
            neighborhood:
                The neighbourhood the space reads, such as offsets on a lattice.

        Returns:
            table:
                One row per site and one column per neighbour.
        """
        error_msg = f'{type(self).__name__} has no sites, so it has no neighbour table.'
        raise SpaceError(error_msg)

    def _ball(self, radius: int) -> BaseNeighborhood:
        """Return the neighbourhood of every site within a radius, the site itself included.

        Args:
            radius:
                The radius.

        Returns:
            neighborhood:
                The neighbourhood.
        """
        error_msg = f'{type(self).__name__} has no sites, so it has no balls.'
        raise SpaceError(error_msg)

    def _neighbor_pairs(self, layer: str | None = None) -> list[tuple[int, int]]:
        """Return the pairs of distinct neighbouring sites, each pair once.

        Args:
            layer:
                The layer of a network, or `None` for the graph of the network or a space with one relation.

        Returns:
            pairs:
                The sites of every pair of neighbours.
        """
        error_msg = f'{type(self).__name__} has no sites, so it has no neighbour pairs.'
        raise SpaceError(error_msg)

    def _run(self, site: int, neighborhood: Any = None) -> npt.NDArray[np.intp]:  # noqa: ANN401
        """Return the sites around a site inside the space, in the order of a neighbourhood or the default one."""
        reads = self._default_neighborhood() if neighborhood is None else neighborhood
        row = self._index_table(reads)[site]
        return row[row < self.size_]


class BaseTime(BaseEstimator):
    """Base class of the time of a system, which decides the laws the system may hold.

    Subclasses set the class variable `kind`. `fit()` checks the arguments and derives the kind and the schedule of
    the time, and `run(until)` returns the times of the frames of a run up to a time.

    Attributes:
        time_kind_:
            The kind of the time, the class variable `kind`.

        schedule_:
            The schedule of a discrete time, or `None`.
    """

    kind: ClassVar[TimeKind]

    time_kind_: TimeKind
    schedule_: Any

    def _fit(self) -> None:
        """Set the kind of the time and no schedule."""
        self.time_kind_ = self.kind
        self.schedule_ = None

    def _run(self, until: float) -> npt.NDArray[np.float64]:
        """Return the times of the frames of a run up to a time, its start and its end by default."""
        return np.array([0.0, float(until)])
