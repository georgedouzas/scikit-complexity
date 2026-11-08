"""Define the bases of the space and the time of a system, and of neighbourhoods.

The neighbourhoods are those of the members of a space.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ...errors import SpaceError
from ...specs import Tags
from ..base import BasePart


class BaseNeighborhood(BasePart[None, SimpleNamespace]):
    """Base of the neighbourhoods.

    A neighbourhood is the part of a system that gives the members a law reads around a member of a space. A
    neighbourhood is a part: it stores its arguments in its own `__init__`, `prepare` validates them and derives what
    the engine reads, such as the offsets of a lattice neighbourhood. The space that reads it rejects a neighbourhood of
    another kind. It contributes no laws.

    Args:
        label:
            The label of the neighbourhood.
    """

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Run nothing, since a neighbourhood contributes no laws to a system."""

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return no laws, since a neighbourhood only tells laws which members they read."""
        return []

    def _check(self) -> None:
        """Accept the neighbourhood.

        By default it has nothing to check besides its label.
        """

    def _prepare(self, context: None = None) -> None:
        """Leave the neighbourhood as it is, since it has nothing to prepare by default."""


class BaseSpace(BasePart[None, SimpleNamespace]):
    """Base of the spaces.

    A space is the part of a system where members live and which gives vector states their dimension. A space is a part
    of its system: it stores its arguments in its own `__init__`, `prepare` validates them and derives the facts of the
    space, and `_get_laws` returns the laws the space contributes, none by default. A space with sites answers the hooks
    the engine reads: `_get_neighbors` gives the sites around a site, `_index_table` the neighbour of every site in
    every slot of a neighbourhood, `_translation_table` the sites a group space reads at the products of a site and a
    group element, `_ball` the neighbourhood within a radius, `_neighbor_pairs` the pairs of neighbouring sites, and
    `_default_neighborhood` the neighbourhood `_get_neighbors` reads when it is given none. A space without sites, such
    as a continuous one, answers none of them.

    Args:
        label:
            The label of the space in its system.

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
            The name of the visual of groups on the space when neither the group nor its element sets one, or `None`.
    """

    space_tags: ClassVar[tuple[str, ...]] = ()

    def _get_tags(self) -> Tags:
        """Return the tags of the space: the first space tag of its class, and no laws."""
        return Tags(space=frozenset(self.space_tags[:1]))

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Run nothing, since a space contributes no laws to a system."""

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws the space contributes for a namespace of symbols.

        There are none by default.
        """
        return []

    def _check(self) -> None:
        """Accept the space.

        By default it has nothing to check besides its label.
        """

    def _get_lacks(self) -> frozenset[str]:
        """Return the options lattices offer that the space does not.

        An example is `'image'`, and there are none by default.
        """
        return frozenset()

    def _count_parts(self, on: str, layer: str | None = None) -> int:
        """Return the number of sites of the space.

        A group on its sites has one member on each.

        Args:
            on:
                `'sites'`, `'edges'`, or `'hyperedges'`.

            layer:
                The layer of a network whose edges a group lives on, or `None` for the graph of the network.

        Returns:
            count:
                The number of parts.

        Raises:
            SpaceError:
                The space does not have the part.
        """
        if on == 'sites' and self.size_ is not None:
            return int(self.size_)
        error_msg = f'{type(self).__name__} has no {on}.'
        raise SpaceError(error_msg)

    def _default_neighborhood(self) -> BaseNeighborhood:
        """Return the neighbourhood that `run` reads when it is given none."""
        error_msg = f'{type(self).__name__} has no sites, so it has no neighbourhood.'
        raise SpaceError(error_msg)

    def _index_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
        """Return the site of every neighbour of every site.

        A neighbour outside the space gets `size_`.

        Args:
            neighborhood:
                The neighbourhood the space reads, such as offsets on a lattice.

        Returns:
            table:
                One row per site and one column per neighbour.
        """
        error_msg = f'{type(self).__name__} has no sites, so it has no neighbour table.'
        raise SpaceError(error_msg)

    def _translation_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
        """Return the site of `g h s` for every site, group element, and word.

        The site is `g`, the group element `h`, and the word `s` of a translated memory set.

        Args:
            neighborhood:
                The memory set translated by the value of a cell.

        Returns:
            table:
                One row per site `g`, one column per element `h`, and one layer per word `s`, with `size_` outside.
        """
        error_msg = f'{type(self).__name__} is no group space, so it reads no memory translated by a group element.'
        raise SpaceError(error_msg)

    def _ball(self, radius: int) -> BaseNeighborhood:
        """Return the neighbourhood of every site within a radius.

        The site itself is included.

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

    def _get_neighbors(self, site: int, neighborhood: BaseNeighborhood | None = None) -> npt.NDArray[np.intp]:
        """Return the sites around a site inside the space.

        They come in the order of a neighbourhood or the default one.

        Args:
            site:
                The site.

            neighborhood:
                The neighbourhood to read, or `None` for the default neighbourhood of the space.

        Returns:
            sites:
                The sites around the site, without those outside the space.
        """
        reads = self._default_neighborhood() if neighborhood is None else neighborhood
        row = self._index_table(reads)[site]
        return row[row < self.size_]


class BaseTime(BasePart[None, SimpleNamespace]):
    """Base of the time of a system.

    A time is the part that decides the laws the system may hold. A time is a part of its system: it stores its
    arguments in its own `__init__`, `prepare` validates them and sets its schedule, and `_get_frames` returns the times
    of the frames of a run up to a time. It contributes no laws.

    Args:
        label:
            The label of the time in its system.

    Attributes:
        schedule_:
            The schedule of a discrete time, or `None`.
    """

    time_tag: ClassVar[str] = ''

    def _get_tags(self) -> Tags:
        """Return the tags of the time: the time tag of its class, and no laws."""
        return Tags(time=frozenset({self.time_tag} - {''}))

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Run nothing, since a time contributes no laws to a system."""

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return no laws.

        The time decides which laws a system may hold but contributes none.
        """
        return []

    def _check(self) -> None:
        """Accept the time.

        By default it has nothing to check besides its label.
        """

    def _get_frames(self, until: float) -> npt.NDArray[np.float64]:
        """Return the times of the frames of a run up to a time.

        By default they are its start and its end.

        Args:
            until:
                The end of the run.

        Returns:
            times:
                The times of the frames.
        """
        return np.array([0.0, float(until)])
