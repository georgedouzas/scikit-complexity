"""Define the base of the spaces that systems live in."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

from ...errors import SpaceError
from ..base import BasePart
from ..neighborhoods import BaseNeighborhood


class BaseSpace(BasePart[None, SimpleNamespace]):
    """Base of the spaces, the parts of a system where members live and which give vector states their dimension.

    A space is a part of its system: it stores its arguments in its own `__init__`, `check` validates them, `prepare`
    derives the facts of the space, and `_get_laws` returns the laws the space contributes, none by default. A space
    with sites answers the hooks the engine reads: `_get_neighbors` gives the sites around a site, `_index_table` the
    neighbour of every site in every slot of a neighbourhood, `_ball` the neighbourhood within a radius,
    `_neighbor_pairs` the pairs of neighbouring sites, and `_default_neighborhood` the neighbourhood `_get_neighbors`
    reads when it is given none. A space without sites, such as a continuous one, answers none of them.

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

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Run nothing, since a space contributes no laws to a system."""

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws the space contributes for a namespace of symbols, none by default."""
        return []

    def _check(self) -> None:
        """Accept the space, which has nothing to check besides its label by default."""

    def _get_lacks(self) -> frozenset[str]:
        """Return the options lattices offer that the space does not, such as `'image'`, none by default."""
        return frozenset()

    def _count_parts(self, on: str, layer: str | None = None) -> int:
        """Return the number of sites of the space, which a group on its sites has one member on.

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

    def _get_neighbors(self, site: int, neighborhood: BaseNeighborhood | None = None) -> npt.NDArray[np.intp]:
        """Return the sites around a site inside the space, in the order of a neighbourhood or the default one.

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
