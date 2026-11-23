"""Define the abstract bases of spaces, neighbourhoods, and times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ...errors import SpaceError
from ...types import IntTuple
from .._tags import Tags
from ..base import BaseComponent


class BaseNeighborhood(BaseComponent[None]):
    """An abstract neighbourhood, the members a law reads around a member of a space.

    A subclass says which members count as neighbours of a member, such as the eight cells around a site in
    `RadiusNeighborhood(1, metric='chebyshev')`. It stores its arguments in `__init__`, and `prepare` checks them and
    derives what the engine reads, such as the offsets on a lattice. A space rejects a neighbourhood of a kind it cannot
    read. Its subclasses pick the cells around a tree in a forest fire, the contacts of a person in an epidemic, and the
    birds within sight of a bird in a flock.

    Read more in the user guide, under Spaces.

    Args:
        label:
            The name of the neighbourhood, such as `'moore'`.

    See Also:
        BaseSpace: Places the members whose neighbours it picks.

    Examples:
        >>> from skcomplex.core.components.spaces import BaseNeighborhood, RadiusNeighborhood
        >>> isinstance(RadiusNeighborhood(1), BaseNeighborhood)
        True
    """

    def _get_offsets(self, n_dim: int | None = None) -> tuple[IntTuple, ...] | None:
        """Return the lattice offsets of the neighbourhood, none by default.

        Args:
            n_dim:
                The number of dimensions of the lattice, or `None` when no lattice is known yet.

        Returns:
            offsets:
                The offsets in the order laws read them, or `None` for a neighbourhood that has none.
        """
        return None

    def _get_slots(self) -> tuple[Hashable, ...] | None:
        """Return the slots a rule reads before any space is known, such as offsets.

        A neighbourhood of fixed slots, such as the words of a memory set of a group, returns them in the order a rule
        reads them, and by default the slots are the offsets known without a lattice.

        Returns:
            slots:
                The slots in their order, or `None` for a neighbourhood whose slots depend on the space.
        """
        return self._get_offsets()

    def _join_slots(self, base: Hashable, slot: Hashable) -> Hashable:
        """Return the slot that reads a slot after a base slot, such as a sum of offsets.

        Args:
            base:
                The slot read first.

            slot:
                The slot read after it.

        Returns:
            joined:
                The slot that reads both in turn.

        Raises:
            SpaceError:
                The neighbourhood has no slots to join.
        """
        error_msg = f'{type(self).__name__} has no fixed slots, so its slots cannot be joined.'
        raise SpaceError(error_msg)

    def _get_from_slots(self, slots: Sequence[Hashable]) -> BaseNeighborhood:
        """Return a neighbourhood of the same kind that reads other slots.

        Args:
            slots:
                The slots of the new neighbourhood, in their order.

        Returns:
            neighborhood:
                The neighbourhood of the slots.

        Raises:
            SpaceError:
                The neighbourhood has no fixed slots.
        """
        error_msg = f'{type(self).__name__} has no fixed slots, so it cannot read other slots.'
        raise SpaceError(error_msg)

    def _get_centered(self) -> BaseNeighborhood | None:
        """Return the neighbourhood with the member itself first, or `None`.

        A memory set of a group, for example, puts the identity first. By default it is `None`, and the space decides
        where the member goes, as a lattice does with the zero offset.
        """
        return None

    def _count_centers(self, n_dim: int | None = None) -> int:
        """Return the number of slots that read the member itself, such as zero offsets.

        Args:
            n_dim:
                The number of dimensions of the lattice, or `None` when no lattice is known yet.

        Returns:
            count:
                The number of slots that read the member itself.
        """
        return sum(not any(offset) for offset in self._get_offsets(n_dim) or ())

    def _is_translated(self) -> bool:
        """Check whether the sites read move with the value each cell holds.

        They do not by default. A translated neighbourhood reads the translation table of its space,
        `_get_translation_table`, at the values of the first state a rule reads.
        """
        return False

    def _get_spaces(self) -> tuple[type, ...]:
        """Return the classes of the spaces the neighbourhood is read on, or none.

        A neighbourhood defined outside the core, such as a memory set of a group, names its spaces here. By default it
        names none, and the engine knows the spaces of the neighbourhoods of the core.
        """
        return ()

    def _prepare(self, context: None = None) -> None:
        """Leave the neighbourhood as it is, since it has nothing to prepare by default."""
        super()._prepare(context)


class BaseSpace(BaseComponent[None]):
    """An abstract space, where the members of a group live and who neighbours whom.

    A subclass places the members of a group and says which of them are neighbours, such as the 100 sites of
    `LatticeSpace((10, 10))`. It holds no values, only the structure the members sit on, and it gives vector values
    their length. It stores its arguments in `__init__`, and `prepare` checks them and derives its size, shape, and
    positions. A space with sites answers the private hooks the engine reads, such as `_get_neighbors` for the sites
    around a site, and a space without sites, such as a continuous box, answers none of them. Its subclasses hold the
    grid of a forest, the contact network of an epidemic, and the box of a gas.

    Read more in the user guide, under Spaces.

    Args:
        label:
            The name of the space in its system, such as `'grid'`.

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
            The name or class of the visual of groups on the space when neither the group nor its element sets one, or
            `None`.

    See Also:
        LatticeSpace: Places the members on a grid of sites.
        NetworkSpace: Places the members on the nodes of a graph.
        ContinuousSpace: Lets the members move to any real position.
        BaseNeighborhood: Picks the members a law reads around a member.

    Examples:
        >>> from skcomplex.core.components.spaces import BaseSpace, LatticeSpace
        >>> space = LatticeSpace((10, 10)).prepare()
        >>> isinstance(space, BaseSpace), space.size_
        (True, 100)
    """

    space_tags: ClassVar[tuple[str, ...]] = ()

    def _get_tags(self) -> Tags:
        """Return the tags of the space: the first space tag of its class, and no laws."""
        return Tags(space=frozenset(self.space_tags[:1]))

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

    def _has_sites(self) -> bool:
        """Check whether the space has sites joined by neighbours, as a lattice does.

        A space without sites, such as a continuous one, has none by default. A space with sites returns `True` and
        answers the hooks of its sites.
        """
        return False

    def _get_default_neighborhood(self) -> BaseNeighborhood:
        """Return the neighbourhood that `run` reads when it is given none."""
        error_msg = f'{type(self).__name__} has no sites, so it has no neighbourhood.'
        raise SpaceError(error_msg)

    def _get_index_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
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

    def _get_translation_table(self, neighborhood: BaseNeighborhood) -> npt.NDArray[np.intp]:
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

    def _get_ball(self, radius: int) -> BaseNeighborhood:
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

    def _get_neighbor_pairs(self, layer: str | None = None) -> list[tuple[int, int]]:
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

    def _get_volumes(self) -> npt.NDArray[np.float64] | None:
        """Return the fraction of every cell inside the domain, or `None` for whole cells.

        Every cell is whole by default. A space whose domain cuts its cells, such as a simplex, returns the fractions.
        """
        return None

    def _get_apertures(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the open fraction of every face along every axis, empty for open faces.

        Every face is open by default.
        """
        return ()

    def _supports_operator(self, name: str) -> bool:
        """Check whether a field on the space takes an operator.

        Every space takes every operator by default.

        Args:
            name:
                The name of the operator, such as `'laplacian'`.

        Returns:
            supported:
                Whether a law may apply the operator to a field on the space.
        """
        return True

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
        reads = self._get_default_neighborhood() if neighborhood is None else neighborhood
        row = self._get_index_table(reads)[site]
        return row[row < self.size_]


class _BaseTime(BaseComponent[None]):
    """An abstract time, which decides the kind of laws a system may hold.

    A time stores its arguments in its `__init__`, and `prepare` checks them and sets its schedule.

    Unlike `BaseSpace`, which says where members are, it says how the states advance. Choose it when a system advances
    in a new way, and choose `_ContinuousTime`, `_DiscreteTime`, or `_StaticTime` otherwise.

    It appears in every field through its subclasses, such as the flow of an epidemic in continuous time, the
    generations of a population in discrete time, and the static balance of an economy.

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
