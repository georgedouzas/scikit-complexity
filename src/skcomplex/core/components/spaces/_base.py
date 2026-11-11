"""Define the abstract bases of spaces, neighbourhoods, and times."""

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
from ..base import BaseComponent


class BaseNeighborhood(BaseComponent[None, SimpleNamespace]):
    """An abstract neighbourhood, the members a law reads around a member of a space.

    A neighbourhood stores its arguments in its `__init__`, and `prepare` checks them and derives what the engine reads,
    such as the offsets of a lattice neighbourhood. The space that reads it rejects a neighbourhood of another kind. It
    returns no laws.

    Unlike `BaseSpace`, which places members, it selects some of them around one member. Choose it when a new rule for
    choosing neighbours is needed, and choose an existing neighbourhood, such as `BallNeighborhood`, otherwise.

    It appears in every field through its subclasses, such as the eight cells around a site of a lattice, the contacts
    of a person in a network, and the birds within sight of a bird in a flock.

    Args:
        label:
            The name of the neighbourhood, such as `'moore'`.
    """

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Run nothing, since a neighbourhood contributes no laws to a system."""

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return no laws, since a neighbourhood only tells laws which members they read."""
        return []

    def _prepare(self, context: None = None) -> None:
        """Leave the neighbourhood as it is, since it has nothing to prepare by default."""
        super()._prepare(context)


class BaseSpace(BaseComponent[None, SimpleNamespace]):
    """An abstract space, where the members of a group live and who neighbours whom.

    A space also gives vector states their dimension. It stores its arguments in its `__init__`, `prepare` checks them
    and derives its facts, such as its size and shape, and `_get_laws` returns the laws it adds, none by default. A
    space with sites answers the hooks the engine reads: `_get_neighbors` gives the sites around a site, `_index_table`
    the neighbour of every site in every slot of a neighbourhood, `_translation_table` the sites a group space reads at
    the products of a site and a group element, `_ball` the neighbourhood within a radius, `_neighbor_pairs` the pairs
    of neighbouring sites, and `_default_neighborhood` the neighbourhood that `_get_neighbors` reads when none is given.
    A space without sites, such as a continuous one, answers none of them.

    Unlike `BaseElement`, it holds no states, only the structure that members sit on. Choose it when members live on a
    new kind of structure, and choose `LatticeSpace`, `NetworkSpace`, or `ContinuousSpace` otherwise.

    It appears in every field through its subclasses, such as a grid of cells in an ecosystem, a contact network in an
    epidemic, and a box of particles in a gas.

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
            The name of the visual of groups on the space when neither the group nor its element sets one, or `None`.
    """

    space_tags: ClassVar[tuple[str, ...]] = ()

    def _get_tags(self) -> Tags:
        """Return the tags of the space: the first space tag of its class, and no laws."""
        return Tags(space=frozenset(self.space_tags[:1]))

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Run nothing, since a space contributes no laws to a system."""

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the laws the space contributes for a namespace of symbols.

        There are none by default.
        """
        return []

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


class BaseTime(BaseComponent[None, SimpleNamespace]):
    """An abstract time, which decides the kind of laws a system may hold.

    A time stores its arguments in its `__init__`, `prepare` checks them and sets its schedule, and `_get_frames`
    returns the times of the frames of a run up to a time. It returns no laws.

    Unlike `BaseSpace`, which says where members are, it says how the states advance. Choose it when a system advances
    in a new way, and choose `ContinuousTime`, `DiscreteTime`, or `StaticTime` otherwise.

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

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Run nothing, since a time contributes no laws to a system."""

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return no laws.

        The time decides which laws a system may hold but contributes none.
        """
        return []

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
