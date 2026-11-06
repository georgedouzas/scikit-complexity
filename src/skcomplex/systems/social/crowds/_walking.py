"""Walking of pedestrians to the next room on the way out, an encounter with the exit signs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import EncounterInteraction
from ....core.specs import ParameterSpec, RoleSpec


def building_of(branching: int, depth: int) -> nx.Graph:
    """Return a building whose rooms form a balanced tree, with the exit door in room 0, the root.

    Args:
        branching:
            The number of rooms behind every door of a room further from the exit.

        depth:
            The number of doors between the exit room and the rooms furthest from it.

    Returns:
        building:
            The network of rooms, labelled `0` to `rooms - 1`, whose edges are the doors.

    Examples:
        >>> from skcomplex.systems.social.crowds import building_of
        >>> building_of(2, 2).number_of_nodes()
        7
    """
    return nx.balanced_tree(branching, depth)


def walking_speeds(speeds: int) -> npt.NDArray[np.float64]:
    """Return the walking speed of every speed class, evenly spaced from a quarter of the free speed to the free speed.

    Args:
        speeds:
            The number of speed classes.

    Returns:
        speeds:
            The walking speed of every class, as a fraction of the free speed.

    Examples:
        >>> from skcomplex.systems.social.crowds import walking_speeds
        >>> walking_speeds(4).tolist()
        [0.25, 0.5, 0.75, 1.0]
    """
    return np.linspace(0.25, 1.0, speeds)


def next_rooms(building: nx.Graph, exit_room: int) -> list[int]:
    """Return the next room on the shortest path to the exit room from every room, and the outside for the exit room.

    Args:
        building:
            The network of rooms, labelled `0` to `rooms - 1`, whose edges are the doors.

        exit_room:
            The room whose door leads outside.

    Returns:
        next_rooms:
            The next room of every room, with `rooms` standing for the outside.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.social.crowds import next_rooms
        >>> next_rooms(nx.path_graph(3), 0)
        [3, 0, 1]
    """
    paths = nx.shortest_path(building, target=exit_room)
    rooms = building.number_of_nodes()
    return [rooms if room == exit_room else int(paths[room][1]) for room in range(rooms)]


def walking_table(building: nx.Graph, exit_room: int, speeds: int) -> tuple[npt.NDArray[np.float64], ...]:
    """Return the rate pattern and the transition table of walking to the next room, an encounter with exit signs.

    The sites of the crowd are indexed by room and speed class in row-major order, with one last row of sites for the
    outside. A pedestrian of speed `v` in a room walks to the next room on the way to the exit at the rate `v`.

    Args:
        building:
            The network of rooms, labelled `0` to `rooms - 1`.

        exit_room:
            The room whose door leads outside.

        speeds:
            The number of speed classes.

    Returns:
        pattern:
            The rate pattern `eta[h, 0]`, the walking speed of the site, 0 outside.

        table:
            The transition table `A[h, 0, i]`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.social.crowds import walking_table
        >>> pattern, table = walking_table(nx.path_graph(2), 0, 2)
        >>> pattern.ravel().tolist()
        [0.25, 1.0, 0.25, 1.0, 0.0, 0.0]
    """
    following = next_rooms(building, exit_room)
    velocity = walking_speeds(speeds)
    sites = (len(following) + 1) * speeds
    pattern = np.zeros((sites, 1))
    table = np.zeros((sites, 1, sites))
    for site in range(sites):
        room, speed = divmod(site, speeds)
        if room == len(following):
            table[site, 0, site] = 1.0
            continue
        pattern[site, 0] = velocity[speed]
        table[site, 0, following[room] * speeds + speed] = 1.0
    return pattern, table


class WalkingInteraction(EncounterInteraction):
    """Walking of pedestrians to the next room on the way out, an encounter with the exit signs.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the crowd, a distribution over rooms and speed classes.

        field:
            The path of the group of one member whose fixed density is the guidance of the exit signs.

        branching:
            The number of rooms behind every door of the building, a balanced tree of rooms with the exit in room 0.

        depth:
            The number of doors between the exit room and the rooms furthest from it.

        speeds:
            The number of speed classes.

        mobility:
            The rate at which pedestrians at the free speed leave a room.

    Examples:
        >>> from skcomplex.systems.social.crowds import WalkingInteraction
        >>> WalkingInteraction('walking', candidate='crowd', field='signs').mobility
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        branching: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 2,
        depth: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 2,
        speeds: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 3,
        mobility: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.branching = branching
        self.depth = depth
        self.speeds = speeds
        self.mobility = mobility

    def _get_rate(self, s: SimpleNamespace) -> object:
        """Return the rate at which every speed class of every room leaves it."""
        building = building_of(int(self.branching), int(self.depth))
        return walking_table(building, 0, int(self.speeds))[0] * s.mobility

    def _get_kernel(self, s: SimpleNamespace) -> object:
        """Return the table that moves a walker to the next room on the way out."""
        building = building_of(int(self.branching), int(self.depth))
        return walking_table(building, 0, int(self.speeds))[1]
