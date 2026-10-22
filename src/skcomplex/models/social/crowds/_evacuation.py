"""Crowd models: the kinetic theory of the evacuation of a building whose rooms form a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.base import Domain, Expression, parameter, role, state
from ....core.components import Element, Group
from ....core.interactions import Interaction
from ....core.laws import Encounter, Law, Symbols, moment
from ....core.runtime import register_model
from ....core.space import ContinuousTime, LatticeSpace
from ....core.system import Observable, System

EVACUATION_REFERENCES = ('Bellomo et al. 2017', 'Bellomo and Gibelli 2015', 'Agnelli, Colasuonno, and Knopoff 2015')


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
        >>> from skcomplex.models.social.crowds import building_of
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
        >>> from skcomplex.models.social.crowds import walking_speeds
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
        >>> from skcomplex.models.social.crowds import next_rooms
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
        >>> from skcomplex.models.social.crowds import walking_table
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


def crowding_table(rooms: int, speeds: int, step: int) -> tuple[npt.NDArray[np.float64], ...]:
    """Return the rate pattern and the transition table of encounters that change the speed class by one step.

    With `step = -1`, two pedestrians in the same room meet and the candidate slows down one class, so crowded rooms
    walk slower. With `step = 1`, a pedestrian meets the exit signs and speeds up one class.

    Args:
        rooms:
            The number of rooms.

        speeds:
            The number of speed classes.

        step:
            `-1` to slow down in encounters within a room, `1` to speed up in encounters with the exit signs.

    Returns:
        pattern:
            The rate pattern, 1 for the encounters that act.

        table:
            The transition table.

    Examples:
        >>> from skcomplex.models.social.crowds import crowding_table
        >>> crowding_table(1, 2, 1)[0].ravel().tolist()
        [1.0, 0.0, 0.0, 0.0]
    """
    sites = (rooms + 1) * speeds
    fields = sites if step < 0 else 1
    pattern = np.zeros((sites, fields))
    table = np.zeros((sites, fields, sites))
    for site in range(sites):
        room, speed = divmod(site, speeds)
        outcome = min(max(speed + step, 0), speeds - 1)
        table[site, :, room * speeds + outcome] = 1.0
        if room == rooms or outcome == speed:
            continue
        if step > 0:
            pattern[site, 0] = 1.0
        else:
            pattern[site, room * speeds : (room + 1) * speeds] = 1.0
    return pattern, table


def evacuated_fraction(s: Symbols, rooms: int, speeds: int) -> Expression:
    """Return the fraction of the crowd that has left the building.

    Args:
        s:
            The namespace of one site of the crowd.

        rooms:
            The number of rooms.

        speeds:
            The number of speed classes.

    Returns:
        fraction:
            The moment of the indicator of the outside sites, divided by the density.
    """
    outside = (np.arange((rooms + 1) * speeds) >= rooms * speeds).astype(float)
    return moment(s.f, 1, outside, normalized=True)


def mean_speed(s: Symbols, rooms: int, speeds: int) -> Expression:
    """Return the mean walking speed of the crowd, as a fraction of the free speed.

    Args:
        s:
            The namespace of one site of the crowd.

        rooms:
            The number of rooms.

        speeds:
            The number of speed classes.

    Returns:
        speed:
            The normalized first moment of the speeds of the sites.
    """
    return moment(s.f, 1, np.tile(walking_speeds(speeds), rooms + 1), normalized=True)


class Pedestrians(Element):
    """Density of pedestrians at one site of a crowd, a pair of a room and a speed class.

    Args:
        label:
            The label of the crowd.

        f:
            The initial density.

    Examples:
        >>> from skcomplex.models.social.crowds import Pedestrians
        >>> Pedestrians.group('signs', 1, f=1.0).size
        1
    """

    f: float = state(default=0.0, domain=Domain.NONNEGATIVE)


class Walking(Interaction):
    """Walking of pedestrians to the next room on the way out, an encounter with the exit signs.

    Args:
        label:
            The label of the interaction.

        crowd:
            The path of the crowd, a distribution over rooms and speed classes.

        signs:
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
        >>> from skcomplex.models.social.crowds import Walking
        >>> Walking('walking', crowd='crowd', signs='signs').mobility
        0.5
    """

    crowd: str = role()
    signs: str = role()
    branching: int = parameter(default=2, domain=Domain.POSITIVE)
    depth: int = parameter(default=2, domain=Domain.POSITIVE)
    speeds: int = parameter(default=3, domain=Domain.POSITIVE)
    mobility: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the encounter of the crowd with the exit signs.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        building = building_of(int(self.branching), int(self.depth))
        pattern, table = walking_table(building, 0, int(self.speeds))
        return [Encounter(s.crowd.f, s.signs.f, pattern * s.mobility, table)]


class Crowding(Interaction):
    """Encounters that slow pedestrians down in crowded rooms, and the exit signs that speed them up again.

    Two pedestrians in the same room meet at the rate `crowding`, and the candidate slows down one speed class. A
    pedestrian meets the exit signs at the rate `recovery` and speeds up one class. Pedestrians outside keep their
    speed.

    Args:
        label:
            The label of the interaction.

        crowd:
            The path of the crowd, a distribution over rooms and speed classes.

        signs:
            The path of the group of one member whose fixed density is the guidance of the exit signs.

        rooms:
            The number of rooms.

        speeds:
            The number of speed classes.

        crowding:
            The rate of the encounters within a room that slow pedestrians down.

        recovery:
            The rate at which pedestrians speed up again.

    Examples:
        >>> from skcomplex.models.social.crowds import Crowding
        >>> Crowding('crowding', crowd='crowd', signs='signs').crowding
        2.0
    """

    crowd: str = role()
    signs: str = role()
    rooms: int = parameter(default=7, domain=Domain.POSITIVE)
    speeds: int = parameter(default=3, domain=Domain.POSITIVE)
    crowding: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    recovery: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the slowing encounters within rooms and the recovery with the exit signs.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        rooms, speeds = int(self.rooms), int(self.speeds)
        slow_pattern, slow_table = crowding_table(rooms, speeds, -1)
        fast_pattern, fast_table = crowding_table(rooms, speeds, 1)
        return [
            Encounter(s.crowd.f, s.crowd.f, slow_pattern * s.crowding, slow_table),
            Encounter(s.crowd.f, s.signs.f, fast_pattern * s.recovery, fast_table),
        ]


def _sizes_of(system: System) -> tuple[int, int]:
    """Return the number of rooms and of speed classes of a model, from its crowding interaction."""
    crowding = next(item for item in system.interactions if isinstance(item, Crowding))
    return int(crowding.rooms), int(crowding.speeds)


def evacuation_observables(rooms: int, speeds: int) -> list[Observable]:
    """Return the evacuated fraction and the mean walking speed of a crowd `crowd`.

    Args:
        rooms:
            The number of rooms.

        speeds:
            The number of speed classes.

    Returns:
        observables:
            The observables, each reading the moments of the crowd.

    Examples:
        >>> from skcomplex.models.social.crowds import evacuation_observables
        >>> [item.label for item in evacuation_observables(7, 3)]
        ['evacuated', 'mean_speed']
    """
    return [
        Observable('evacuated', partial(evacuated_fraction, rooms=rooms, speeds=speeds), group='crowd'),
        Observable('mean_speed', partial(mean_speed, rooms=rooms, speeds=speeds), group='crowd'),
    ]


@register_model(
    field='social/crowds',
    tags=('crowds', 'evacuation', 'kinetic', 'network', 'distribution'),
    references=EVACUATION_REFERENCES,
    defaults={'until': 40, 'dt': 0.5},
    observables=lambda system: evacuation_observables(*_sizes_of(system)),
)
def evacuation(
    branching: int = 2,
    depth: int = 2,
    speeds: int = 3,
    *,
    mobility: float = 0.5,
    crowding: float = 2.0,
    recovery: float = 0.5,
) -> System:
    """Return the kinetic model of the evacuation of a building whose rooms form a network.

    The crowd is a distribution over pairs of a room and a speed class, held by a site group on a two-dimensional
    lattice indexed by room and speed, with one last row of sites for the outside. The rooms form a balanced tree with
    the exit in the root room, and the walking encounters read its doors, so pedestrians walk to the next room on the
    way out at a rate proportional to their speed. Encounters within a room slow pedestrians down, and the exit signs
    speed them up again, so the crowd queues in the rooms near the exit and the evacuated fraction grows along an
    S-shaped curve.

    Args:
        branching:
            The number of rooms behind every door of the building.

        depth:
            The number of doors between the exit room and the rooms furthest from it.

        speeds:
            The number of speed classes.

        mobility:
            The rate at which pedestrians at the free speed leave a room.

        crowding:
            The rate of the encounters within a room that slow pedestrians down.

        recovery:
            The rate at which pedestrians speed up again.

    Returns:
        system:
            The model, with the crowd `crowd`, the exit signs `signs`, and the interactions `walking` and `crowding`.
            Its standard observables are the evacuated fraction and the mean walking speed.

    Examples:
        >>> from skcomplex.models.social.crowds import evacuation
        >>> evacuation().components[0].size
        24
    """
    rooms = building_of(branching, depth).number_of_nodes()
    initial = np.zeros((rooms + 1, speeds))
    initial[:rooms, speeds // 2] = 1.0 / rooms
    crowd_space = LatticeSpace((rooms + 1, speeds), boundary='fixed')
    return System(
        components=[
            Group('crowd', element=Pedestrians, on='sites', space=crowd_space, values={'f': initial.ravel()}),
            Pedestrians.group('signs', 1, f=1.0),
        ],
        interactions=[
            Walking(
                'walking',
                crowd='crowd',
                signs='signs',
                branching=branching,
                depth=depth,
                speeds=speeds,
                mobility=mobility,
            ),
            Crowding(
                'crowding',
                crowd='crowd',
                signs='signs',
                rooms=rooms,
                speeds=speeds,
                crowding=crowding,
                recovery=recovery,
            ),
        ],
        space=crowd_space,
        time=ContinuousTime(),
    )
