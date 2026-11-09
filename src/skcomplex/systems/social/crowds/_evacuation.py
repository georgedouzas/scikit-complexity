"""Crowd systems: the kinetic theory of the evacuation of a building.

The rooms of the building form a network.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import GroupElement
from ....core.components.laws.utils import moment
from ....core.components.spaces import ContinuousTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec
from ...general.kinetics import EncounterInteraction
from ..common.kinetics import ActivityDensityElement


def crowding_table(rooms: int, speeds: int, step: int) -> tuple[npt.NDArray[np.float64], ...]:
    """Return the rate pattern and the transition table of crowding encounters.

    The encounters change the speed class by one step. With `step = -1`, two pedestrians in the same room meet and the
    candidate slows down one class, so crowded rooms walk slower. With `step = 1`, a pedestrian meets the exit signs and
    speeds up one class.

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
        >>> from skcomplex.systems.social.crowds import crowding_table
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


class CrowdingInteraction(EncounterInteraction):
    """Encounters that slow pedestrians down in crowded rooms.

    It is an `EncounterInteraction` of the crowd with itself. Two pedestrians in the same room meet at the rate
    `crowding`, and the candidate slows down one speed class. Pedestrians outside keep their speed.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the crowd, a distribution over rooms and speed classes.

        field:
            The path of the crowd that the candidates meet, the crowd again.

        rooms:
            The number of rooms.

        speeds:
            The number of speed classes.

        crowding:
            The rate of the encounters within a room that slow pedestrians down.

    Examples:
        >>> from skcomplex.systems.social.crowds import CrowdingInteraction
        >>> CrowdingInteraction('crowding', candidate='crowd', field='crowd').crowding
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        rooms: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 7,
        speeds: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 3,
        crowding: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.rooms = rooms
        self.speeds = speeds
        self.crowding = crowding

    def _get_rate(self, s: SimpleNamespace) -> object:
        """Return the rate of the slowing encounters of every pair of classes.

        The encounters happen within a room.
        """
        return crowding_table(int(self.rooms), int(self.speeds), -1)[0] * s.crowding

    def _get_kernel(self, s: SimpleNamespace) -> object:
        """Return the table that moves a pedestrian down one speed class."""
        return crowding_table(int(self.rooms), int(self.speeds), -1)[1]


class GuidanceInteraction(EncounterInteraction):
    """Encounters with the exit signs that speed pedestrians up again.

    It is an `EncounterInteraction` of the crowd with the exit signs. A pedestrian meets the exit signs at the rate
    `recovery` and speeds up one class. Pedestrians outside keep their speed.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the crowd, a distribution over rooms and speed classes.

        field:
            The path of the group of one member whose fixed density is the guidance of the exit signs.

        rooms:
            The number of rooms.

        speeds:
            The number of speed classes.

        recovery:
            The rate at which pedestrians speed up again.

    Examples:
        >>> from skcomplex.systems.social.crowds import GuidanceInteraction
        >>> GuidanceInteraction('guidance', candidate='crowd', field='signs').recovery
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        rooms: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 7,
        speeds: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 3,
        recovery: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.rooms = rooms
        self.speeds = speeds
        self.recovery = recovery

    def _get_rate(self, s: SimpleNamespace) -> object:
        """Return the rate of the encounters of every class with the exit signs."""
        return crowding_table(int(self.rooms), int(self.speeds), 1)[0] * s.recovery

    def _get_kernel(self, s: SimpleNamespace) -> object:
        """Return the table that moves a pedestrian up one speed class."""
        return crowding_table(int(self.rooms), int(self.speeds), 1)[1]


def building_of(branching: int, depth: int) -> nx.Graph:
    """Return a building whose rooms form a balanced tree.

    The exit door is in room 0, the root.

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
    """Return the walking speed of every speed class.

    The speeds are evenly spaced from a quarter of the free speed to the free speed.

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
    """Return the next room on the shortest path to the exit room from every room.

    For the exit room it is the outside.

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
    """Return the rate pattern and the transition table of walking to the next room.

    The walking is an encounter with exit signs. The sites of the crowd are indexed by room and speed class in row-major
    order, with one last row of sites for the outside. A pedestrian of speed `v` in a room walks to the next room on the
    way to the exit at the rate `v`.

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
    """Walking of pedestrians to the next room on the way out.

    It is an encounter with the exit signs.

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


EVACUATION_REFERENCES = ('Bellomo et al. 2017 §3.4', 'Bellomo and Gibelli 2015')


def evacuated_fraction(s: SimpleNamespace, rooms: int, speeds: int) -> Expression:
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
    return moment(s.x, 1, outside, normalized=True)


def mean_speed(s: SimpleNamespace, rooms: int, speeds: int) -> Expression:
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
    return moment(s.x, 1, np.tile(walking_speeds(speeds), rooms + 1), normalized=True)


def _sizes_of(system: CompositeSystem) -> tuple[int, int]:
    """Return the number of rooms and of speed classes of a system.

    They are read from its crowding interaction.
    """
    crowding = next(item for item in system.interactions if isinstance(item, CrowdingInteraction))
    return int(crowding.rooms), int(crowding.speeds)


def evacuation_observables(rooms: int, speeds: int) -> list[ObservableSpec]:
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
        >>> from skcomplex.systems.social.crowds import evacuation_observables
        >>> [item.label for item in evacuation_observables(7, 3)]
        ['evacuated', 'mean_speed']
    """
    return [
        ObservableSpec('evacuated', partial(evacuated_fraction, rooms=rooms, speeds=speeds), group='crowd'),
        ObservableSpec('mean_speed', partial(mean_speed, rooms=rooms, speeds=speeds), group='crowd'),
    ]


class EvacuationSystem(BaseSystem):
    """The kinetic system of the evacuation of a building whose rooms form a network.

    The crowd is a distribution over pairs of a room and a speed class, held by a site group on a two-dimensional
    lattice indexed by room and speed, with one last row of sites for the outside. The rooms form a balanced tree with
    the exit in the root room, and the walking encounters read its doors, so pedestrians walk to the next room on the
    way out at a rate proportional to their speed. Encounters within a room slow pedestrians down, and the exit signs
    speed them up again, so the crowd queues in the rooms near the exit and the evacuated fraction grows along an
    S-shaped curve.

    The rooms are the nodes and the speed classes the discrete activity of the kinetic structure for a network of
    Bellomo et al. (2017, sections 3.4 and 3.4.1), and the slowing and guiding encounters follow the behavioural crowd
    of Bellomo and Gibelli (2015). The crowd of chapter 6 of Bellomo et al. (2017), which moves in continuous space
    around walls and obstacles, is a different model. The system has the crowd `crowd`, the exit signs `signs`, and the
    interactions `walking`, `crowding`, and `guidance`. Its standard observables are the evacuated fraction and the mean
    walking speed.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.crowds import EvacuationSystem
        >>> EvacuationSystem().prepare().system_.elements[0].prepare().size_
        24
    """

    field = 'social/crowds'
    topics = ('crowds', 'evacuation', 'kinetic', 'network', 'distribution')
    references = EVACUATION_REFERENCES
    until = 40
    dt = 0.5

    def __init__(
        self,
        branching: int = 2,
        depth: int = 2,
        speeds: int = 3,
        *,
        mobility: float = 0.5,
        crowding: float = 2.0,
        recovery: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.branching = branching
        self.depth = depth
        self.speeds = speeds
        self.mobility = mobility
        self.crowding = crowding
        self.recovery = recovery

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rooms = building_of(self.branching, self.depth).number_of_nodes()
        initial = np.zeros((rooms + 1, self.speeds))
        initial[:rooms, self.speeds // 2] = 1.0 / rooms
        crowd_space = LatticeSpace((rooms + 1, self.speeds), boundary='fixed')
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'crowd',
                    element=ActivityDensityElement,
                    on='sites',
                    space=crowd_space,
                    values={'x': initial.ravel()},
                ),
                GroupElement('signs', element=ActivityDensityElement, size=1, values={'x': 1.0}),
            ],
            interactions=[
                WalkingInteraction(
                    'walking',
                    candidate='crowd',
                    field='signs',
                    branching=self.branching,
                    depth=self.depth,
                    speeds=self.speeds,
                    mobility=self.mobility,
                ),
                CrowdingInteraction(
                    'crowding',
                    candidate='crowd',
                    field='crowd',
                    rooms=rooms,
                    speeds=self.speeds,
                    crowding=self.crowding,
                ),
                GuidanceInteraction(
                    'guidance',
                    candidate='crowd',
                    field='signs',
                    rooms=rooms,
                    speeds=self.speeds,
                    recovery=self.recovery,
                ),
            ],
            space=crowd_space,
            time=ContinuousTime(),
        )

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        system = self._pass_parameters()
        settings['observables'] = evacuation_observables(*_sizes_of(system))
        return settings
