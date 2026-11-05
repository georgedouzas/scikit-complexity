"""Crowd models: the kinetic theory of the evacuation of a building whose rooms form a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.elements import DensityElement, GroupElement
from ....core.laws.utils import moment
from ....core.spaces import ContinuousTime, LatticeSpace
from ....core.system import System
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._crowding import CrowdingInteraction
from ._guidance import GuidanceInteraction
from ._walking import WalkingInteraction, building_of, walking_speeds

EVACUATION_REFERENCES = ('Bellomo et al. 2017', 'Bellomo and Gibelli 2015', 'Agnelli, Colasuonno, and Knopoff 2015')


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
    return moment(s.f, 1, outside, normalized=True)


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
    return moment(s.f, 1, np.tile(walking_speeds(speeds), rooms + 1), normalized=True)


def _sizes_of(system: System) -> tuple[int, int]:
    """Return the number of rooms and of speed classes of a model, from its crowding interaction."""
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
        >>> from skcomplex.models.social.crowds import evacuation_observables
        >>> [item.label for item in evacuation_observables(7, 3)]
        ['evacuated', 'mean_speed']
    """
    return [
        ObservableSpec('evacuated', partial(evacuated_fraction, rooms=rooms, speeds=speeds), group='crowd'),
        ObservableSpec('mean_speed', partial(mean_speed, rooms=rooms, speeds=speeds), group='crowd'),
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
            The model, with the crowd `crowd`, the exit signs `signs`, and the interactions `walking`, `crowding`, and
            `guidance`.
            Its standard observables are the evacuated fraction and the mean walking speed.

    Examples:
        >>> from skcomplex.models.social.crowds import evacuation
        >>> evacuation().elements[0].prepare().size_
        24
    """
    rooms = building_of(branching, depth).number_of_nodes()
    initial = np.zeros((rooms + 1, speeds))
    initial[:rooms, speeds // 2] = 1.0 / rooms
    crowd_space = LatticeSpace((rooms + 1, speeds), boundary='fixed')
    return System(
        elements=[
            GroupElement(
                'crowd',
                element=DensityElement,
                on='sites',
                space=crowd_space,
                values={'f': initial.ravel()},
            ),
            GroupElement('signs', element=DensityElement, size=1, values={'f': 1.0}),
        ],
        interactions=[
            WalkingInteraction(
                'walking',
                candidate='crowd',
                field='signs',
                branching=branching,
                depth=depth,
                speeds=speeds,
                mobility=mobility,
            ),
            CrowdingInteraction(
                'crowding',
                candidate='crowd',
                field='crowd',
                rooms=rooms,
                speeds=speeds,
                crowding=crowding,
            ),
            GuidanceInteraction(
                'guidance',
                candidate='crowd',
                field='signs',
                rooms=rooms,
                speeds=speeds,
                recovery=recovery,
            ),
        ],
        space=crowd_space,
        time=ContinuousTime(),
    )
