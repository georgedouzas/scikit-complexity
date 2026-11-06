"""Test the `_schooling` module of `skcomplex.systems.biology.collective_behaviour`."""

import math

import numpy as np

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace, DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec
from skcomplex.systems.biology.collective_behaviour import SchoolingInteraction, ShoalMemberElement


def _headings(locations):
    """Return the headings of fish heading east after one simultaneous turn."""
    shoal = GroupElement(
        'shoal',
        element=ShoalMemberElement,
        size=len(locations),
        values={'vx': 1.0, 'vy': 0.0, 'max_turn': 0.4},
        location=locations,
    )
    system = System(
        elements=[shoal],
        interactions=[SchoolingInteraction('schooling', target='shoal', phase='turn')],
        space=ContinuousSpace(extent=(100.0, 100.0), boundary='periodic'),
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('turn', activation='simultaneous')])),
    )
    snapshot = Simulation(system, random_state=0).prepare().run(until=1).trajectory_.agents('shoal', 1)
    return np.stack([snapshot.values['vx'], snapshot.values['vy']], axis=1)


def test_schooling_interaction_repulsion_overrides_attraction():
    """Test that a fish with a neighbour within the zone of repulsion turns away from it by at most `max_turn`."""
    found = _headings([(50.0, 50.0), (50.0, 50.5), (50.0, 54.0)])
    np.testing.assert_allclose(found[0], [math.cos(-0.4), math.sin(-0.4)])


def test_schooling_interaction_attracts_beyond_repulsion():
    """Test that a fish with neighbours only in the zone of interaction turns toward them by at most `max_turn`."""
    found = _headings([(50.0, 50.0), (50.0, 54.0)])
    np.testing.assert_allclose(found, [[math.cos(0.4), math.sin(0.4)], [math.cos(-0.4), math.sin(-0.4)]])
