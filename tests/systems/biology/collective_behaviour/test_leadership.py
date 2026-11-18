"""Test the `_leadership` module of `skcomplex.systems.biology.collective_behaviour`."""

import math

import numpy as np

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.engine.compile import PhaseSpec
from skcomplex.core.engine.solvers import MesaSolver
from skcomplex.core.systems import System
from skcomplex.systems.biology.collective_behaviour import (
    CouzinLeadershipSystem,
    SchoolingInteraction,
    ShoalMemberElement,
    SwimmingInteraction,
)


def _accuracy(informed):
    """Return the mean cosine between the shoal's late heading and the preferred direction, over four shoals."""
    cosines = []
    for seed in range(4):
        system = CouzinLeadershipSystem(30, informed=informed, random_state=seed).prepare()
        trajectory = system.simulate(until=150, dt=1, random_state=seed)
        for frame in range(100, 151):
            snapshot = trajectory.get_agents('shoal', frame)
            heading = np.arctan2(np.mean(snapshot.values['vy']), np.mean(snapshot.values['vx']))
            cosines.append(np.cos(heading))
    return float(np.mean(cosines))


def test_couzin_leadership_informed_minority_leads():
    """Test that an informed minority turns the shoal toward its direction, more accurately when it is larger."""
    naive, few, many = _accuracy(0.0), _accuracy(0.1), _accuracy(0.3)
    assert naive < 0.5
    assert naive < few < many
    assert many > 0.6


def test_swimming_interaction_turns_by_deviation_and_swims():
    """Test that a fish turns its heading by its deviation and swims `speed` along the new heading."""
    shoal = ShoalMemberElement(
        'shoal',
        deviation=0.5,
        speed=2.0,
        values={'x': [99.0], 'y': [1.0]},
        size=1,
        space=ContinuousSpace(extent=(100.0, 100.0), boundary='periodic'),
    )
    system = System(elements=[shoal], interactions=[SwimmingInteraction('swimming', target='shoal')]).prepare()
    snapshot = system.simulate(random_state=0, until=1).get_agents('shoal', 1)
    heading = [math.cos(0.5), math.sin(0.5)]
    np.testing.assert_allclose([snapshot.values['vx'][0], snapshot.values['vy'][0]], heading)
    np.testing.assert_allclose(snapshot.locations[0], np.mod([99.0 + 2 * heading[0], 1.0 + 2 * heading[1]], 100.0))


def _headings(locations):
    """Return the headings of fish heading east after one simultaneous turn."""
    points = np.asarray(locations)
    shoal = ShoalMemberElement(
        'shoal',
        max_turn=0.4,
        values={'x': points[:, 0], 'y': points[:, 1]},
        size=len(locations),
        space=ContinuousSpace(extent=(100.0, 100.0), boundary='periodic'),
    )
    system = System(
        elements=[shoal],
        interactions=[SchoolingInteraction('schooling', target='shoal', phase='turn')],
        solver=MesaSolver(phases=[PhaseSpec('turn', activation='simultaneous')]),
    ).prepare()
    snapshot = system.simulate(random_state=0, until=1).get_agents('shoal', 1)
    return np.stack([snapshot.values['vx'], snapshot.values['vy']], axis=1)


def test_schooling_interaction_repulsion_overrides_attraction():
    """Test that a fish with a neighbour within the zone of repulsion turns away from it by at most `max_turn`."""
    found = _headings([(50.0, 50.0), (50.0, 50.5), (50.0, 54.0)])
    np.testing.assert_allclose(found[0], [math.cos(-0.4), math.sin(-0.4)])


def test_schooling_interaction_attracts_beyond_repulsion():
    """Test that a fish with neighbours only in the zone of interaction turns toward them by at most `max_turn`."""
    found = _headings([(50.0, 50.0), (50.0, 54.0)])
    np.testing.assert_allclose(found, [[math.cos(0.4), math.sin(0.4)], [math.cos(-0.4), math.sin(-0.4)]])
