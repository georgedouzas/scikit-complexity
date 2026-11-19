"""Test the `_zones` module of `skcomplex.systems.biology.collective_behaviour`."""

import math

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.biology.collective_behaviour import ClampSpec, RadiusSpec, ZoneCouplingInteraction


def _velocities(locations, velocities, **arguments):
    """Return the velocities of agents after one step of an agent coupling over radii."""
    space = ContinuousSpace(extent=(10.0, 10.0))
    values = {
        'x': np.array([item[0] for item in locations]),
        'y': np.array([item[1] for item in locations]),
        'vx': np.array([item[0] for item in velocities], dtype=float),
        'vy': np.array([item[1] for item in velocities], dtype=float),
    }
    agents = Element('agents', values=values, size=len(locations), space=space)
    coupling = ZoneCouplingInteraction('coupling', target='agents', write=('vx', 'vy'), **arguments)
    system = System(elements=[agents], interactions=[coupling], space=space)
    snapshot = system.prepare().simulate(until=1, random_state=0).get_agents('agents', 1)
    return np.stack([snapshot.values['vx'], snapshot.values['vy']], axis=1)


def test_coupling_agents_over_radii_by_priority():
    """Test that only the radii of the highest priority that hold another agent add their reduced terms."""
    radii = (
        RadiusSpec(1.0, lambda agent, other, system: (1.0, 0.0), priority=1),
        RadiusSpec(5.0, lambda agent, other, system: (0.0, 1.0)),
        RadiusSpec(5.0, lambda agent, other, system: (0.0, 2.0), reduce='mean'),
    )
    found = _velocities([(1.0, 1.0), (1.5, 1.0), (4.0, 1.0)], [(0.0, 0.0)] * 3, radii=radii)
    np.testing.assert_allclose(found, [[1.0, 0.0], [1.0, 0.0], [0.0, 4.0]])


def test_coupling_agents_with_clamped_turn_and_norm():
    """Test that the agent coupling turns the vector by at most the largest angle and clips its norm."""
    radii = (RadiusSpec(5.0, lambda agent, other, system: (-1.0, 3.0)),)
    clamp = ClampSpec(max_angle=0.5, max_norm=2.0)
    found = _velocities([(1.0, 1.0), (2.0, 1.0)], [(1.0, 0.0)] * 2, radii=radii, clamp=clamp)
    np.testing.assert_allclose(found, [[2 * math.cos(0.5), 2 * math.sin(0.5)]] * 2)


def test_coupling_agents_combine_reduction_and_priority():
    """Test that the combination of the agent form reads the reduction and its priority, or `None` for no agent."""
    radii = (RadiusSpec(1.0, lambda agent, other, system: (1.0, 0.0), priority=2),)

    def combine(agent, system, reduction, priority):
        return (0.0, -1.0) if reduction is None else (float(priority), 0.0)

    found = _velocities([(1.0, 1.0), (1.5, 1.0), (8.0, 8.0)], [(0.0, 0.0)] * 3, radii=radii, combine=combine)
    np.testing.assert_allclose(found, [[2.0, 0.0], [2.0, 0.0], [0.0, -1.0]])


def test_check_needs_radii():
    """Test that a zone coupling needs at least one radius."""
    with pytest.raises(ValidationError, match='at least one radius'):
        ZoneCouplingInteraction('coupling', target='agents').prepare()


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (RadiusSpec(1.0, lambda agent, other, system: (0.0,), reduce='max'), 'reduction'),
        (RadiusSpec(1.0, 'term'), 'callable'),
        (RadiusSpec(-1.0, lambda agent, other, system: (0.0,)), 'nonnegative'),
        (ClampSpec(max_angle=-0.1), 'max_angle'),
    ],
)
def test_check_agent_coupling_specs_reject_invalid_settings(spec, message):
    """Test that a radius or a clamp with an invalid setting raises an error naming it."""
    with pytest.raises(ValidationError, match=message):
        spec._check()
