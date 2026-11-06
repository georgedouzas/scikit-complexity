"""Test the `_squirmers` module of `skcomplex.systems.physics.active_matter`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_simulation
from skcomplex.systems.physics.active_matter import SquirmerElement, SquirmerSlipInteraction


def _pair(beta, offset, until=8.0):
    """Return the positions and orientations of two squirmers that start along `x` at an offset."""
    group = GroupElement(
        'squirmers',
        element=SquirmerElement,
        size=2,
        values={'x': [[0.0, 0.0, 0.0], offset], 'p': [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
    )
    slip = SquirmerSlipInteraction('slip', target='squirmers', B1=1.5, B2=1.5 * beta)
    system = System(elements=[group], interactions=[slip], space=ContinuousSpace(n_dim=3))
    trajectory = Simulation(system, dt=0.5).prepare().run(until=until).trajectory_
    return trajectory.values('squirmers.x'), trajectory.values('squirmers.p')


def test_lone_squirmer_swims_at_two_thirds_of_b1():
    """Test that a squirmer far from any other swims along its orientation at `2 B1 / 3`."""
    positions, orientations = _pair(-3.0, [0.0, 1e4, 0.0])
    assert (positions[-1, 0] - positions[0, 0]) / 8.0 == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert orientations[-1, 0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_pushers_attract_and_pullers_repel_side_by_side():
    """Test that side by side pushers draw each other in and pullers push each other away (Ishikawa et al.

    2006).
    """
    pushers, _ = _pair(-3.0, [0.0, 4.0, 0.0])
    pullers, _ = _pair(3.0, [0.0, 4.0, 0.0])
    neutral, _ = _pair(0.0, [0.0, 4.0, 0.0])
    gap = [np.linalg.norm(positions[-1, 1] - positions[-1, 0]) for positions in (pushers, pullers, neutral)]
    assert gap[0] < 3.0
    assert gap[1] > 4.5
    assert gap[2] == pytest.approx(4.0, abs=0.05)


def test_pullers_and_pushers_turn_in_opposite_senses():
    """Test that an oblique neighbour turns a pusher and a puller in opposite senses about the normal."""
    _, pushers = _pair(-3.0, [3.0, 3.0, 0.0], until=0.5)
    _, pullers = _pair(3.0, [3.0, 3.0, 0.0], until=0.5)
    assert np.sign(pushers[-1, 0, 1]) == -np.sign(pullers[-1, 0, 1]) != 0


def test_suspension_keeps_unit_orientations_and_separated_spheres():
    """Test that the catalog suspension keeps unit orientations and the repulsion keeps spheres from overlapping."""
    trajectory = create_simulation('squirmers', n_squirmers=8).prepare().run(until=20.0).trajectory_
    orientations = trajectory.values('squirmers.p')
    positions = trajectory.values('squirmers.x')[-1]
    assert np.linalg.norm(orientations, axis=-1) == pytest.approx(1.0, abs=1e-3)
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)[np.triu_indices(8, 1)]
    assert distances.min() > 1.5
