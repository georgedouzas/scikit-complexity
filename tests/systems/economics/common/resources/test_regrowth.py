"""Test the `_regrowth` module of `skcomplex.systems.economics.common.resources`."""

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.systems import System
from skcomplex.core.types import AnyArray
from skcomplex.systems.economics.common.resources import RegrowthInteraction


def _levels(*, relative: bool) -> AnyArray:
    """Return the levels of two stocks over three steps of regrowth, one row per frame."""
    stocks = Element('stocks', values={'level': [0.0, 3.0], 'capacity': 4.0, 'growth': 1.5}, size=2)
    regrowth = RegrowthInteraction('regrowth', target='stocks', state='level', growth='growth', relative=relative)
    system = System(elements=[stocks], interactions=[regrowth])
    trajectory = system.prepare().simulate(until=3, random_state=0)
    return np.array([trajectory.get_agents('stocks', frame).values['level'] for frame in range(4)])


def test_regrowth_adds_an_amount_up_to_the_capacity():
    """Test that an absolute regrowth adds its amount every step and stops at the capacity."""
    levels = _levels(relative=False)
    np.testing.assert_allclose(levels[:, 0], [0.0, 1.5, 3.0, 4.0])
    np.testing.assert_allclose(levels[:, 1], [3.0, 4.0, 4.0, 4.0])


def test_relative_regrowth_adds_a_share_of_the_capacity():
    """Test that a relative regrowth adds its share of the capacity every step."""
    np.testing.assert_allclose(_levels(relative=True)[:, 0], [0.0, 4.0, 4.0, 4.0])
