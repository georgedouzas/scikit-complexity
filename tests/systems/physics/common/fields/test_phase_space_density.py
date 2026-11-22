"""Test the `_phase_space_density` module of `skcomplex.systems.physics.common.fields`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import AxisSpec, LatticeSpace, PhaseSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.fields import PhaseSpaceDensityElement


def test_phase_space_density_holds_one_value_per_point_of_its_phase_space():
    """Test that a density on a phase space of 4 positions and 3 velocities keeps its 12 values without laws."""
    space = PhaseSpace(LatticeSpace(4, discretization='finite-volume'), [AxisSpec('v', 3, -1.0, 1.0)])
    density = np.linspace(0.1, 1.2, 12)
    trajectory = System(elements=[PhaseSpaceDensityElement('gas', x=density, space=space)]).prepare().simulate(1, dt=1)
    np.testing.assert_allclose(trajectory.get_values('gas.x')[-1], density)


def test_phase_space_density_rejects_a_negative_value():
    """Test that preparing a system rejects a negative density."""
    with pytest.raises(ValidationError):
        System(elements=[PhaseSpaceDensityElement('a', x=-1.0)]).prepare()
