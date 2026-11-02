"""Test the `_phase_field` module of `skcomplex.models.physics.common.fields`."""

import numpy as np

from skcomplex.core.elements import FieldElement
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.common.fields import PhaseFieldElement


def test_phase_field_holds_its_order_parameter_without_interactions():
    """Test that a phase field without interactions keeps its order parameter."""
    values = {'phi': np.linspace(-1.0, 1.0, 8)}
    field = FieldElement('mixture', element=PhaseFieldElement, space=LatticeSpace((8,)), values=values)
    trajectory = Simulation(System(elements=[field]), dt=0.5).check().prepare().run(until=1.0).trajectory_
    np.testing.assert_allclose(trajectory.values('mixture.phi')[-1], values['phi'])
