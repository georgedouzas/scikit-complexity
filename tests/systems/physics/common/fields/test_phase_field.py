"""Test the `_phase_field` module of `skcomplex.systems.physics.common.fields`."""

import numpy as np

from skcomplex.core.components.elements import FieldElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.physics.common.fields import PhaseFieldElement


def test_phase_field_holds_its_order_parameter_without_interactions():
    """Test that a phase field without interactions keeps its order parameter."""
    values = {'phi': np.linspace(-1.0, 1.0, 8)}
    field = FieldElement('mixture', element=PhaseFieldElement, space=LatticeSpace((8,)), values=values)
    trajectory = Simulation(System(elements=[field]), dt=0.5).prepare().run(until=1.0).trajectory_
    np.testing.assert_allclose(trajectory.values('mixture.phi')[-1], values['phi'])
