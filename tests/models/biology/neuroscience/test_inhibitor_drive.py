"""Test the `_inhibitor_drive` module of `skcomplex.models.biology.neuroscience`."""

import numpy as np
import pytest

from skcomplex.core.elements import FieldElement
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.neuroscience import InhibitorDriveInteraction, InhibitorElement, TermanWangElement


def test_inhibitor_drive_interaction_follows_most_active_oscillator():
    """Test that the inhibitor relaxes toward the largest activity of the oscillators, 1 when one is active."""
    values = {'x': np.array([[2.0, -2.0]]), 'm': np.array([[1.0, 1.0]]), 'epsilon': 1e-9}
    oscillators = FieldElement('oscillators', element=TermanWangElement, space=LatticeSpace((1, 2)), values=values)
    drive = InhibitorDriveInteraction('drive', inhibitor='inhibitor', oscillators='oscillators')
    system = System(elements=[oscillators, InhibitorElement('inhibitor')], interactions=[drive])
    trajectory = Simulation(system, dt=0.01).prepare().run(until=0.1).trajectory_
    assert trajectory.values('inhibitor.z')[-1] == pytest.approx(1 - np.exp(-0.3), rel=1e-3)
