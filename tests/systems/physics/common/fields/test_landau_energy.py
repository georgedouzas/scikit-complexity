"""Test the `_landau_energy` module of `skcomplex.systems.physics.common.fields`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import FieldElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.physics.common.fields import LandauEnergyInteraction, PhaseFieldElement


def _run(conserved):
    """Return the order parameter of a periodic line relaxing in the Landau free energy."""
    values = {'phi': 0.1 * np.sin(2 * np.pi * np.arange(32) / 32) + 0.2}
    field = FieldElement('mixture', element=PhaseFieldElement, space=LatticeSpace((32,), spacing=0.5), values=values)
    energy = LandauEnergyInteraction('energy', target='mixture', kappa=0.5, conserved=conserved)
    system = System(elements=[field], interactions=[energy])
    return Simulation(system, dt=1.0).prepare().run(until=40.0).trajectory_.values('mixture.phi')


def test_non_conserved_field_falls_into_the_nearer_well():
    """Test that a non-conserved field everywhere positive relaxes to the well at 1."""
    np.testing.assert_allclose(_run(conserved=False)[-1], 1.0, atol=1e-3)


def test_conserved_field_keeps_its_mean():
    """Test that a conserved field keeps the mean of the order parameter while it relaxes."""
    phi = _run(conserved=True)
    assert phi[-1].mean() == pytest.approx(phi[0].mean(), rel=1e-9)
