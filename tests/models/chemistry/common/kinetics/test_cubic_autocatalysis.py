"""Test the `_cubic_autocatalysis` module of `skcomplex.models.chemistry.common.kinetics`."""

import numpy as np

from skcomplex.core.elements import FieldElement
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.chemistry.common.kinetics import BrusselatorReactionElement, CubicAutocatalysisInteraction


def test_cubic_autocatalysis_follows_its_mass_action():
    """Test that `A + 2B -> 3B` changes the substrate by `-k a b^2` and keeps the total."""
    reaction = CubicAutocatalysisInteraction('autocatalysis', target='reactor', substrate='v', autocatalyst='u', k=0.5)
    reactor = BrusselatorReactionElement('reactor', u=1.0, v=1.0)
    simulation = Simulation(System(elements=[reactor], interactions=[reaction]), dt=1e-4).prepare()
    states = simulation.run(until=1e-4).trajectory_.states
    np.testing.assert_allclose((states[-1] - states[0]) / 1e-4, [0.5, -0.5], rtol=1e-2)
    np.testing.assert_allclose(states.sum(axis=1), 2.0)


def test_cubic_autocatalysis_reacts_at_every_site_of_a_field():
    """Test that the reaction on a field reacts at every site with the states of that site."""
    values = {'u': np.array([0.0, 1.0, 2.0]), 'v': 1.0}
    cells = FieldElement('cells', element=BrusselatorReactionElement, space=LatticeSpace((3,)), values=values)
    reaction = CubicAutocatalysisInteraction('autocatalysis', target='cells', substrate='v', autocatalyst='u')
    simulation = Simulation(System(elements=[cells], interactions=[reaction]), dt=1e-4).prepare()
    trajectory = simulation.run(until=1e-4).trajectory_
    change = (trajectory.lattice('cells.v')[-1] - 1.0) / 1e-4
    np.testing.assert_allclose(change, -values['u'] ** 2, rtol=1e-3, atol=1e-9)
