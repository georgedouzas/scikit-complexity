"""Test the `_cubic_autocatalysis` module of `skcomplex.systems.chemistry.common.kinetics`."""

import numpy as np

from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System
from skcomplex.systems.chemistry.common.kinetics import BrusselatorReactionElement, CubicAutocatalysisInteraction


def test_cubic_autocatalysis_follows_its_mass_action():
    """Test that `A + 2B -> 3B` changes the substrate by `-k a b^2` and keeps the total."""
    reaction = CubicAutocatalysisInteraction('autocatalysis', target='reactor', substrate='v', autocatalyst='u', k=0.5)
    reactor = BrusselatorReactionElement('reactor', u=1.0, v=1.0)
    trajectory = System(elements=[reactor], interactions=[reaction]).prepare().simulate(until=1e-4, dt=1e-4)
    states = trajectory.states
    np.testing.assert_allclose((states[-1] - states[0]) / 1e-4, [0.5, -0.5], rtol=1e-2)
    np.testing.assert_allclose(states.sum(axis=1), 2.0)


def test_cubic_autocatalysis_reacts_at_every_site_of_a_field():
    """Test that the reaction on a field reacts at every site with the states of that site."""
    values = {'u': np.array([0.0, 1.0, 2.0]), 'v': 1.0}
    cells = BrusselatorReactionElement('cells', **values, space=LatticeSpace((3,)))
    reaction = CubicAutocatalysisInteraction('autocatalysis', target='cells', substrate='v', autocatalyst='u')
    trajectory = System(elements=[cells], interactions=[reaction]).prepare().simulate(until=1e-4, dt=1e-4)
    change = (trajectory.get_lattice('cells.v')[-1] - 1.0) / 1e-4
    np.testing.assert_allclose(change, -values['u'] ** 2, rtol=1e-3, atol=1e-9)
