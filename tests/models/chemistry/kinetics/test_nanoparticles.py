"""Test the `_nanoparticles` module of `skcomplex.models.chemistry.kinetics`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.chemistry.kinetics import nanoparticle_corona

PROTEINS = ('albumin', 'lipoprotein', 'fibrinogen')


def test_nanoparticle_corona_shows_the_vroman_effect():
    """Test that albumin covers the nanoparticles first and that lipoprotein replaces it, conserving every species."""
    solver = ScipyODESolver(method='LSODA', rtol=1e-10, atol=1e-14)
    trajectory = Simulation(nanoparticle_corona(), dt=60.0, solver=solver).check().run(until=36000.0).trajectory_
    free = trajectory.values('nanoparticle.n')
    bound = {name: trajectory.values(f'{name}_corona.n') for name in PROTEINS}
    assert np.allclose(free + sum(bound.values()), 36.0, rtol=1e-8)
    for name, initial in zip(PROTEINS, (6e5 + 14.0, 1.5e4 + 4.0, 8.8e3), strict=True):
        assert np.allclose(trajectory.values(f'{name}.n') + bound[name], initial, rtol=1e-10)
    assert bound['albumin'][1] > bound['lipoprotein'][1]
    assert bound['lipoprotein'][-1] > 25 * bound['albumin'][-1]
    sites = {'albumin': 4 * 40**2 / 5**2, 'lipoprotein': 4 * 39**2 / 4**2, 'fibrinogen': 4 * 43.3**2 / 8.3**2}
    rates = {'albumin': (2.4e3, 2e-3), 'lipoprotein': (3e4, 3e-5), 'fibrinogen': (2e3, 2e-3)}
    for name in PROTEINS:
        association, dissociation = rates[name]
        equilibrium = sites[name] * association * 1e-9 * free[-1] * trajectory.values(f'{name}.n')[-1] / dissociation
        assert abs(bound[name][-1] / equilibrium - 1) < 1e-3
