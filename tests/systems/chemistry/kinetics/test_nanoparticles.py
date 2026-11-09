"""Test the `_nanoparticles` module of `skcomplex.systems.chemistry.kinetics`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.chemistry.kinetics import NanoparticleCoronaSystem, SpeciesElement
from skcomplex.systems.general.elements import NonnegativeElement

PROTEINS = ('albumin', 'lipoprotein', 'fibrinogen')


def test_nanoparticle_corona_shows_the_vroman_effect():
    """Test that albumin covers the nanoparticles first and that lipoprotein replaces it, conserving every species."""
    solver = ScipyODESolver(method='LSODA', rtol=1e-10, atol=1e-14)
    trajectory = simulate(NanoparticleCoronaSystem(), until=36000.0, solver=solver)
    free = trajectory.get_values('nanoparticle.x')
    bound = {name: trajectory.get_values(f'{name}_corona.x') for name in PROTEINS}
    assert np.allclose(free + sum(bound.values()), 36.0, rtol=1e-8)
    for name, initial in zip(PROTEINS, (6e5 + 14.0, 1.5e4 + 4.0, 8.8e3), strict=True):
        assert np.allclose(trajectory.get_values(f'{name}.x') + bound[name], initial, rtol=1e-10)
    assert bound['albumin'][1] > bound['lipoprotein'][1]
    assert bound['lipoprotein'][-1] > 25 * bound['albumin'][-1]
    sites = {'albumin': 4 * 40**2 / 5**2, 'lipoprotein': 4 * 39**2 / 4**2, 'fibrinogen': 4 * 43.3**2 / 8.3**2}
    rates = {'albumin': (2.4e3, 2e-3), 'lipoprotein': (3e4, 3e-5), 'fibrinogen': (2e3, 2e-3)}
    for name in PROTEINS:
        association, dissociation = rates[name]
        equilibrium = (
            sites[name] * association * 1e-9 * free[-1] * trajectory.get_values(f'{name}.x')[-1] / dissociation
        )
        assert abs(bound[name][-1] / equilibrium - 1) < 1e-3


def test_nanoparticle_corona_species_are_nonnegative_elements():
    """Test that every species of the corona is a chemical species, a realization of the core atom with state `x`."""
    elements = NanoparticleCoronaSystem().prepare().system_.elements
    assert all(isinstance(element, SpeciesElement) for element in elements)
    assert all(isinstance(element, NonnegativeElement) for element in elements)
    assert [element.label for element in elements][:3] == ['nanoparticle', 'albumin', 'albumin_corona']
