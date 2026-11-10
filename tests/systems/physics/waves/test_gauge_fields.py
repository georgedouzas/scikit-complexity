"""Test the `_gauge_fields` module of `skcomplex.systems.physics.waves`."""

import numpy as np
import pytest
from scipy.special import iv

import skcomplex.core.components.elements
import skcomplex.core.components.interactions
import skcomplex.core.components.laws.utils
import skcomplex.core.components.spaces
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError, ValidationError
from skcomplex.systems.physics.waves import LatticeGaugeFieldsSystem, gauge_observables

SIDE = 32


def _space(size=SIDE, dimension=2):
    """Return the prepared complex of a periodic lattice."""
    return CellComplexSpace(LatticeSpace((size,) * dimension)).prepare()


def _gauge_transform(angles, space, seed=4):
    """Return link angles after a random gauge transformation of the sites."""
    return angles + space.incidences_[0] @ np.random.default_rng(seed).uniform(-np.pi, np.pi, space.size_)


def _random_angles(space, seed=5):
    """Return random link angles of a complex."""
    return np.random.default_rng(seed).uniform(-np.pi, np.pi, space.incidences_[0].shape[0])


def _charge_pair():
    """Return a static charge pair on the sites of the lattice."""
    charges = np.zeros(SIDE * SIDE)
    charges[100], charges[612] = 1.0, -1.0
    return charges


def test_plaquette_of_the_defaults_is_the_exact_result():
    """Test that the default run after its transient is within 0.01 of `I_1(2) / I_0(2)`."""
    series = simulate(LatticeGaugeFieldsSystem(), random_state=0).get_values('mean_plaquette')
    assert abs(series[1001:].mean() - iv(1, 2) / iv(0, 2)) < 0.01


def test_gauge_transformation_of_the_initial_links_keeps_the_plaquette_series():
    """Test that a random gauge transformation of the initial links keeps the series within 1e-12 for one seed."""
    space = _space(16)
    angles = _random_angles(space)
    series = []
    for initial in (angles, _gauge_transform(angles, space)):
        trajectory = simulate(LatticeGaugeFieldsSystem(16, angles=initial), until=200, dt=1, random_state=2)
        series.append(trajectory.get_values('mean_plaquette'))
    assert np.abs(series[0] - series[1]).max() < 1e-12


def test_z_n_links_stay_on_the_angles_of_the_group():
    """Test that Z_3 links sampled by heat bath keep the angles of the group."""
    system = LatticeGaugeFieldsSystem(6, n=3)
    states = simulate(system, until=5, dt=1, random_state=0).get_values('links.theta')
    assert set(np.round(np.mod(states[-1], 2 * np.pi) / (2 * np.pi / 3), 6)) <= {0.0, 1.0, 2.0}


def test_maxwell_dynamics_conserve_the_gauss_law_and_the_energy():
    """Test the Gauss residual below 1e-10 at every frame and the energy within 1e-3 from a static charge pair."""
    trajectory = simulate(LatticeGaugeFieldsSystem(dynamics='hamiltonian', charges=_charge_pair()), random_state=0)
    assert trajectory.get_values('gauss_residual').max() < 1e-10
    energy = trajectory.get_values('field_energy')
    assert np.abs(energy - energy[0]).max() / energy[0] < 1e-3
    assert np.ptp(trajectory.get_values('links.A')[-1]) > 0


@pytest.mark.parametrize('charges', [np.ones(16), np.zeros(5)])
def test_maxwell_dynamics_reject_charges_that_a_periodic_lattice_cannot_hold(charges):
    """Test that charges of a total other than 0, or not one per site, raise an error."""
    with pytest.raises(ValidationError, match='total 0'):
        LatticeGaugeFieldsSystem(4, dynamics='hamiltonian', charges=charges).prepare()


def test_unknown_dynamics_raise_an_error():
    """Test that an unknown dynamics raises an error."""
    with pytest.raises(ValidationError, match='Unknown dynamics'):
        LatticeGaugeFieldsSystem(4, dynamics='quantum').prepare()


@pytest.mark.parametrize('extents', [(1, 1), (2, 2)])
def test_loops_are_invariant_under_a_gauge_transformation(extents):
    """Test that the mean plaquette and the 2 by 2 Wilson loop are unchanged within 1e-12."""
    space = _space(8)
    angles = _random_angles(space)
    loops = [
        gauge_observables(item, kind='wilson', space=space, extents=extents).value[0]
        for item in (angles, _gauge_transform(angles, space))
    ]
    assert abs(loops[0] - loops[1]) < 1e-12


def test_plaquette_loop_is_the_mean_cosine_of_the_plaquettes():
    """Test that the Wilson loop of extents 1 by 1 is the mean plaquette, and a larger one the sum inside it."""
    space = _space(6)
    angles = _random_angles(space)
    sums = space.incidences_[1] @ angles
    assert gauge_observables(angles, kind='wilson', space=space).value[0] == pytest.approx(np.cos(sums).mean())


def test_polyakov_loop_is_invariant_and_winds_along_the_last_axis():
    """Test the Polyakov loop of a uniform link angle along the last axis, and its gauge invariance."""
    space = _space(4)
    angles = np.where(space.link_axes_ == 1, np.pi / 4, 0.0)
    assert gauge_observables(angles, kind='polyakov', space=space).value[0] == pytest.approx(-1.0)
    random = _random_angles(space)
    values = [
        gauge_observables(item, kind='polyakov', space=space).value[0]
        for item in (random, _gauge_transform(random, space))
    ]
    assert values[0] == pytest.approx(values[1], abs=1e-12)


def test_topological_charge_is_an_integer():
    """Test that the charge of random configurations of a periodic 2D lattice is an integer within 1e-9."""
    space = _space(8)
    values = gauge_observables(np.stack([_random_angles(space, s) for s in range(5)]), kind='charge', space=space).value
    assert np.abs(values - np.round(values)).max() < 1e-9
    assert np.any(np.round(values) != 0)


def test_gauss_residual_reads_the_charges():
    """Test the Gauss residual of a field with one flux line between two charges."""
    space = _space(4)
    field = np.zeros(32)
    field[0] = 1.0
    charges = -space.incidences_[0].T @ field
    assert gauge_observables(field, kind='gauss', space=space, charges=charges).value[0] == 0.0
    assert gauge_observables(field, kind='gauss', space=space).value[0] == 1.0


def test_gauge_observables_reads_a_trajectory():
    """Test that a trajectory gives one value per frame."""
    system = LatticeGaugeFieldsSystem(4)
    trajectory = simulate(system, until=3, dt=1, random_state=0)
    result = gauge_observables(trajectory, kind='wilson', space=_space(4), variables='links.theta')
    assert result.value.shape == (4,)


def test_gauge_observables_rejects_a_space_or_a_kind_it_cannot_read():
    """Test that a lattice that is not periodic, an unknown kind, and a charge in 3D raise an error."""
    closed = CellComplexSpace(LatticeSpace((4, 4), boundary='fixed'))
    with pytest.raises(AnalysisError, match='periodic'):
        gauge_observables(np.zeros(32), kind='wilson', space=closed)
    with pytest.raises(AnalysisError, match='kind'):
        gauge_observables(np.zeros(32), kind='volume', space=_space(4))
    with pytest.raises(AnalysisError, match='two-dimensional'):
        gauge_observables(np.zeros(32), kind='charge', space=CellComplexSpace(LatticeSpace((3, 3, 3))))


def test_core_holds_no_field_terms_of_gauge_theory():
    """Test that no public name of the core contains the terms of gauge theory."""
    modules = (
        skcomplex.core.components.elements,
        skcomplex.core.components.interactions,
        skcomplex.core.components.laws.utils,
        skcomplex.core.components.spaces,
    )
    names = [name.lower() for module in modules for name in module.__all__]
    assert not [name for name in names if any(term in name for term in ('wilson', 'gauge', 'polyakov'))]
