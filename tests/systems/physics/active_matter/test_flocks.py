"""Test the `_flocks` module of `skcomplex.systems.physics.active_matter`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ParameterError
from skcomplex.systems.physics.active_matter import ginelli_nematics, gregoire_chate, vicsek

from ..conftest import _run_observed


def test_vicsek_flocks_at_low_noise_and_disorders_at_high_noise():
    """Test that the polar order of the Vicsek system exceeds 0.9 at low noise and stays small at high noise."""
    ordered = np.asarray(_run_observed('vicsek', vicsek(300), 150.0, 10.0).values('polarization'))
    disordered = np.asarray(
        _run_observed('vicsek', vicsek(300, eta=0.9), 150.0, 10.0).values('polarization'),
    )
    assert ordered[0] < 0.2
    assert ordered[-1] > 0.9
    assert disordered[-3:].mean() < 0.3


def test_gregoire_chate_forms_an_aligned_flock():
    """Test that cohesive particles at low noise align into a flock with a polar order above 0.8."""
    polarization = np.asarray(
        _run_observed('gregoire_chate', gregoire_chate(), 300.0, 10.0).values('polarization'),
    )
    assert polarization[0] < 0.2
    assert polarization[-5:].mean() > 0.8


def test_ginelli_nematics_orders_an_axis_without_a_direction():
    """Test that nematic alignment orders the axes of the particles while they keep moving both ways along it."""
    trajectory = _run_observed('ginelli_nematics', ginelli_nematics(), 200.0, 10.0)
    nematic, polar = np.asarray(trajectory.values('nematic_order')), np.asarray(trajectory.values('polarization'))
    assert nematic[-1] > 0.85
    assert polar[-1] < 0.8


def _neighbors_within_unit_distance(system, until):
    """Return the mean number of other particles within the unit distance over the last frames of a run."""
    trajectory = Simulation(system, dt=50.0, random_state=0).prepare().run(until=until).trajectory_
    size = system.space.extent[0]
    counts = []
    for positions in np.asarray(trajectory.values('flock.x'))[-5:]:
        gaps = np.abs(positions[:, None, :] - positions[None, :, :])
        distances = np.hypot(*np.moveaxis(np.minimum(gaps, size - gaps), -1, 0))
        counts.append(((distances < 1).sum(axis=1) - 1).mean())
    return float(np.mean(counts))


def test_topological_vicsek_orders_more_with_more_neighbours():
    """Test that aligning with the 8 nearest particles orders a sparse flock more than aligning with the nearest one."""
    orders = [
        np.asarray(
            _run_observed('vicsek', vicsek(100, size=100.0, v0=1.0, neighbors=k), 200.0, 10.0).values('polarization'),
        )[-5:].mean()
        for k in (1, 8)
    ]
    assert orders[1] > 0.9
    assert orders[1] > orders[0] + 0.3


def test_topological_neighbourhood_is_not_reciprocal():
    """Test that the nearest neighbourhoods of the initial flock contain pairs that are not mutual, Ex.

    8.10.
    """
    positions = vicsek(100, size=100.0, neighbors=1).elements[0].values['x']
    gaps = np.abs(positions[:, None, :] - positions[None, :, :])
    distances = np.hypot(*np.moveaxis(np.minimum(gaps, 100.0 - gaps), -1, 0))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.argmin(axis=1)
    assert (nearest[nearest] != np.arange(100)).any()


def test_blind_vicsek_flock_stays_disordered():
    """Test that a vision cone too narrow to see anyone leaves the flock as disordered as its noise."""
    seeing = _run_observed('vicsek', vicsek(300, vision=np.pi), 150.0, 10.0).values('polarization')
    blind = _run_observed('vicsek', vicsek(300, vision=1e-6), 150.0, 10.0).values('polarization')
    assert np.asarray(seeing)[-1] > 0.9
    assert np.asarray(blind)[-3:].mean() < 0.3


def test_long_and_negative_delays_hinder_clustering():
    """Test that a delay of 25 steps and a prediction 2 steps ahead leave fewer neighbours than no delay, Fig.

    8.8-8.9.
    """
    settings = {'n_particles': 100, 'size': 50.0, 'v0': 0.15, 'eta': 0.2 / np.pi}
    counts = {delay: _neighbors_within_unit_distance(vicsek(**settings, delay=delay), 1500.0) for delay in (0, 25, -2)}
    assert counts[25] < counts[0] / 2
    assert counts[-2] < counts[0] / 2


def test_vicsek_refuses_delays_beyond_its_memory_and_combined_variants():
    """Test that a delay beyond the kept headings and a cone with nearest neighbours are refused."""
    with pytest.raises(ParameterError):
        vicsek(10, delay=26)
    with pytest.raises(ParameterError):
        vicsek(10, neighbors=4, vision=1.0)
