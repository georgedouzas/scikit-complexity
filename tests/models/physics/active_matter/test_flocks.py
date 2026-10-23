"""Test the `_flocks` module of `skcomplex.models.physics.active_matter`."""

import numpy as np

from skcomplex.models.physics.active_matter import ginelli_nematics, gregoire_chate, vicsek

from ..conftest import _run_observed


def test_vicsek_flocks_at_low_noise_and_disorders_at_high_noise():
    """Test that the polar order of the Vicsek model exceeds 0.9 at low noise and stays small at high noise."""
    ordered = np.asarray(_run_observed('vicsek', vicsek(300), 150.0, 10.0)['polarization'])
    disordered = np.asarray(
        _run_observed('vicsek', vicsek(300, eta=0.9), 150.0, 10.0)['polarization'],
    )
    assert ordered[0] < 0.2
    assert ordered[-1] > 0.9
    assert disordered[-3:].mean() < 0.3


def test_gregoire_chate_forms_an_aligned_flock():
    """Test that cohesive particles at low noise align into a flock with a polar order above 0.8."""
    polarization = np.asarray(
        _run_observed('gregoire_chate', gregoire_chate(), 300.0, 10.0)['polarization'],
    )
    assert polarization[0] < 0.2
    assert polarization[-5:].mean() > 0.8


def test_ginelli_nematics_orders_an_axis_without_a_direction():
    """Test that nematic alignment orders the axes of the particles while their polar order stays small."""
    trajectory = _run_observed('ginelli_nematics', ginelli_nematics(), 200.0, 10.0)
    nematic, polar = np.asarray(trajectory['nematic_order']), np.asarray(trajectory['polarization'])
    assert nematic[-1] > 0.85
    assert polar[-1] < 0.3
