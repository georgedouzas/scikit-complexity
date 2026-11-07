"""Test the `_toxicity` module of `skcomplex.systems.biology.physiology`."""

import numpy as np

from skcomplex.systems.biology.physiology import DrugToxicitySystem

from ..conftest import _run_precise


def test_drug_toxicity_integrates_the_concentration():
    """Test that a cumulative toxicity is the integral of the concentration, and that a cleared one decays after."""
    cumulative = _run_precise(DrugToxicitySystem(eta=0.0, k=0.5, dose=1.0), until=84.0, dt=0.2)
    times = cumulative.times
    infused = np.clip(times[:, None] - 7.0 * np.arange(6), 0.0, 1.0).sum(axis=1)
    assert np.allclose(cumulative.values('patient.T'), (infused - cumulative.values('patient.c')) / 0.5, atol=1e-6)
    cleared = _run_precise(DrugToxicitySystem(eta=0.05), until=84.0, dt=0.2)
    late = cleared.times >= 60.0
    assert np.all(np.diff(cleared.values('patient.T')[late]) < 0)
