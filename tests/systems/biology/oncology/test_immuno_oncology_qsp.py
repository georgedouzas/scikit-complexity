"""Test the `_immuno_oncology_qsp` module of `skcomplex.systems.biology.oncology`."""

import numpy as np
from scipy.linalg import expm

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.oncology import ImmunoOncologyQspSystem


def test_treatment_shrinks_the_tumour():
    """Test that doses every two weeks for 180 days leave a smaller tumour than no doses from the same state."""
    treated = simulate(ImmunoOncologyQspSystem(), until=180.0, dt=1.0)
    untreated = simulate(ImmunoOncologyQspSystem(dose=0.0), until=180.0, dt=1.0)
    assert treated.get_values('diameter')[-1] < untreated.get_values('diameter')[-1]
    assert treated.get_values('diameter')[0] == untreated.get_values('diameter')[0]


def test_antibody_decays_in_two_exponential_phases():
    """Test that one dose without T cells decays as the matrix exponential of the two-compartment kinetics."""
    system = ImmunoOncologyQspSystem(effector=0.0, treg=0.0, recruitment=0.0, treg_recruitment=0.0, period=1000.0)
    trajectory = simulate(system, until=60.0, dt=0.5)
    times = trajectory.times
    amounts = np.stack([trajectory.get_values('central.x'), trajectory.get_values('peripheral.x')], axis=1)
    assert np.all(trajectory.get_values('effector.x') == 0.0)
    rates = np.asarray([[-(system.clearance + system.k12), system.k21], [system.k12, -system.k21]])
    after = np.flatnonzero(times >= 1.0)
    expected = [expm(rates * (times[index] - 1.0)) @ amounts[after[0]] for index in after]
    np.testing.assert_allclose(amounts[after], expected, rtol=1e-5)
    phases = np.sort(-np.linalg.eigvals(rates))
    assert phases[1] > 5 * phases[0]
    slopes = np.diff(np.log(amounts[after, 0])) / np.diff(times[after])
    assert slopes[0] < 2 * slopes[-1]
    assert np.isclose(slopes[-1], -phases[0], rtol=0.1)


def test_observables_give_the_diameter_and_the_ratio():
    """Test that the diameter is that of a sphere of the tumour volume, and the ratio that of the T cells."""
    trajectory = simulate(ImmunoOncologyQspSystem(cancer=2e8, effector=3e7, treg=1e7), until=1.0, dt=1.0)
    volume = 2e8 * 2.572e-9
    assert np.isclose(trajectory.get_values('diameter')[0], (6 * volume / np.pi) ** (1 / 3))
    assert np.isclose(trajectory.get_values('effector_treg_ratio')[0], 3e7 / (1e7 + 1))
