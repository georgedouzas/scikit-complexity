"""Test the `_genes` module of `skcomplex.systems.biology.systems_biology`."""

import numpy as np

from skcomplex.core.analysis import correlation
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.systems_biology import (
    NegativeAutoregulationSystem,
    RepressilatorSystem,
    ToggleSwitchSystem,
)

from ..conftest import TOLERANCE


def _autocorrelation_period(series, dt):
    """Return the period of a noisy oscillation, the lag of the first autocorrelation peak after its first zero."""
    values = correlation(np.asarray(series) - np.mean(series)).correlation[0]
    first_negative = int(np.argmax(values < 0))
    window = values[first_negative : first_negative + len(values) // 2]
    return (first_negative + int(np.argmax(window))) * dt


def test_repressilator_oscillates_as_equations_and_jumps():
    """Test that the repressilator oscillates as ODEs and as a jump process, with periods within 10%."""
    periods = []
    for stochastic in (False, True):
        system = RepressilatorSystem(stochastic=stochastic).prepare().system_
        trajectory = simulate(system, dt=0.5, random_state=1, until=250)
        protein = trajectory.get_values('gene_0.p')[trajectory.times > 50]
        assert protein.max() > 3 * max(protein.min(), 1.0)
        periods.append(_autocorrelation_period(protein, 0.5))
    deterministic, stochastic_period = periods
    assert abs(stochastic_period - deterministic) <= TOLERANCE * deterministic


def test_toggle_switch_remembers_its_initial_state():
    """Test that the toggle switch settles to opposite states from opposite initial conditions."""
    finals = []
    for proteins in ((60.0, 0.0), (0.0, 60.0)):
        state = simulate(ToggleSwitchSystem(proteins), dt=1.0, until=100)
        finals.append((state.get_values('gene_0.p')[-1], state.get_values('gene_1.p')[-1]))
    (first_high, first_low), (second_low, second_high) = finals
    assert first_high > 10 * first_low
    assert second_high > 10 * second_low


def test_negative_autoregulation_is_bursty():
    """Test that bursts make the protein noise super-Poissonian and that feedback lowers the mean."""
    statistics = []
    for K in (10.0, 1e9):
        trajectory = simulate(NegativeAutoregulationSystem(K=K), dt=1.0, random_state=0, until=400)
        protein = trajectory.get_values('gene.p')[trajectory.times > 100]
        statistics.append((protein.mean(), protein.var() / protein.mean()))
    (regulated_mean, regulated_fano), (free_mean, free_fano) = statistics
    assert regulated_mean < 0.7 * free_mean
    assert free_fano > 3
    assert regulated_fano > 1
