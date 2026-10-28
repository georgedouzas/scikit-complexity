"""Test the `_genes` module of `skcomplex.models.biology.systems_biology`."""

import numpy as np

from skcomplex.analysis import CorrelationAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.systems_biology import (
    negative_autoregulation,
    repressilator,
    toggle_switch,
)

from ..conftest import TOLERANCE


def _autocorrelation_period(series, dt):
    """Return the period of a noisy oscillation, the lag of the first autocorrelation peak after its first zero."""
    correlation = CorrelationAnalysis().check().prepare(np.asarray(series) - np.mean(series)).run().correlation_[0]
    first_negative = int(np.argmax(correlation < 0))
    window = correlation[first_negative : first_negative + len(correlation) // 2]
    return (first_negative + int(np.argmax(window))) * dt


def test_repressilator_oscillates_as_equations_and_jumps():
    """Test that the repressilator oscillates as ODEs and as a jump process, with periods within 10%."""
    periods = []
    for stochastic in (False, True):
        system = repressilator(stochastic=stochastic)
        trajectory = Simulation(system, dt=0.5, random_state=1).run(until=250).trajectory_
        protein = trajectory.values('gene_0.p')[trajectory.times > 50]
        assert protein.max() > 3 * max(protein.min(), 1.0)
        periods.append(_autocorrelation_period(protein, 0.5))
    deterministic, stochastic_period = periods
    assert abs(stochastic_period - deterministic) <= TOLERANCE * deterministic


def test_toggle_switch_remembers_its_initial_state():
    """Test that the toggle switch settles to opposite states from opposite initial conditions."""
    finals = []
    for proteins in ((60.0, 0.0), (0.0, 60.0)):
        state = Simulation(toggle_switch(proteins), dt=1.0).run(until=100).trajectory_
        finals.append((state.values('gene_0.p')[-1], state.values('gene_1.p')[-1]))
    (first_high, first_low), (second_low, second_high) = finals
    assert first_high > 10 * first_low
    assert second_high > 10 * second_low


def test_negative_autoregulation_is_bursty():
    """Test that bursts make the protein noise super-Poissonian and that feedback lowers the mean."""
    statistics = []
    for K in (10.0, 1e9):
        trajectory = Simulation(negative_autoregulation(K=K), dt=1.0, random_state=0).run(until=400).trajectory_
        protein = trajectory.values('gene.p')[trajectory.times > 100]
        statistics.append((protein.mean(), protein.var() / protein.mean()))
    (regulated_mean, regulated_fano), (free_mean, free_fano) = statistics
    assert regulated_mean < 0.7 * free_mean
    assert free_fano > 3
    assert regulated_fano > 1
