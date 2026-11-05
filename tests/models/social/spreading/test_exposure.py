"""Test the `_exposure` module of `skcomplex.models.social.spreading`."""

import numpy as np
import pytest

from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.models.social.spreading import hypergraph_exposure_sis

from ..conftest import _run


def _prevalence(**settings) -> np.ndarray:
    """Return the share of infected participants at every frame of a run of 400 steps."""
    return _run(hypergraph_exposure_sis(**settings), 400, dt=1).values('participants.infected').mean(axis=1)


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_hypergraph_exposure_sis_infects_with_the_probability_of_the_doses(backend):
    """Test that the share of runs in which a participant is infected after a step follows its doses."""
    system = hypergraph_exposure_sis(60, share=0.4, rate=0.3, exponent=1.5, seed=2)
    participants = system.elements[0]
    infected = participants.values['infected']
    doses = np.zeros(60)
    for hyperedge in participants.space.prepare().hyperedges_:
        members = list(hyperedge)
        doses[members] += 0.3 * ((infected[members].sum() - infected[members]) / (len(members) - 1)) ** 1.5
    expected = np.where(infected == 1, 0.95, 1 - np.exp(-doses))
    runs = 200
    hits = np.zeros(60)
    for seed in range(runs):
        simulation = Simulation(system, random_state=seed, dt=1, backend=backend).prepare().run(until=1)
        hits += simulation.trajectory_.values('participants.infected')[1]
    error = np.sqrt(np.maximum(expected * (1 - expected), 0.01) / runs)
    assert np.all(np.abs(hits / runs - expected) < 4 * error)


def test_hypergraph_exposure_sis_is_bistable_with_a_supralinear_kernel():
    """Test that with a supralinear kernel the epidemic dies out from a small seed and persists from a large one."""
    assert _prevalence(share=0.01)[-1] == 0
    assert _prevalence(share=0.5)[-50:].mean() > 0.5


def test_hypergraph_exposure_sis_has_one_endemic_state_with_a_linear_kernel():
    """Test that with a linear kernel a small and a large seed reach the same endemic prevalence."""
    small = _prevalence(share=0.01, exponent=1.0, rate=0.03, sizes=(5, 10, 15))
    large = _prevalence(share=0.5, exponent=1.0, rate=0.03, sizes=(5, 10, 15))
    assert small[-50:].mean() > 0.4
    assert abs(small[-50:].mean() - large[-50:].mean()) < 0.05
