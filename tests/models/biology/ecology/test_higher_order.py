"""Test the `_higher_order` module of `skcomplex.models.biology.ecology`."""

import numpy as np
import pytest
from scipy.special import comb

from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.experiments import Experiment, RowsDesign
from skcomplex.models.biology.ecology import higher_order_random_lotka_volterra

from ..conftest import SYSTEMS_PRECISE, _run_with_solver

SURVIVING_FRACTION = 0.9276
MEAN_ABUNDANCE = 0.3924
FRACTION_PRECISION = 0.02
ABUNDANCE_PRECISION = 0.01
EXTINCT = 1e-6


def test_higher_order_random_lotka_volterra_tensor_has_the_moments_of_the_paper():
    """Test that the coefficients have the moments of Sidhom and Galla, Eq. 4, and 0 on repeated indices."""
    size = 60
    tensor = higher_order_random_lotka_volterra(size, mu=-4.0, sigma=1.0, gamma=0.0, random_state=0).interactions[1].a
    i, j, k = np.indices(tensor.shape)
    distinct = (i != j) & (i != k) & (j != k)
    pairs = comb(size - 1, 2)
    assert np.all(tensor[~distinct] == 0)
    np.testing.assert_array_equal(tensor, tensor.transpose(0, 2, 1))
    assert tensor[distinct].mean() == pytest.approx(-4.0 / pairs, abs=4 * np.sqrt(1.5 / pairs / distinct.sum()))
    assert tensor[distinct].var() == pytest.approx(1.5 / pairs, rel=0.02)


@pytest.mark.parametrize('gamma', [-0.5, 0.0, 1.0])
def test_higher_order_random_lotka_volterra_tensor_has_the_correlation_of_the_paper(gamma):
    """Test that the coefficients of a set of three species have the correlation of Sidhom and Galla, Eq. 5."""
    size = 40
    tensor = higher_order_random_lotka_volterra(size, mu=0.0, sigma=1.0, gamma=gamma, random_state=0).interactions[1].a
    i, j, k = np.indices(tensor.shape)
    ordered = (i < j) & (j < k)
    correlation = np.corrcoef(tensor[i[ordered], j[ordered], k[ordered]], tensor[j[ordered], i[ordered], k[ordered]])
    assert correlation[0, 1] == pytest.approx(gamma, abs=0.02)


def test_higher_order_random_lotka_volterra_draws_repeat_with_a_seed():
    """Test that one seed repeats the tensor and the initial abundances, and another seed changes them."""
    first = higher_order_random_lotka_volterra(10, random_state=3)
    again = higher_order_random_lotka_volterra(10, random_state=3)
    other = higher_order_random_lotka_volterra(10, random_state=4)
    np.testing.assert_array_equal(first.interactions[1].a, again.interactions[1].a)
    np.testing.assert_array_equal(first.elements[0].values['n'], again.elements[0].values['n'])
    assert not np.array_equal(first.interactions[1].a, other.interactions[1].a)


def test_higher_order_random_lotka_volterra_without_interactions_reaches_the_capacity():
    """Test that with a tensor of zeros every species converges to its carrying capacity within 1e-6."""
    system = higher_order_random_lotka_volterra(8, mu=0.0, sigma=0.0, k=1.5)
    assert np.all(system.interactions[1].a == 0)
    final = _run_with_solver(system, 40.0, 1.0, SYSTEMS_PRECISE).values('species.n')[-1]
    np.testing.assert_allclose(final, 1.5, atol=1e-6)


def test_higher_order_random_lotka_volterra_tensor_changes_by_an_experiment():
    """Test that an experiment sets the tensor parameter, so two tensors give different final abundances."""
    system = higher_order_random_lotka_volterra(8)
    rows = [{'triplets.a': np.zeros((8, 8, 8))}, {'triplets.a': -np.ones((8, 8, 8))}]
    experiment = Experiment(system, RowsDesign(rows), until=20.0, dt=20.0, outputs=['species.n'])
    final = experiment.check().prepare().run().result_.series('species.n')[:, -1]
    np.testing.assert_allclose(final[0], 1.0, atol=1e-3)
    assert np.all(final[1] < final[0])


@pytest.mark.slow
def test_higher_order_random_lotka_volterra_random_community_matches_the_paper():
    """Test that 20 seeds stay finite, with the surviving fraction and mean abundance of the paper's theory."""
    drawn = [higher_order_random_lotka_volterra(random_state=seed) for seed in range(20)]
    rows = [{'triplets.a': system.interactions[1].a, 'species.n': system.elements[0].values['n']} for system in drawn]
    solver = ScipyODESolver('RK45', rtol=1e-6, atol=1e-9)
    experiment = Experiment(drawn[0], RowsDesign(rows), until=1000.0, dt=1000.0, solver=solver, outputs=['species.n'])
    final = experiment.check().prepare().run().result_.series('species.n')[:, -1]
    assert np.all(np.isfinite(final))
    assert np.all(final >= 0)
    assert np.mean(final > EXTINCT) == pytest.approx(SURVIVING_FRACTION, abs=FRACTION_PRECISION)
    assert np.mean(final) == pytest.approx(MEAN_ABUNDANCE, abs=ABUNDANCE_PRECISION)
