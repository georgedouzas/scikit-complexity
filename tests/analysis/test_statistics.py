"""Test distribution statistics and diffusion measures."""

import warnings

import networkx as nx
import numpy as np
import pytest

from skcomplex.analysis import (
    ccdf,
    degrees,
    fit_power_law,
    gini,
    hill,
    histogram,
    mean_squared_displacement,
)
from skcomplex.core.errors import AnalysisError
from skcomplex.core.spaces import NetworkSpace


def test_power_law_exponent_recovered():
    """Test that 10,000 samples of a power law with exponent 2.5 above 1 give 2.5 within 0.1 and p above 0.1."""
    samples = (1 - np.random.default_rng(1).random(10_000)) ** (-1 / 1.5)
    fit = fit_power_law(samples, bootstrap=50, random_state=0)
    assert fit.alpha == pytest.approx(2.5, abs=0.1)
    assert fit.p_value > 0.1
    ratio, p_value = fit.comparisons['exponential']
    assert ratio > 0
    assert p_value < 0.01


def test_exponential_sample_is_not_a_power_law():
    """Test that an exponential sample loses the comparison with an exponential tail."""
    samples = np.random.default_rng(2).exponential(5.0, 5000) + 1
    assert fit_power_law(samples, xmin=1.0, bootstrap=0).comparisons['exponential'][0] < 0


def test_fit_power_law_silences_powerlaw_warnings():
    """Test that the fits of `powerlaw` warn nothing, and that the silencing ends with the call."""
    rng = np.random.default_rng(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit_power_law(np.floor((1 - rng.random(3000)) ** (-1 / 1.5)), discrete=True, bootstrap=0)
        fit_power_law(rng.exponential(size=2000), bootstrap=0)
        assert caught == []
        warnings.warn('after the fit', UserWarning, stacklevel=1)
    assert [str(warning.message) for warning in caught] == ['after the fit']


def test_discrete_fit_of_preferential_attachment():
    """Test that the degrees of a Barabási-Albert graph have a tail exponent within 0.3 of 3."""
    degrees = np.array([degree for _, degree in nx.barabasi_albert_graph(10_000, 2, seed=0).degree()])
    assert fit_power_law(degrees, discrete=True, bootstrap=0).alpha == pytest.approx(3, abs=0.3)


def test_gini_and_hill():
    """Test the Gini coefficient of equal and concentrated values and the Hill tail index."""
    assert gini([5, 5, 5, 5]) == 0
    assert gini([0, 0, 0, 1]) == pytest.approx(0.75)
    samples = (1 - np.random.default_rng(3).random(50_000)) ** (-1 / 1.5)
    assert hill(samples, 5000) == pytest.approx(1.5, abs=0.1)
    with pytest.raises(AnalysisError, match='k = 0'):
        hill(samples, 0)


def test_histogram_and_ccdf():
    """Test that log-binned densities integrate to 1 and the survival function starts at 1."""
    samples = (1 - np.random.default_rng(4).random(1000)) ** (-1 / 1.5)
    edges, density = histogram(samples, log=True)
    assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)
    _, survival = ccdf(samples)
    assert survival[0] == 1.0
    assert np.all(np.diff(survival) < 0)


def test_brownian_motion_diffuses_normally():
    """Test that 1000 Brownian particles have a mean squared displacement exponent within 0.05 of 1."""
    walks = np.cumsum(np.random.default_rng(5).standard_normal((2000, 1000, 2)), axis=0)
    assert mean_squared_displacement(walks).exponent == pytest.approx(1.0, abs=0.05)


def test_barabasi_albert_degree_exponent():
    """Test that the degree distribution of a Barabási-Albert graph has an exponent within 0.3 of 3."""
    graph = nx.barabasi_albert_graph(10_000, 2, seed=0)
    assert fit_power_law(degrees(NetworkSpace(graph)), discrete=True, bootstrap=0).alpha == pytest.approx(3, abs=0.3)
