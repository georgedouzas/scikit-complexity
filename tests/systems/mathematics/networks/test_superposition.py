"""Test the `_superposition` module of `skcomplex.systems.mathematics.networks`."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import poisson

from skcomplex.core.errors import ValidationError
from skcomplex.systems.mathematics.networks import network_superposition


def test_network_superposition_mixes_the_poisson_degrees_of_its_subgraphs_as_eq_4_34():
    """Test that the degrees of superposed Erdős-Rényi subgraphs follow `P(k) = ∫ dp P(k | N p) Q(p)`, Eq. 4.34."""
    n_nodes, exponent, low, high = 100, 2.0, 0.005, 0.1
    system = network_superposition(n_nodes, n_subgraphs=1000, exponent=exponent, low=low, high=high)
    degrees = np.asarray(system.elements[0].values['degree'], dtype=int)
    norm = (low ** (1 - exponent) - high ** (1 - exponent)) / (exponent - 1)
    support = np.arange(degrees.max() + 30)
    expected = np.array(
        [quad(lambda p, k=k: poisson.pmf(k, (n_nodes - 1) * p) * p**-exponent / norm, low, high)[0] for k in support],
    )
    observed = np.bincount(degrees, minlength=support.size) / degrees.size
    assert expected.sum() == pytest.approx(1.0, abs=1e-3)
    assert 0.5 * np.abs(observed - expected).sum() < 0.05


def test_network_superposition_rejects_bounds_outside_the_unit_interval():
    """Test that linking probabilities outside `(0, 1]` raise an error."""
    with pytest.raises(ValidationError, match='0 < low'):
        network_superposition(10, low=0.0)
