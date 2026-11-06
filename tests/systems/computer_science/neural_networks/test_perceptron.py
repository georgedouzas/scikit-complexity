"""Test the `_perceptron` module of `skcomplex.systems.computer_science.neural_networks`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.systems.computer_science.neural_networks import perceptron_satisfiability

STEPS = 26


def _satisfiable_shares(n_weights: int) -> np.ndarray:
    """Return the share of 20 seeds that are satisfiable at every number of patterns."""
    bounds = {'n_weights': [n_weights], 'random_state': list(range(20))}
    outputs = ['decide.satisfiable']
    experiment = Experiment(perceptron_satisfiability, GridDesign(), bounds, until=STEPS, outputs=outputs)
    return experiment.prepare().run().result_.series('decide.satisfiable').mean(axis=0)


@pytest.mark.parametrize('n_weights', [15, 21])
def test_perceptron_satisfiability_satisfiable_fraction(n_weights: int) -> None:
    """Test that the share of satisfiable seeds falls from above 0.9 at the load 0.5 to below 0.1 at the load 1.2."""
    shares = _satisfiable_shares(n_weights)
    assert shares[round(0.5 * n_weights)] > 0.9
    assert shares[round(1.2 * n_weights)] < 0.1


def test_perceptron_satisfiability_even_weights() -> None:
    """Test that an even number of weights, whose patterns can tie, is refused."""
    with pytest.raises(ValidationError, match='n_weights'):
        perceptron_satisfiability(20)
