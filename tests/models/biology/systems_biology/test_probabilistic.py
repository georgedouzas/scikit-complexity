"""Test the `_probabilistic` module of `skcomplex.models.biology.systems_biology`."""

import numpy as np

from ...conftest import _series


def test_probabilistic_boolean_network_draws_functions_with_their_probabilities():
    """Test that a node with two functions of probabilities 0.7 and 0.3 uses the first in 70 percent of the steps."""
    network = {
        'x1': (((0, 1, 1, 1, 0, 1, 1, 1), 0.7), ((0, 0, 0, 1, 0, 1, 1, 1), 0.3)),
        'x2': (((0, 1, 1, 0, 0, 1, 0, 1), 1.0),),
        'x3': (((0, 0, 0, 1, 0, 1, 1, 1), 0.5), ((0, 0, 0, 0, 0, 0, 0, 1), 0.5)),
    }
    used = _series('probabilistic_boolean_network', 20000, network=network)['x1.function'][1:]
    error = np.sqrt(0.7 * 0.3 / used.size)
    assert abs(np.mean(used == 0) - 0.7) < 3 * error
