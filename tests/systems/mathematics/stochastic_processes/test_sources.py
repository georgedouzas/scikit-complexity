"""Test the `_sources` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import numpy as np
import pytest

from skcomplex.core.analysis import entropy

from ...conftest import _series

TWO_STATE = ((0.9, 0.1), (0.4, 0.6))


def test_markov_source_has_the_entropy_rate_of_its_chain():
    """Test that the entropy rate estimated from block entropies is 0.569 bits."""
    symbols = _series('markov_source', 100000, transition=TWO_STATE).values('source.symbol')
    rate = entropy(symbols.astype(int), kind='rate', dimension=8).entropy[0]
    assert rate == pytest.approx(0.569, abs=0.01)


def test_hidden_markov_source_emits_at_the_stationary_frequency():
    """Test that the first symbol has the frequency `0.8 x 0.7 + 0.2 x 0.2 = 0.6`."""
    symbols = _series('hidden_markov_source', 100000).values('source.symbol')
    assert np.mean(symbols == 0) == pytest.approx(0.6, abs=0.01)
