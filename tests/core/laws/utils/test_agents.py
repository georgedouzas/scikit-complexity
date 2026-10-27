"""Test the `_agents` module of `skcomplex.core.laws.utils`."""

import numpy as np

from skcomplex.core.laws.utils import random_pairing


def test_random_pairing_within_one_group_avoids_self_pairs():
    """Test that random pairing within one group never pairs a member with itself."""
    members = list(range(9))
    pairs = random_pairing(members, members, np.random.default_rng(0), 2)
    assert all(first != second for first, second in pairs)
    assert len({frozenset(pair) for pair in pairs}) == len(pairs)
