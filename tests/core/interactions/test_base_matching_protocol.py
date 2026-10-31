"""Test the `_base_matching_protocol` module of `skcomplex.core.interactions`."""

import numpy as np
import pytest

from skcomplex.core.interactions import BaseMatchingProtocol, MatchingRoundSpec, RandomSearchProtocol
from skcomplex.errors import ValidationError


class NoMatchingProtocol(BaseMatchingProtocol):
    """Protocol that never matches."""

    def _match(self, requesters, providers, rng, storage):
        """Return no matches."""
        return []


def test_run_raises_without_clearing():
    """Test that a protocol run without the offers of a clearing raises an error."""
    with pytest.raises(ValidationError, match='MatchingRoundSpec'):
        NoMatchingProtocol().check().prepare().run()


def test_run_sets_no_matches_for_an_empty_side():
    """Test that a clearing without offers sets no matches."""
    protocol = RandomSearchProtocol().check().prepare()
    assert protocol.run(MatchingRoundSpec([], [], np.random.default_rng(0), {})).trades_ == []


def test_check_rejects_an_unknown_option():
    """Test that an option outside the allowed ones raises an error naming them."""
    with pytest.raises(ValidationError, match='midpoint'):
        RandomSearchProtocol(clearing='bid').check()
