"""Test the `types` module of `skcomplex`."""

import typing

from skcomplex.core.types import Preference


def test_preferences_score_two_agents():
    """Test that a preference is a function of two agents that returns a score."""
    assert typing.get_args(Preference.__value__)[-1] is float
