"""Test the `_reaction_spec` module of `skcomplex.systems.biology.common.metabolism`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.common.metabolism import (
    ReactionSpec,
)


def test_reaction_spec_rejects_crossed_bounds():
    """Test that a lower bound above the upper bound raises an error."""
    with pytest.raises(ValidationError, match='above its upper bound'):
        ReactionSpec({'A': 1.0}, lower=2.0, upper=1.0).prepare()
