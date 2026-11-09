"""Test the variants of the general interactions that take names and functions, or names alone."""

import pytest

from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PartnerSpec


def test_variant_refuses_a_spec_and_lists_what_it_takes():
    """Test that a move refuses a spec, and lists its names and functions."""
    interaction = SamplingInteraction('a', target='b', state='x', objective=lambda s: 0.0, move=PartnerSpec())
    with pytest.raises(ValidationError, match=r'Unknown move .*a function'):
        interaction.prepare()
