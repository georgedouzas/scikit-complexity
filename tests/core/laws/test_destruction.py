"""Test the `_destruction` module of `skcomplex.core.laws`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.laws import DestructionLaw
from skcomplex.errors import ValidationError

F, G, MU = var('f g mu')


def test_destruction_targets_the_candidate_and_runs_into_its_rate():
    """Test that a destruction targets the candidate and runs into its rate at the values of its parameters."""
    law = DestructionLaw(F, G, 2 * MU).check().prepare()
    assert law.targets_ == (F,)
    assert law.run({'mu': 0.1}) == pytest.approx(0.2)


def test_destruction_rejects_a_rate_of_wrong_shape():
    """Test that a rate that is neither a number nor a table of two dimensions is rejected."""
    with pytest.raises(ValidationError, match='0 or 2 dimensions'):
        DestructionLaw(F, G, np.ones(3)).check()
