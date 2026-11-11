"""Test the `_wiener_process_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import WienerProcessSpec


@pytest.mark.parametrize(
    ('spec', 'message'),
    [(WienerProcessSpec(1), 'string'), (WienerProcessSpec('market', dimension=0), 'positive whole dimension')],
)
def test_wiener_process_spec_checks_its_label_and_dimension(spec, message):
    """Test that Wiener processes need a string label and a positive whole dimension."""
    with pytest.raises(ValidationError, match=message):
        spec._check()
