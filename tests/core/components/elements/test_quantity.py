"""Test the `_quantity` module of `skcomplex.core.components.elements`."""

import pytest

from skcomplex.core.components.elements import Quantity
from skcomplex.core.errors import ValidationError


def test_quantity_holds_its_name():
    """Test that a quantity holds the name it is given."""
    assert Quantity('energy').name == 'energy'


@pytest.mark.parametrize('name', ['', 3])
def test_quantity_rejects_a_name_that_is_not_a_nonempty_string(name):
    """Test that the check names the expected form of a quantity."""
    with pytest.raises(ValidationError, match=r"Quantity\('energy'\)"):
        Quantity(name)._check()
