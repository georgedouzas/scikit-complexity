"""Test the `_quantities` module of `skcomplex.core.base`."""

import dataclasses

import numpy as np
import pytest

from skcomplex.core.base import BaseComponent, BaseLaw, BaseQuantity, ValidationError, symbol
from skcomplex.core.components import AggregateQuantity
from skcomplex.core.laws import AuxiliaryQuantity, InvariantQuantity, ObjectiveQuantity
from skcomplex.core.system import ObservableQuantity

QUANTITIES = (ObservableQuantity, AuxiliaryQuantity, InvariantQuantity, ObjectiveQuantity, AggregateQuantity)


@pytest.mark.parametrize('quantity', QUANTITIES, ids=lambda quantity: quantity.__name__)
def test_quantity_is_a_quantity_only(quantity):
    """Test that every quantity derives from the quantity base and from neither the law nor the component base."""
    assert issubclass(quantity, BaseQuantity)
    assert not issubclass(quantity, BaseLaw)
    assert not issubclass(quantity, BaseComponent)


@pytest.mark.parametrize('quantity', QUANTITIES, ids=lambda quantity: quantity.__name__)
def test_quantity_is_a_frozen_value_with_a_label_and_an_expression(quantity):
    """Test that every quantity is a frozen dataclass whose first fields are the label and the expression."""
    assert dataclasses.is_dataclass(quantity)
    assert quantity.__dataclass_params__.frozen
    assert [item.name for item in dataclasses.fields(quantity)][:2] == ['label', 'expression']


def test_check_returns_a_checked_copy():
    """Test that checking a quantity returns a quantity with its expression checked and leaves the original as is."""
    x = symbol('x')
    auxiliary = AuxiliaryQuantity('double', 2 * x)
    checked = auxiliary.check()
    assert checked == auxiliary
    assert BaseQuantity('plain', x).check() == BaseQuantity('plain', x)


def test_check_rejects_an_invalid_expression():
    """Test that checking a quantity whose expression is a NumPy array raises a named error that names it."""
    with pytest.raises(ValidationError, match='auxiliary double'):
        AuxiliaryQuantity('double', np.ones(2)).check()
