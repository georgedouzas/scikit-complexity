"""Test the `_annotated_element` module of `skcomplex.systems.general.dynamics`."""

from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from sklearn.base import clone

from skcomplex.core.components.elements import Quantity
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.general.dynamics import AnnotatedElement


def test_annotated_element_gives_the_hints_annotated_would_write():
    """Test that the allowed values and quantities of the values become their hints, in the order given."""
    nonnegative = RealSet.closed_open(0, oo)
    element = AnnotatedElement(
        'herd',
        values={'n': 4.0, 'status': 'S'},
        allowed={'n': nonnegative, 'status': {'S', 'I'}},
        quantities={'n': 'animals'},
    )
    hints = element._get_hints()
    assert hints == {
        'n': Annotated[object, nonnegative, Quantity('animals')],
        'status': Annotated[object, {'S', 'I'}],
    }


def test_annotated_element_keeps_its_allowed_values_when_cloned():
    """Test that a clone keeps the allowed values, which are no values of the element."""
    element = AnnotatedElement('tank', values={'level': 1.0}, allowed={'level': RealSet.closed_open(0, oo)})
    copy = clone(element)
    assert copy.allowed == element.allowed
    system = System(elements=[copy]).prepare()
    assert 'tank.allowed' not in system.parameters_


def test_annotated_element_rejects_a_category_it_does_not_allow():
    """Test that a categorical value outside its allowed set is rejected."""
    element = AnnotatedElement('herd', values={'status': 'X'}, allowed={'status': {'S', 'I'}})
    with pytest.raises(ValidationError, match='status'):
        System(elements=[element]).prepare()
