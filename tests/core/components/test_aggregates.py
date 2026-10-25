"""Test the `_aggregates` module of `skcomplex.core.components`."""

import pytest

from skcomplex.core.base import ValidationError, symbol
from skcomplex.core.components import AggregateQuantity
from skcomplex.core.system import System
from skcomplex.models.biology.ecology import PopulationComponent


def test_aggregate_counts_by_default():
    """Test that an aggregate counts the members of its group unless it names a reduction."""
    aggregate = AggregateQuantity('size', group='farmers').check()
    assert (aggregate.reduce, aggregate.variable, aggregate.expression) == ('count', None, None)


def test_aggregate_without_group_is_rejected():
    """Test that an aggregate that names no group raises a named error that names it."""
    with pytest.raises(ValidationError, match="'size' names no group"):
        AggregateQuantity('size').check()


def test_aggregate_with_expression_is_rejected():
    """Test that an aggregate with an expression raises a named error that asks for a variable."""
    with pytest.raises(ValidationError, match='Give `variable`'):
        AggregateQuantity('total', symbol('n'), group='herd').check()


def test_aggregate_is_a_component_of_its_system_but_not_an_element():
    """Test that a system lists its aggregates apart from its elements."""
    aggregate = AggregateQuantity('total', group='herd', reduce='sum', variable='n')
    system = System(components=[PopulationComponent.group('herd', 3), aggregate])
    assert system.aggregates == (aggregate,)
    assert [element.label for element in system.elements] == ['herd']
