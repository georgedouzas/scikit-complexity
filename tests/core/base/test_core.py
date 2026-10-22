"""Test the domains, declarations, and registry of the core package."""

import pytest

from skcomplex.core.base import Domain, Registry, VariableKind, declarations_of, parameter, role, state
from skcomplex.core.components import Element
from skcomplex.core.interactions import Interaction


class Body(Element):
    """Element with one scalar state, one vector state, and one parameter."""

    n: float = state(1.0, domain=Domain.NONNEGATIVE)
    x: list[float] = state([0.0, 0.0], vector=True)
    m: float = parameter(1.0, domain=Domain.POSITIVE)


class Link(Interaction):
    """Interaction with two roles."""

    first: str = role()
    second: str = role()


@pytest.mark.parametrize(
    ('domain', 'value', 'expected'),
    [
        (Domain.REAL, -1.5, True),
        (Domain.POSITIVE, 0.0, False),
        (Domain.NONNEGATIVE, 0.0, True),
        (Domain.INTEGER, 2.5, False),
        (Domain.NONNEGATIVE_INTEGER, [0, 3], True),
        (Domain.REAL, float('nan'), False),
        (Domain.REAL, 'm', False),
    ],
)
def test_domain_contains_checks_values(domain, value, expected):
    """Test that domains accept and reject values."""
    assert domain.contains(value) is expected


def test_domain_maps_to_sage_domain():
    """Test that domains give the Sage domains of their symbols, which have no nonnegative domains."""
    assert Domain.NONNEGATIVE_INTEGER.sage_domain == 'integer'


def test_declarations_of_returns_fields_by_kind():
    """Test that declarations are grouped by kind in declaration order."""
    assert list(declarations_of(Body, VariableKind.STATE)) == ['n', 'x']
    assert list(declarations_of(Body, VariableKind.PARAMETER)) == ['m']
    assert list(declarations_of(Link, VariableKind.ROLE)) == ['first', 'second']


def test_state_default_sequence_is_copied():
    """Test that a sequence default is not shared between instances."""
    first, second = Body('a'), Body('b')
    first.x.append(1.0)
    assert second.x == [0.0, 0.0]


def test_registry_resolve_unknown_key_lists_keys():
    """Test that resolving an unknown key names the registered keys."""
    registry = Registry[int]('number')
    registry.register('one', 1)
    with pytest.raises(KeyError, match='Registered keys are: one'):
        registry.resolve('two')


def test_state_accepts_fractional_order_without_companions():
    """Test that a state of a fractional order keeps its order and gets no companion states."""
    Relaxing = Element.define('Relaxing', states={'x': state(1.0, order=0.6)})
    declaration = declarations_of(Relaxing, VariableKind.STATE)['x']
    assert declaration.order == 0.6
    assert declaration.is_fractional
    assert declaration.derivatives == 0
    assert list(declarations_of(Relaxing, VariableKind.STATE)) == ['x']


def test_state_normalizes_integer_valued_orders():
    """Test that an order with an integer value is an integer order with its companions."""
    declarations = declarations_of(Element.define('Mass', states={'x': state(0.0, order=2.0)}), VariableKind.STATE)
    assert declarations['x'].order == 2
    assert not declarations['x'].is_fractional
    assert 'x_d1' in declarations


@pytest.mark.parametrize(
    ('arguments', 'error'),
    [
        ({'order': 0}, ValueError),
        ({'order': -0.5}, ValueError),
        ({'order': float('nan')}, ValueError),
        ({'order': 0.5, 'lags': 1}, ValueError),
        ({'order': True}, TypeError),
        ({'order': '0.5'}, TypeError),
    ],
)
def test_state_rejects_invalid_orders(arguments, error):
    """Test that an order that is not positive, not a number, or given with lags is rejected."""
    with pytest.raises(error, match='order'):
        state(1.0, **arguments)
