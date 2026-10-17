"""Test the domains, declarations, and registry of the core package."""

import pytest

from skcomplex.core import Domain, Registry, VariableKind, declarations_of, parameter, role, state
from skcomplex.model import Element, Interaction


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


def test_domain_assumptions_map_to_sympy():
    """Test that domains give SymPy assumptions."""
    assert Domain.NONNEGATIVE_INTEGER.assumptions == {'integer': True, 'nonnegative': True}


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
