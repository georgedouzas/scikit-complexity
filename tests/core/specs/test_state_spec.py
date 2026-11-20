"""Test the state spec."""

import pytest
from sage.all__sagemath_symbolics import CC, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import MatrixGroupSpec, StateSpec

COLOR = MatrixGroupSpec('SU', 3)


def test_check_rejects_vector_and_tensor_together():
    """A state that is both a vector and a tensor names both settings."""
    with pytest.raises(ValidationError, match=r'vector=True or tensor=.symmetric'):
        StateSpec(vector=True, tensor='symmetric')._check()


def test_check_rejects_unknown_symmetry():
    """An unknown tensor symmetry raises an error."""
    with pytest.raises(ValidationError, match='tensor symmetry'):
        StateSpec(tensor='skew')._check()


def test_algebraic_state_has_order_one_and_no_lags():
    """Test that an algebraic state rejects an order other than 1 and lags, which its equations leave no room for."""
    assert StateSpec(algebraic=True).algebraic
    with pytest.raises(ValidationError, match='An algebraic state has order 1 and no lags'):
        StateSpec(algebraic=True, order=2)._check()
    with pytest.raises(ValidationError, match='An algebraic state has order 1 and no lags'):
        StateSpec(algebraic=True, lags=1)._check()


@pytest.mark.parametrize('arguments', [{'vector': True}, {'tensor': 'full'}, {'order': 2}, {'lags': 1}])
def test_matrix_group_state_holds_its_matrices_alone(arguments):
    """Test that a matrix group state that is also a vector, a tensor, or of another order or with lags is refused."""
    with pytest.raises(ValidationError, match='holds its matrices alone'):
        StateSpec(group=COLOR, **arguments)._check()


@pytest.mark.parametrize(
    ('arguments', 'tag'),
    [
        ({}, 'real'),
        ({'domain': ZZ}, 'integer'),
        ({'domain': FiniteEnumeratedSet([0, 1])}, 'boolean'),
        ({'domain': FiniteEnumeratedSet(['S', 'I'])}, 'categorical'),
        ({'domain': CC}, 'complex'),
        ({'vector': True}, 'vector'),
        ({'group': COLOR}, 'matrix group'),
        ({'domain': None}, None),
    ],
)
def test_state_gives_its_domain_tag(arguments, tag):
    """Test that a state spec gives the domain tag of its values."""
    assert StateSpec(**arguments)._get_domain_tag() == tag


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (StateSpec(order=0), 'order'),
        (StateSpec(order=-0.5), 'order'),
        (StateSpec(order=float('nan')), 'order'),
        (StateSpec(order=1.5), 'order'),
        (StateSpec(lags=-1), 'lags'),
        (StateSpec(lags=1.5), 'lags'),
        (StateSpec(order=2, lags=1), 'not both'),
        (StateSpec(order=2, initial=(0.0, 1.0)), '2 initial values were given for 1'),
        (StateSpec(reference='banks', vector=True), 'scalar of order 1'),
        (StateSpec(reference='banks', lags=1), 'scalar of order 1'),
    ],
)
def test_state_spec_rejects_invalid_orders_and_lags(spec, message):
    """Test that an order that is not positive or integral, negative lags, or too many initial values are rejected."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


@pytest.mark.parametrize('spec', [StateSpec(order=0.6), StateSpec(order=2.0), StateSpec(lags=2, initial=(1.0, 2.0))])
def test_state_spec_accepts_fractional_orders_and_lags(spec):
    """Test that a fractional order, an integral order given as a float, and lags with their initial values pass."""
    spec._check()


def test_state_spec_rejects_an_unknown_role():
    """Test that a role is one of the names analyses read."""
    with pytest.raises(ValidationError, match='The role of a state'):
        StateSpec(role='spin')._check()
