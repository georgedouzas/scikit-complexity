"""Test the `_local_operators` module of `skcomplex.core.components.laws.utils`."""

from sage.all__sagemath_symbolics import SR, var

from skcomplex.core.components.laws.utils import dagger, has_operator, local_operator, operator_product


def test_a_coupling_keeps_the_order_of_its_operators():
    """Test that a coupling of two subsystems keeps its product and adjoint unevaluated and in order."""
    a = local_operator(SR.symbol('a'), 'x')
    b = local_operator(SR.symbol('b'), 'minus')
    coupling = var('J') * operator_product(a, b) + dagger(b)
    assert str(coupling) == (
        'J*operator_product(local_operator(a, x), local_operator(b, minus)) + dagger(local_operator(b, minus))'
    )
    assert has_operator(coupling)


def test_has_operator_is_false_for_numbers_and_coefficients():
    """Test that a coefficient or a plain number holds no local operator."""
    assert not has_operator(var('J'))
    assert not has_operator(2.0)
