"""Test the `_operator_spec` module of `skcomplex.core.specs`."""

from types import SimpleNamespace

import numpy as np
import pytest
import qutip

from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import LocalOperatorSpec, OperatorSpec
from skcomplex.systems.physics.common.quantum import ModeElement, QubitElement, mode_space

from ...conftest import SpinElement


def _namespace(element, key):
    """Return the local operators of an element by name, each as an operator on the subsystem of a key."""
    operators = OperatorCompiler._get_view_of(element).get_local_operators()
    return SimpleNamespace(**{name: OperatorSpec({((key, local),): 1}) for name, local in operators.items()})


def test_operator_exchange_term_has_three_terms():
    """Test that the XXZ exchange of two spins has the coefficients J, J, and J delta on both spins."""
    a, b = _namespace(SpinElement('a'), 'a'), _namespace(SpinElement('b'), 'b')
    J, delta = symbol('J'), symbol('delta')
    operator = a.x.multiply(b.x).add(a.y.multiply(b.y)).add(a.z.multiply(b.z).multiply(delta)).multiply(J)
    assert list(operator.terms.values()) == [J, J, J * delta]
    assert operator.subsystems == [{'a', 'b'}] * 3


def test_operator_arithmetic_combines_terms():
    """Test sums, differences, scalar products, divisions, and the identity of operators."""
    q = _namespace(QubitElement('q'), 'q')
    assert q.z.subtract(q.z).text == '0'
    assert q.z.negate().add(1).divide(2).text == '(-1/2)*q.z + 1/2'
    assert q.x.multiply(np.float64(2.0)).text == '2.0*q.x'
    assert q.x.negate().add(2).add(q.x).text == '2'


def test_operator_product_keeps_its_order():
    """Test that a product of operators keeps the order of its factors."""
    a, m = _namespace(QubitElement('q'), 'q'), _namespace(ModeElement('m', hilbert=mode_space(4)), 'm')
    assert a.plus.multiply(m.annihilate).text == 'q.plus*m.annihilate'
    assert m.annihilate.multiply(a.plus).text == 'm.annihilate*q.plus'


def test_operator_rejects_non_scalars():
    """Test that an operator does not combine with arrays, strings, or a divisor that is an operator."""
    q = _namespace(QubitElement('q'), 'q')
    with pytest.raises(ValidationError, match='cannot multiply by'):
        q.x.multiply(np.ones(2))
    with pytest.raises(ValidationError, match="cannot add 'a' of type str"):
        q.x.add('a')
    with pytest.raises(ValidationError, match='cannot subtract'):
        q.x.subtract([1, 2])
    with pytest.raises(ValidationError, match='cannot divide by'):
        q.x.divide(q.z)


def test_operator_adjoint_reverses_and_conjugates():
    """Test that the adjoint reverses products, adjoins factors, and conjugates coefficients."""
    a, m = _namespace(QubitElement('q'), 'q'), _namespace(ModeElement('m', hilbert=mode_space(4)), 'm')
    adjoint = a.plus.multiply(m.annihilate).multiply(1j).get_adjoint()
    assert adjoint.text == '(-1j)*m.create*q.minus'
    assert LocalOperatorSpec('hop', qutip.sigmap()).get_adjoint().get_adjoint().name == 'hop'
