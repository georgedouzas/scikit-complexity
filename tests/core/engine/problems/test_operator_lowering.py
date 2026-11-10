"""Test the `operator_lowering` module."""

import qutip
from sage.all__sagemath_symbolics import SR, var

from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.engine.problems._operator_lowering import OperatorLowering
from skcomplex.core.specs import LocalOperatorSpec


def test_operator_lowering_get_operator_of():
    """Test `OperatorLowering._get_operator_of` on a small case."""
    q = SR.symbol('q')
    z = LocalOperatorSpec('z', qutip.sigmaz(), hermitian=True)
    term = var('h') * local_operator(q, SR.symbol('z'))
    assert OperatorLowering._get_operator_of(term, {q: ('q', {'z': z})}).text == 'h*q.z'
