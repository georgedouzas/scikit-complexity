"""Test the `_state_spec` module of `skcomplex.core.components`."""

import pytest
from sage.all__sagemath_symbolics import CC, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components import MatrixGroupSpec
from skcomplex.core.components._state_spec import _StateSpec

COLOR = MatrixGroupSpec('SU', 3)


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
    assert _StateSpec(**arguments)._get_domain_tag() == tag
