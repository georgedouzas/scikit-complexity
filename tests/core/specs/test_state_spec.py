"""Test the state spec."""

from typing import Annotated, ClassVar

import numpy as np
import pytest
from sage.all__sagemath_symbolics import CC, SR, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import MatrixGroupSpec, StateSpec

COLOR = MatrixGroupSpec('SU', 3)


class _NematicElement(Element):
    """Element with a symmetric traceless tensor state, which keeps the matrix its laws read."""

    seen: ClassVar[list[object]] = []

    def __init__(self, label, Q: Annotated[object, StateSpec(tensor='traceless')] = 0.0):
        super().__init__(label)
        self.Q = Q

    def _get_laws(self, namespace):
        """Return no laws, after keeping the matrix of the state."""
        _NematicElement.seen.append(namespace.Q)
        return []


def _matrix_on(shape):
    """Return the matrix laws read of a symmetric traceless state and the size of the field."""
    _NematicElement.seen.clear()
    field = FieldElement('film', element=_NematicElement, space=LatticeSpace(shape))
    compiler = PassCompiler().prepare(System(elements=[field]))
    compiler.run(None)
    return _NematicElement.seen[-1], compiler.problem_.states


def test_check_rejects_vector_and_tensor_together():
    """A state that is both a vector and a tensor names both settings."""
    with pytest.raises(ValidationError, match=r'vector=True or tensor=.symmetric'):
        StateSpec(vector=True, tensor='symmetric').prepare()


def test_check_rejects_unknown_symmetry():
    """An unknown tensor symmetry raises an error."""
    with pytest.raises(ValidationError, match='tensor symmetry'):
        StateSpec(tensor='skew').prepare()


def test_tensor_traceless_stores_two_components_in_two_dimensions():
    """A symmetric traceless state on a 2D lattice stores 2 components per site, and its trace is exactly 0."""
    matrix, states = _matrix_on((32, 32))
    assert states.entries[0].columns == 2
    assert SR(matrix.trace()).is_trivial_zero()


def test_tensor_traceless_stores_five_components_in_three_dimensions():
    """A symmetric traceless state on a 3D lattice stores 5 components per site, with equal off-diagonal entries."""
    matrix, states = _matrix_on((4, 4, 4))
    assert states.entries[0].columns == 5
    assert matrix[1, 0].is_trivially_equal(matrix[0, 1])
    assert SR(matrix.trace()).is_trivial_zero()


def test_algebraic_state_has_order_one_and_no_lags():
    """Test that an algebraic state rejects an order other than 1 and lags, which its equations leave no room for."""
    assert StateSpec(algebraic=True).algebraic
    with pytest.raises(ValidationError, match='An algebraic state has order 1 and no lags'):
        StateSpec(algebraic=True, order=2).prepare()
    with pytest.raises(ValidationError, match='An algebraic state has order 1 and no lags'):
        StateSpec(algebraic=True, lags=1).prepare()


class _ColorLinkElement(Element):
    """Link with a matrix of SU(3)."""

    def __init__(self, label, visual=None, *, U: Annotated[object, StateSpec(group=COLOR)] = tuple(COLOR.identity)):
        super().__init__(label, visual=visual)
        self.U = U


def test_matrix_group_state_stores_the_real_components_of_every_matrix():
    """Test that an SU(3) state on the 32 links of a 4 by 4 lattice stores 18 components per link, the identity."""
    space = CellComplexSpace(LatticeSpace((4, 4)))
    links = GroupElement('links', element=_ColorLinkElement, space=space, on='edges')
    problem = PassCompiler().prepare(System(elements=[links])).run().problem_
    entry = problem.states.entries[0]
    assert (entry.rows, entry.columns) == (32, 18)
    matrices = COLOR.matrices_of(np.reshape(problem.initial_state, (32, 18)))
    assert np.array_equal(matrices, np.broadcast_to(np.eye(3), (32, 3, 3)))


@pytest.mark.parametrize('arguments', [{'vector': True}, {'tensor': 'full'}, {'order': 2}, {'lags': 1}])
def test_matrix_group_state_holds_its_matrices_alone(arguments):
    """Test that a matrix group state that is also a vector, a tensor, or of another order or with lags is refused."""
    with pytest.raises(ValidationError, match='holds its matrices alone'):
        StateSpec(group=COLOR, **arguments).prepare()


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
