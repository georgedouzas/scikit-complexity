"""Test operator subsystems, operators, and the Hamiltonian and dissipator laws."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import qutip

from skcomplex.core.components.elements import LatticeElement
from skcomplex.core.components.laws import DissipatorLaw, GeneratorLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.spaces import LatticeSpace, NetworkSpace
from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.compile import OperatorCompiler
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import HilbertSpec, LocalOperatorSpec, OperatorSpec
from skcomplex.systems.physics.common.quantum import ModeElement, QubitElement, mode_space

from ....conftest import SpinElement, _spin_space


def _namespace(element, key):
    """Return the local operators of an element by name, each as an operator on the subsystem of a key."""
    operators = OperatorCompiler._view_of(element).local_operators()
    return SimpleNamespace(**{name: OperatorSpec({((key, local),): 1}) for name, local in operators.items()})


def test_spin_local_operators_match_qutip():
    """Test that spin operators are QuTiP's angular momentum matrices."""
    operators = OperatorCompiler._view_of(SpinElement('s', hilbert=_spin_space(1))).local_operators()
    for name, component in (('x', 'x'), ('y', 'y'), ('z', 'z'), ('plus', '+'), ('minus', '-')):
        assert np.allclose(operators[name].matrix.toarray(), qutip.jmat(1, component).full())


def test_qubit_local_operators_are_pauli_matrices():
    """Test that qubit operators are the Pauli matrices and their ladder operators."""
    operators = OperatorCompiler._view_of(QubitElement('q')).local_operators()
    assert np.allclose(operators['x'].matrix.toarray(), qutip.sigmax().full())
    assert np.allclose(operators['minus'].matrix.toarray(), qutip.sigmam().full())


def test_mode_local_operators_match_qutip():
    """Test that mode operators are QuTiP's truncated ladder and number operators."""
    operators = OperatorCompiler._view_of(ModeElement('a', hilbert=mode_space(4))).local_operators()
    assert np.allclose(operators['annihilate'].matrix.toarray(), qutip.destroy(5).full())
    assert np.allclose(operators['number'].matrix.toarray(), qutip.num(5).full())


def test_dimension_rejects_an_element_without_a_hilbert_space():
    """Test that a subsystem without a Hilbert space raises an error."""
    with pytest.raises(ValidationError, match='takes no HilbertSpec'):
        _ = OperatorCompiler._view_of(SpinElement('p', hilbert=None)).dimension


def test_initial_vector_resolves_labels_and_vectors():
    """Test that initial states come from labels, indices, and normalised vectors."""
    assert OperatorCompiler._view_of(SpinElement('s', psi='down')).initial_vector().tolist() == [0, 1]
    mode = ModeElement('a', psi=2, hilbert=mode_space(2))
    assert OperatorCompiler._view_of(mode).initial_vector().tolist() == [0, 0, 1]
    vector = np.array([1, 1j]) / np.sqrt(2)
    assert np.allclose(OperatorCompiler._view_of(QubitElement('q', psi=vector)).initial_vector(), vector)


@pytest.mark.parametrize('initial', ['left', 7, [1.0, 1.0]])
def test_initial_vector_rejects_unknown_states(initial):
    """Test that unknown labels and unnormalised vectors raise an error."""
    with pytest.raises(ValidationError, match=r'basis state|Normalise'):
        OperatorCompiler._view_of(QubitElement('q', psi=initial)).initial_vector()


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
    adjoint = a.plus.multiply(m.annihilate).multiply(1j).adjoint()
    assert adjoint.text == '(-1j)*m.create*q.minus'
    assert LocalOperatorSpec('hop', qutip.sigmap()).adjoint().adjoint().name == 'hop'


def test_laws_reject_values_that_are_not_operators():
    """Test that the operator laws check their operator and rate."""
    with pytest.raises(ValidationError, match='generator takes'):
        GeneratorLaw(1.0).prepare()
    with pytest.raises(ValidationError, match='dissipator takes'):
        DissipatorLaw('q').prepare()
    with pytest.raises(ValidationError, match='rate'):
        DissipatorLaw(local_operator(symbol('q'), 'minus'), local_operator(symbol('q'), 'z')).prepare()


def test_neighbor_pairs_list_lattice_and_network_bonds():
    """Test that neighbour pairs are the bonds of periodic and open lattices and of networks."""
    assert LatticeSpace(4).prepare()._neighbor_pairs() == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert LatticeSpace(3, boundary='fixed').prepare()._neighbor_pairs() == [(0, 1), (1, 2)]
    assert len(LatticeSpace((2, 3)).prepare()._neighbor_pairs()) == 9
    assert NetworkSpace(nx.path_graph(3)).prepare()._neighbor_pairs() == [(0, 1), (1, 2)]


def test_operators_keep_the_projectors_and_the_adjacency_of_their_space():
    """Test that an element's operators keep the projectors, the adjacency, and the Laplacian of its Hilbert space."""
    adjacency = nx.to_numpy_array(nx.cycle_graph(4))
    projector = np.zeros((4, 4))
    projector[2, 2] = 1.0
    operators = {
        'adjacency': (adjacency, True),
        'laplacian': (adjacency - np.diag(adjacency.sum(axis=1)), True),
        'site_2': (projector, True),
    }
    site = SpinElement('p', psi=0, hilbert=HilbertSpec(4, operators, display=None))
    operators = OperatorCompiler._view_of(site).local_operators()
    assert operators['site_2'].matrix.toarray()[2, 2] == 1
    assert operators['adjacency'].matrix.sum() == 8
    assert operators['laplacian'].matrix.toarray()[0, 0] == -2


def test_lattice_group_keeps_the_expectation_visual():
    """Test that a lattice of operator elements is drawn by its expectations."""
    chain = LatticeElement('chain', element=SpinElement, space=LatticeSpace(3)).prepare()
    assert chain.default_visual_.kind == 'expectations'
