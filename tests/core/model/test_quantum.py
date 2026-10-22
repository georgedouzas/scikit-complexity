"""Test quantum subsystems, operators, and the Hamiltonian and dissipator laws."""

import networkx as nx
import numpy as np
import pytest
import qutip

from skcomplex.core.base import ValidationError, symbol
from skcomplex.core.model import (
    Dissipator,
    Generator,
    LatticeSpace,
    LocalOperator,
    NetworkSpace,
    Operator,
    QuantumView,
    Symbols,
    expectation,
    neighbour_pairs,
)
from skcomplex.models.physics.realizations import Mode, Qubit, Site, Spin


def _namespace(element, key):
    """Return the namespace of the local operators of an element."""
    return Symbols(**QuantumView(element).namespace(key))


def test_spin_local_operators_match_qutip():
    """Test that spin operators are QuTiP's angular momentum matrices."""
    operators = QuantumView(Spin('s', j=1)).local_operators()
    for name, component in (('x', 'x'), ('y', 'y'), ('z', 'z'), ('plus', '+'), ('minus', '-')):
        assert np.allclose(operators[name].matrix.toarray(), qutip.jmat(1, component).full())


def test_qubit_local_operators_are_pauli_matrices():
    """Test that qubit operators are the Pauli matrices and their ladder operators."""
    operators = QuantumView(Qubit('q')).local_operators()
    assert np.allclose(operators['x'].matrix.toarray(), qutip.sigmax().full())
    assert np.allclose(operators['minus'].matrix.toarray(), qutip.sigmam().full())


def test_mode_local_operators_match_qutip():
    """Test that mode operators are QuTiP's truncated ladder and number operators."""
    operators = QuantumView(Mode('a', quanta=4)).local_operators()
    assert np.allclose(operators['annihilate'].matrix.toarray(), qutip.destroy(5).full())
    assert np.allclose(operators['number'].matrix.toarray(), qutip.num(5).full())


@pytest.mark.parametrize(
    ('element', 'message'),
    [(Spin('s', j=0.3), 'multiple of one half'), (Mode('a', quanta=0), 'positive integer'), (Site('p'), 'Give it')],
)
def test_dimension_rejects_invalid_spaces(element, message):
    """Test that a subsystem with no valid Hilbert space raises an error."""
    with pytest.raises(ValidationError, match=message):
        _ = QuantumView(element).dimension


def test_initial_vector_resolves_labels_and_vectors():
    """Test that initial states come from labels, indices, and normalised vectors."""
    assert QuantumView(Spin('s', initial='down')).initial_vector().tolist() == [0, 1]
    assert QuantumView(Mode('a', quanta=2, initial=2)).initial_vector().tolist() == [0, 0, 1]
    vector = np.array([1, 1j]) / np.sqrt(2)
    assert np.allclose(QuantumView(Qubit('q', initial=vector)).initial_vector(), vector)


@pytest.mark.parametrize('initial', ['left', 7, [1.0, 1.0]])
def test_initial_vector_rejects_unknown_states(initial):
    """Test that unknown labels and unnormalised vectors raise an error."""
    with pytest.raises(ValidationError, match=r'basis state|Normalise'):
        QuantumView(Qubit('q', initial=initial)).initial_vector()


def test_operator_exchange_term_has_three_terms():
    """Test that the XXZ exchange of two spins has the coefficients J, J, and J delta on both spins."""
    a, b = _namespace(Spin('a'), 'a'), _namespace(Spin('b'), 'b')
    J, delta = symbol('J'), symbol('delta')
    operator = J * (a.x * b.x + a.y * b.y + delta * a.z * b.z)
    assert list(operator.terms.values()) == [J, J, J * delta]
    assert operator.subsystems == [{'a', 'b'}] * 3


def test_operator_arithmetic_combines_terms():
    """Test sums, differences, scalar products, divisions, and the identity of operators."""
    q = _namespace(Qubit('q'), 'q')
    assert repr(q.z - q.z) == '0'
    assert repr((1 - q.z) / 2) == '(-1/2)*q.z + 1/2'
    assert repr(np.float64(2.0) * q.x) == '2.0*q.x'
    assert repr(2 - q.x + q.x) == '2'


def test_operator_rejects_non_scalars():
    """Test that an operator does not combine with arrays or strings."""
    q = _namespace(Qubit('q'), 'q')
    with pytest.raises(TypeError):
        _ = q.x * np.ones(2)
    with pytest.raises(TypeError):
        _ = q.x + 'a'


def test_operator_adjoint_reverses_and_conjugates():
    """Test that the adjoint reverses products, adjoins factors, and conjugates coefficients."""
    a, m = _namespace(Qubit('q'), 'q'), _namespace(Mode('m'), 'm')
    adjoint = (1j * a.plus * m.annihilate).adjoint()
    assert repr(adjoint) == '(-1j)*m.create*q.minus'
    assert LocalOperator('hop').adjoint().adjoint().name == 'hop'


def test_laws_reject_values_that_are_not_operators():
    """Test that the quantum laws check their operator and rate."""
    q = _namespace(Qubit('q'), 'q')
    with pytest.raises(ValidationError, match='Generator takes'):
        Generator(1.0)
    with pytest.raises(ValidationError, match='Dissipator takes'):
        Dissipator('q')
    with pytest.raises(ValidationError, match='rate'):
        Dissipator(q.minus, 'fast')


def test_neighbour_pairs_list_lattice_and_network_bonds():
    """Test that neighbour pairs are the bonds of periodic and open lattices and of networks."""
    assert neighbour_pairs(LatticeSpace(4), 4) == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert neighbour_pairs(LatticeSpace(3, boundary='fixed'), 3) == [(0, 1), (1, 2)]
    assert len(neighbour_pairs(LatticeSpace((2, 3)), 6)) == 9
    assert neighbour_pairs(NetworkSpace(nx.path_graph(3)), 3) == [(0, 1), (1, 2)]
    with pytest.raises(ValidationError, match='Neighbours'):
        neighbour_pairs(None, 2)


def test_site_namespace_has_projectors_and_diagonals():
    """Test that a particle on sites exposes projectors, diagonal operators, and the adjacency."""
    site = Site('p', space=NetworkSpace(nx.cycle_graph(4)))
    names = _namespace(site, 'p')
    projector = names.projector(2)
    local = next(iter(projector.terms))[0][1]
    assert local.matrix.toarray()[2, 2] == 1
    assert repr(names.diagonal([1, 2, 3, 4])).startswith('p.diagonal(')
    assert QuantumView(site).local_operators()['adjacency'].matrix.sum() == 8
    assert QuantumView(site).local_operators()['laplacian'].matrix.toarray()[0, 0] == -2


def test_expectation_keeps_member_values():
    """Test that an expectation of a group keeps one value per member unless it is reduced."""
    observable = expectation('sz', abs, group='chain')
    assert observable.reduce([1.0, 2.0]).tolist() == [1.0, 2.0]
    assert expectation('total', abs, group='chain', reduce='sum').reduce == 'sum'


def test_lattice_group_keeps_the_expectation_visual():
    """Test that a lattice of quantum elements is drawn by its expectations."""
    assert Spin.lattice('chain', LatticeSpace(3)).visual.kind == 'expectations'

    assert isinstance(Operator({}), Operator)
