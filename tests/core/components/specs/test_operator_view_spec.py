"""Test the `_operator_view_spec` module of `skcomplex.core.components.specs`."""

import networkx as nx
import numpy as np
import pytest
import qutip

from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import HilbertSpec
from skcomplex.systems.physics.common.quantum import ModeElement, QubitElement, mode_space

from ....conftest import SpinElement, _spin_space


def test_spin_local_operators_match_qutip():
    """Test that spin operators are QuTiP's angular momentum matrices."""
    operators = OperatorCompiler._get_view_of(SpinElement('s', hilbert=_spin_space(1))).get_local_operators()
    for name, component in (('x', 'x'), ('y', 'y'), ('z', 'z'), ('plus', '+'), ('minus', '-')):
        assert np.allclose(operators[name].matrix.toarray(), qutip.jmat(1, component).full())


def test_qubit_local_operators_are_pauli_matrices():
    """Test that qubit operators are the Pauli matrices and their ladder operators."""
    operators = OperatorCompiler._get_view_of(QubitElement('q')).get_local_operators()
    assert np.allclose(operators['x'].matrix.toarray(), qutip.sigmax().full())
    assert np.allclose(operators['minus'].matrix.toarray(), qutip.sigmam().full())


def test_mode_local_operators_match_qutip():
    """Test that mode operators are QuTiP's truncated ladder and number operators."""
    operators = OperatorCompiler._get_view_of(ModeElement('a', hilbert=mode_space(4))).get_local_operators()
    assert np.allclose(operators['annihilate'].matrix.toarray(), qutip.destroy(5).full())
    assert np.allclose(operators['number'].matrix.toarray(), qutip.num(5).full())


def test_initial_vector_resolves_labels_and_vectors():
    """Test that initial states come from labels, indices, and normalised vectors."""
    assert OperatorCompiler._get_view_of(SpinElement('s', psi='down')).get_initial_vector().tolist() == [0, 1]
    mode = ModeElement('a', psi=2, hilbert=mode_space(2))
    assert OperatorCompiler._get_view_of(mode).get_initial_vector().tolist() == [0, 0, 1]
    vector = np.array([1, 1j]) / np.sqrt(2)
    assert np.allclose(OperatorCompiler._get_view_of(QubitElement('q', psi=vector)).get_initial_vector(), vector)


@pytest.mark.parametrize('initial', ['left', 7, [1.0, 1.0]])
def test_initial_vector_rejects_unknown_states(initial):
    """Test that unknown labels and unnormalised vectors raise an error."""
    with pytest.raises(ValidationError, match=r'basis state|Normalise'):
        OperatorCompiler._get_view_of(QubitElement('q', psi=initial)).get_initial_vector()


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
    operators = OperatorCompiler._get_view_of(site).get_local_operators()
    assert operators['site_2'].matrix.toarray()[2, 2] == 1
    assert operators['adjacency'].matrix.sum() == 8
    assert operators['laplacian'].matrix.toarray()[0, 0] == -2
