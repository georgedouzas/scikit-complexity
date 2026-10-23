"""Test the lowering of conserved noise to link processes on lattices, and its validation."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.base import NoiseKind, ValidationError, symbol
from skcomplex.core.compile import compile_system, validate
from skcomplex.core.components import Element, Field
from skcomplex.core.laws import Noise
from skcomplex.core.solvers import EulerMaruyama, Heun, Milstein
from skcomplex.core.space import LatticeSpace, NetworkSpace
from skcomplex.core.system import System

Conserved = Element.define(
    'Conserved',
    states={'rho': 0.0},
    parameters={'a': 2.0},
    laws=lambda s: [Noise(s.rho, s.a, conserved=True)],
)


def _field(shape, boundary='periodic', spacing=1.0):
    """Return a system of one field with conserved noise on a lattice."""
    space = LatticeSpace(shape, spacing=spacing)
    return System(components=[Field('f', Conserved, space, boundary={'rho': boundary})])


def _noise_matrix(problem):
    """Return the noise matrix of a problem, one row per state and one column per process."""
    amplitudes = problem.diffusion(0.0, problem.initial_state, problem.initial_parameters)
    matrix = np.zeros((problem.initial_state.size, problem.n_processes))
    np.add.at(matrix, (problem.noise_rows, problem.noise_columns), amplitudes)
    return matrix


def test_noise_rejects_conserved_process():
    """Test that a conserved noise law rejects a named process."""
    with pytest.raises(ValidationError, match='per link'):
        Noise(symbol('rho'), 1.0, conserved=True, process='common')


def test_compile_system_gives_every_link_two_entries():
    """Test that every link process of a periodic lattice adds `a / h` to one site and takes it from its neighbour."""
    matrix = _noise_matrix(compile_system(_field((4, 3), spacing=0.5)))
    assert matrix.shape == (12, 24)
    np.testing.assert_allclose(np.sort(matrix, axis=0)[[0, -1]], [[-4.0] * 24, [4.0] * 24])
    np.testing.assert_array_equal(np.count_nonzero(matrix, axis=0), 2)
    np.testing.assert_allclose(matrix.sum(axis=0), 0.0)


def test_compile_system_leaves_out_outer_links_of_zero_flux_edges():
    """Test that a zero-flux chain draws its inner links and a silent process for the links that leave it."""
    problem = compile_system(_field((4,), boundary='zero-flux'))
    matrix = _noise_matrix(problem)
    (block,) = problem.noise_blocks
    np.testing.assert_array_equal(block.factor, [[0.0]])
    used = np.delete(matrix, [3, block.start], axis=1)
    np.testing.assert_array_equal(np.count_nonzero(used, axis=0), 2)
    np.testing.assert_allclose(used.sum(axis=0), 0.0)


def test_compile_system_rejects_conserved_amplitude_of_state():
    """Test that a conserved noise whose amplitude reads the field is rejected as out of scope."""
    Multiplicative = Element.define(
        'Multiplicative',
        states={'rho': 1.0},
        laws=lambda s: [Noise(s.rho, s.rho, conserved=True)],
    )
    system = System(components=[Field('f', Multiplicative, LatticeSpace((4,)))])
    with pytest.raises(ValidationError, match='reads the states'):
        compile_system(system)


def test_validate_reports_conserved_noise_off_lattice():
    """Test that conserved noise on an element that is not a field reports `noise-unsupported` naming lattices."""
    (issue,) = validate(System(components=[Conserved('c')])).issues
    assert issue.code == 'noise-unsupported'
    assert 'c.rho' in issue.message
    assert 'LatticeSpace' in issue.fix


def test_validate_reports_conserved_noise_on_network():
    """Test that conserved noise on a field of a network reports `noise-unsupported`."""
    space = NetworkSpace(nx.cycle_graph(5))
    issues = validate(System(components=[Field('f', Conserved, space)])).issues
    assert [issue.code for issue in issues] == ['noise-unsupported']


def test_validate_reports_milstein_with_conserved_noise():
    """Test that Milstein with conserved noise is reported with the solvers that draw link processes."""
    (issue,) = validate(_field((4,)), solver='milstein').issues
    assert issue.code == 'noise-unsupported'
    assert 'conserved noise' in issue.message
    assert 'euler-maruyama' in issue.fix


def test_validate_accepts_conserved_noise_on_lattice():
    """Test that conserved noise on a lattice field validates under the default solver."""
    assert validate(_field((4, 4))).ok


def test_declared_conserved_support():
    """Test that Euler-Maruyama and Heun declare conserved noise and Milstein does not."""
    assert NoiseKind.CONSERVED in EulerMaruyama.supports.noise
    assert NoiseKind.CONSERVED in Heun.supports.noise
    assert NoiseKind.CONSERVED not in Milstein.supports.noise
