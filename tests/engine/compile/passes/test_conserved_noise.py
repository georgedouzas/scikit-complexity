"""Test the lowering of conserved noise to link processes on lattices, and its validation."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, FieldElement
from skcomplex.core.laws import NoiseLaw
from skcomplex.core.spaces import LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine._utils import symbol
from skcomplex.engine.compile import PassCompiler, compile_system
from skcomplex.engine.solvers import EulerMaruyamaSolver, HeunSolver, MilsteinSolver
from skcomplex.errors import ValidationError
from skcomplex.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.types import NoiseKind


class ConservedElement(Element):
    """Conserved element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec()] = 0.0,
        a: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.a = a

    def _get_laws(self, s):
        """Return the laws of the conserved element."""
        return [NoiseLaw(s.rho, s.a, conserved=True)]


def _field(shape, boundary='periodic', spacing=1.0):
    """Return a system of one field with conserved noise on a lattice."""
    space = LatticeSpace(shape, spacing=spacing)
    return System(elements=[FieldElement('f', element=ConservedElement, space=space, boundary={'rho': boundary})])


def _noise_matrix(problem):
    """Return the noise matrix of a problem, one row per state and one column per process."""
    amplitudes = problem.diffusion(0.0, problem.initial_state, problem.initial_parameters)
    matrix = np.zeros((problem.initial_state.size, problem.n_processes))
    np.add.at(matrix, (problem.noise_rows, problem.noise_columns), amplitudes)
    return matrix


def test_noise_rejects_conserved_process():
    """Test that a conserved noise law rejects a named process."""
    with pytest.raises(ValidationError, match='per link'):
        NoiseLaw(symbol('rho'), 1.0, conserved=True, process='common').prepare()


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

    class MultiplicativeElement(Element):
        """Multiplicative element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            rho: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.rho = rho

        def _get_laws(self, s):
            """Return the laws of the multiplicative element."""
            return [NoiseLaw(s.rho, s.rho, conserved=True)]

    system = System(elements=[FieldElement('f', element=MultiplicativeElement, space=LatticeSpace((4,)))])
    with pytest.raises(ValidationError, match='reads the states'):
        compile_system(system)


def test_validate_reports_conserved_noise_off_lattice():
    """Test that conserved noise on an element that is not a field reports `noise-unsupported` naming lattices."""
    (issue,) = PassCompiler().prepare(System(elements=[ConservedElement('c')])).report_.issues
    assert issue.code == 'noise-unsupported'
    assert 'c.rho' in issue.message
    assert 'LatticeSpace' in issue.fix


def test_validate_reports_conserved_noise_on_network():
    """Test that conserved noise on a field of a network reports `noise-unsupported`."""
    space = NetworkSpace(nx.cycle_graph(5))
    system = System(elements=[FieldElement('f', element=ConservedElement, space=space)])
    report = PassCompiler().prepare(system).report_
    issues = report.issues
    assert [issue.code for issue in issues] == ['noise-unsupported']


def test_validate_reports_milstein_with_conserved_noise():
    """Test that Milstein with conserved noise is reported with the solvers that draw link processes."""
    (issue,) = PassCompiler(solver=MilsteinSolver()).prepare(_field((4,))).report_.issues
    assert issue.code == 'noise-unsupported'
    assert 'conserved noise' in issue.message
    assert 'MilsteinSolver' in issue.message


def test_validate_accepts_conserved_noise_on_lattice():
    """Test that conserved noise on a lattice field validates under the default solver."""
    assert PassCompiler().prepare(_field((4, 4))).report_.ok


def test_declared_conserved_support():
    """Test that Euler-Maruyama and Heun declare conserved noise and Milstein does not."""
    assert NoiseKind.CONSERVED in EulerMaruyamaSolver._get_support().noise
    assert NoiseKind.CONSERVED in HeunSolver._get_support().noise
    assert NoiseKind.CONSERVED not in MilsteinSolver._get_support().noise
