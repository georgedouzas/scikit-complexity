"""Test the census pass on couplings over the neighbours of fields."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from skfem import MeshTri

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian, logistic
from skcomplex.core.components.spaces import LatticeSpace, MeshSpace, NetworkSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.specs import ParameterSpec, StateSpec


class SiteElement(Element):
    """Site element whose rate is a Laplacian times a coefficient."""

    def __init__(
        self,
        label: str,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.u = u
        self.D = D

    def _get_laws(self, s):
        """Return the rate of the site."""
        return [RateLaw(s.u, s.D * laplacian(s.u))]


class PlainElement(Element):
    """Site element with no rate of its own."""

    def __init__(self, label: str, *, u: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.u = u

    def _get_laws(self, s):
        """Return the zero rate of the site."""
        return [RateLaw(s.u, 0 * s.u)]


def _rate(group, interactions=(), backend=NumpyBackend):
    """Return the rate of the state of a group at its initial values."""
    system = System(elements=[group], interactions=list(interactions))
    problem = PassCompiler(backend=backend()).prepare(system).run().problem_
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    return np.ravel(problem.states.read(drift, 'm.u'))


def _logistic(s):
    """Return a smooth step of the other member's state."""
    return logistic(s.other.u - 0.4, 0, 0.1)


def _neighbor_sum(padded):
    """Return the sum over the four nearest sites of a smooth step of a padded 2D field."""
    step = 1 / (1 + np.exp(-(padded - 0.4) / 0.1))
    return step[:-2, 1:-1] + step[2:, 1:-1] + step[1:-1, :-2] + step[1:-1, 2:]


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_declared_laws_sum_a_function_of_both_ends_over_the_lattice_neighbours(backend):
    """Test that a coupling on a periodic lattice field equals the direct sum over the four nearest sites."""
    field = np.random.default_rng(0).random((6, 5))
    group = FieldElement('m', element=SiteElement, space=LatticeSpace((6, 5)), values={'u': field})
    coupling = CouplingInteraction('c', target='m', read='u', write='u', function=_logistic)
    rate = _rate(group, [coupling], backend)
    expected = _neighbor_sum(np.pad(field, 1, mode='wrap'))
    assert np.abs(rate - np.ravel(expected)).max() < 1e-12


def test_declared_laws_read_the_fixed_value_beyond_fixed_edges():
    """Test that a coupling on a field with a fixed boundary reads the fixed value beyond the edges."""
    field = np.random.default_rng(1).random((6, 5))
    group = FieldElement(
        'm',
        element=SiteElement,
        space=LatticeSpace((6, 5)),
        boundary={'u': ('fixed', 0.5)},
        values={'u': field},
    )
    coupling = CouplingInteraction('c', target='m', read='u', write='u', function=_logistic)
    expected = _neighbor_sum(np.pad(field, 1, constant_values=0.5))
    assert np.abs(_rate(group, [coupling]) - np.ravel(expected)).max() < 1e-12


def test_declared_laws_agree_with_the_edge_kernel_away_from_fixed_edges():
    """Test that a coupling on a field with fixed edges matches the edge kernel of a group at the interior sites."""
    field = np.random.default_rng(2).random((7, 6))
    values = {'u': field}
    coupling = CouplingInteraction('c', target='m', read='u', write='u', function=_logistic)
    space = LatticeSpace((7, 6))
    lowered = FieldElement('m', element=SiteElement, space=space, boundary={'u': ('fixed', 0.0)}, values=values)
    edges = GroupElement('m', element=PlainElement, size=42, space=LatticeSpace((7, 6)), values={'u': np.ravel(field)})
    difference = np.reshape(_rate(lowered, [coupling]) - _rate(edges, [coupling]), (7, 6))
    assert np.abs(difference[1:-1, 1:-1]).max() < 1e-12
    assert np.abs(difference).max() > 1e-3


def test_declared_laws_lower_a_linear_exchange_on_a_network():
    """Test that a coupling on a network field sums `g (x_j - x_i) + xi x_j` as `g L x + xi A x`."""
    graph = nx.connected_watts_strogatz_graph(20, 4, 0.3, seed=0)
    field = np.random.default_rng(3).random(20)
    group = FieldElement('m', element=SiteElement, space=NetworkSpace(graph), values={'u': field})
    coupling = CouplingInteraction(
        'c',
        target='m',
        read='u',
        write='u',
        function=lambda s: 0.3 * (s.other.u - s.member.u) + 0.05 * s.other.u,
    )
    adjacency = nx.to_numpy_array(graph)
    laplacian_matrix = adjacency - np.diag(adjacency.sum(axis=1))
    expected = 0.3 * laplacian_matrix @ field + 0.05 * adjacency @ field
    assert np.abs(_rate(group, [coupling]) - expected).max() < 1e-12


def test_declared_laws_lower_a_diffusive_coupling_to_the_spectral_laplacian():
    """Test that a coupling `D (x_j - x_i)` on a spectral field is `D` times the field's Fourier Laplacian."""
    positions = np.arange(32) * 2 * np.pi / 32
    field = np.sin(3 * positions)
    space = LatticeSpace(32, spacing=2 * np.pi / 32)
    group = FieldElement('m', element=SiteElement, space=space, values={'u': field}, discretization='spectral')
    coupling = CouplingInteraction(
        'c',
        target='m',
        read='u',
        write='u',
        function=lambda s: 0.7 * (s.other.u - s.member.u),
    )
    expected = -0.7 * (2 * np.pi / 32) ** 2 * 9 * field
    assert np.abs(_rate(group, [coupling]) - expected).max() < 1e-12


def test_declared_laws_lower_a_diffusive_coupling_to_the_finite_element_laplacian():
    """Test that a coupling `D (x_j - x_i)` on a mesh field is `D` times the finite element Laplacian."""
    mesh = MeshTri.init_sqsymmetric().refined(2)
    field = np.sin(np.pi * mesh.p[0]) * mesh.p[1]
    space = MeshSpace(mesh.p, mesh.t)
    settings = {'element': SiteElement, 'space': space, 'discretization': 'finite-element'}
    coupled = FieldElement('m', values={'u': field}, **settings)
    coupling = CouplingInteraction(
        'c',
        target='m',
        read='u',
        write='u',
        function=lambda s: 0.7 * (s.other.u - s.member.u),
    )
    direct = FieldElement('m', values={'u': field, 'D': 0.7}, **settings)
    assert np.abs(_rate(coupled, [coupling]) - _rate(direct)).max() < 1e-12


def test_declared_laws_divide_the_mean_by_the_lattice_neighbours():
    """Test that the mean of a coupling on a periodic lattice field divides the sum by the four nearest sites."""
    field = np.random.default_rng(4).random((5, 5))
    group = FieldElement('m', element=SiteElement, space=LatticeSpace((5, 5)), values={'u': field})
    coupling = CouplingInteraction('c', target='m', read='u', write='u', function=_logistic, reduce='mean')
    expected = _neighbor_sum(np.pad(field, 1, mode='wrap')) / 4
    assert np.abs(_rate(group, [coupling]) - np.ravel(expected)).max() < 1e-12
