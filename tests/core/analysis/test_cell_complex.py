"""Test the `_cell_complex` module of `skcomplex.core.analysis`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import pi

from skcomplex.core.analysis import cell_complex
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace, NetworkSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import StateSpec


class AngleLinkElement(Element):
    """Link with an angle `theta`."""

    def __init__(self, label, visual=None, *, theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0):
        super().__init__(label, visual=visual)
        self.theta = theta


def _square(size=8):
    """Return the prepared complex of a periodic square lattice."""
    return CellComplexSpace(LatticeSpace((size, size))).prepare()


def test_consistent_exchange_rates_are_a_gradient():
    """Test that the logarithms of consistent rates among 6 currencies recover `log p` up to a constant."""
    space = CellComplexSpace(NetworkSpace(nx.complete_graph(6)), faces='cliques').prepare()
    prices = np.random.default_rng(0).uniform(0.5, 2.0, 6)
    sources, targets = space.edges_[None]
    flows = np.log(prices[targets] / prices[sources])
    hodge = cell_complex(flows, kind='hodge', space=space)
    assert np.abs(hodge.curl).max() < 1e-12
    assert np.abs(hodge.harmonic).max() < 1e-12
    assert np.allclose(hodge.gradient[0], flows, atol=1e-12)
    assert np.allclose(hodge.potential[0], np.log(prices) - np.log(prices).mean(), atol=1e-12)


def test_uniform_flow_around_a_torus_is_harmonic_with_the_betti_numbers_of_a_torus():
    """Test that a uniform flow along the first axis of a periodic 8 by 8 complex is harmonic, with Betti 1, 2, 1."""
    space = _square()
    flow = np.where(space.link_axes_ == 0, 1.0, 0.0)
    hodge = cell_complex(flow, kind='hodge', space=space)
    assert np.allclose(hodge.harmonic[0], flow, atol=1e-12)
    assert np.abs(hodge.gradient).max() < 1e-12
    assert np.abs(hodge.curl).max() < 1e-12
    assert cell_complex(None, kind='betti', space=space).betti == (1, 2, 1)


def test_parts_of_a_random_flow_are_orthogonal_and_sum_to_it():
    """Test the three parts of a random flow on a lattice with one frame per row."""
    space = _square(5)
    flows = np.random.default_rng(1).normal(size=(3, 50))
    hodge = cell_complex(flows, kind='hodge', space=space)
    assert np.allclose(hodge.gradient + hodge.curl + hodge.harmonic, flows)
    for first, second in (
        (hodge.gradient, hodge.curl),
        (hodge.gradient, hodge.harmonic),
        (hodge.curl, hodge.harmonic),
    ):
        assert np.abs((first * second).sum(axis=1)).max() < 1e-9


def test_decomposition_of_a_complex_without_faces_has_no_curl():
    """Test that a ring without faces has a harmonic flow around it."""
    space = CellComplexSpace(NetworkSpace(nx.cycle_graph(4)), faces=[]).prepare()
    hodge = cell_complex(np.array([1.0, -1.0, -1.0, 1.0]), kind='hodge', space=space)
    assert np.abs(hodge.curl).max() == 0.0
    assert cell_complex(None, kind='betti', space=space).betti == (1, 1, 0)


def test_flat_and_curved_connections_are_told_apart():
    """Test that a flat angle state has no curved face, and a state with a sum of pi around one face has one."""
    space = _square()
    sources, targets = space.edges_[None]
    potential = np.random.default_rng(2).uniform(-3.0, 3.0, 64)
    flat = potential[targets] - potential[sources]
    curvature = cell_complex(flat, kind='curvature', space=space)
    assert curvature.share[0] == 0.0
    assert curvature.energy[0] == pytest.approx(0.0, abs=1e-12)
    boundary = CellComplexSpace(LatticeSpace((8, 8), boundary='fixed')).prepare()
    corner = np.flatnonzero(np.diff(boundary.incidences_[1].T.tocsr().indptr) == 1)[0]
    theta = np.zeros(boundary.incidences_[0].shape[0])
    theta[corner] = np.pi
    state = cell_complex(theta, kind='curvature', space=boundary)
    assert state.share[0] * boundary.incidences_[1].shape[0] == pytest.approx(1.0)
    assert state.energy[0] == pytest.approx(2.0)


def test_curvature_of_a_trajectory_reads_the_group_and_its_period():
    """Test that a trajectory gives the complex of the group and one row per frame."""
    space = _square(4)
    links = GroupElement('links', element=AngleLinkElement, space=space, on='edges')
    rate = RateInteraction('rate', target='links', state='theta', rate=0.5)
    trajectory = simulate(System(elements=[links], interactions=[rate]), until=1.0, dt=0.1)
    curvature = cell_complex(trajectory, kind='curvature', variables='links.theta')
    assert curvature.share.shape == trajectory.times.shape
    assert np.all(curvature.share == 0.0)


@pytest.mark.parametrize(
    ('settings', 'source', 'message'),
    [
        ({}, np.zeros(4), 'no complex'),
        ({'space': _square(3)}, np.zeros(4), 'edges'),
    ],
)
def test_prepare_rejects_a_source_without_a_complex_or_of_another_size(settings, source, message):
    """Test that values need a complex of as many edges."""
    with pytest.raises(AnalysisError, match=message):
        cell_complex(source, kind='hodge', **settings)


def test_check_rejects_an_unknown_kind_and_a_space_that_is_no_complex():
    """Test that an unknown kind or another space raise an error."""
    with pytest.raises(AnalysisError, match='kind'):
        cell_complex(None, kind='volume')
    with pytest.raises(AnalysisError, match='CellComplexSpace'):
        cell_complex(None, kind='betti', space=LatticeSpace((3, 3)))
