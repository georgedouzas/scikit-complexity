"""Test the `_cell_complex` module of `skcomplex.analysis`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import pi

from skcomplex.analysis import CellComplexAnalysis
from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import RateInteraction
from skcomplex.core.spaces import CellComplexSpace, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import AnalysisError
from skcomplex.specs import StateSpec


class AngleLinkElement(Element):
    """Link with an angle `theta`."""

    def __init__(self, label, visual=None, *, theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0):
        super().__init__(label, visual=visual)
        self.theta = theta


def _square(size=8):
    """Return the prepared complex of a periodic square lattice."""
    return CellComplexSpace(LatticeSpace((size, size))).check().prepare()


def test_consistent_exchange_rates_are_a_gradient():
    """Test that the logarithms of consistent rates among 6 currencies recover `log p` up to a constant."""
    space = CellComplexSpace(NetworkSpace(nx.complete_graph(6)), faces='cliques').check().prepare()
    prices = np.random.default_rng(0).uniform(0.5, 2.0, 6)
    sources, targets = space.edges_[None]
    flows = np.log(prices[targets] / prices[sources])
    hodge = CellComplexAnalysis('hodge', space=space).check().prepare(flows).run()
    assert np.abs(hodge.curl_).max() < 1e-12
    assert np.abs(hodge.harmonic_).max() < 1e-12
    assert np.allclose(hodge.gradient_[0], flows, atol=1e-12)
    assert np.allclose(hodge.potential_[0], np.log(prices) - np.log(prices).mean(), atol=1e-12)


def test_uniform_flow_around_a_torus_is_harmonic_with_the_betti_numbers_of_a_torus():
    """Test that a uniform flow along the first axis of a periodic 8 by 8 complex is harmonic, with Betti 1, 2, 1."""
    space = _square()
    flow = np.where(space.link_axes_ == 0, 1.0, 0.0)
    hodge = CellComplexAnalysis('hodge', space=space).check().prepare(flow).run()
    assert np.allclose(hodge.harmonic_[0], flow, atol=1e-12)
    assert np.abs(hodge.gradient_).max() < 1e-12
    assert np.abs(hodge.curl_).max() < 1e-12
    assert CellComplexAnalysis('betti', space=space).check().prepare().run().betti_ == (1, 2, 1)


def test_parts_of_a_random_flow_are_orthogonal_and_sum_to_it():
    """Test the three parts of a random flow on a lattice with one frame per row."""
    space = _square(5)
    flows = np.random.default_rng(1).normal(size=(3, 50))
    hodge = CellComplexAnalysis('hodge', space=space).check().prepare(flows).run()
    assert np.allclose(hodge.gradient_ + hodge.curl_ + hodge.harmonic_, flows)
    for first, second in (
        (hodge.gradient_, hodge.curl_),
        (hodge.gradient_, hodge.harmonic_),
        (hodge.curl_, hodge.harmonic_),
    ):
        assert np.abs((first * second).sum(axis=1)).max() < 1e-9


def test_decomposition_of_a_complex_without_faces_has_no_curl():
    """Test that a ring without faces has a harmonic flow around it."""
    space = CellComplexSpace(NetworkSpace(nx.cycle_graph(4)), faces=[]).check().prepare()
    hodge = CellComplexAnalysis('hodge', space=space).check().prepare(np.array([1.0, -1.0, -1.0, 1.0])).run()
    assert np.abs(hodge.curl_).max() == 0.0
    assert CellComplexAnalysis('betti', space=space).check().prepare().run().betti_ == (1, 1, 0)


def test_flat_and_curved_connections_are_told_apart():
    """Test that a flat angle state has no curved face, and a state with a sum of pi around one face has one."""
    space = _square()
    sources, targets = space.edges_[None]
    potential = np.random.default_rng(2).uniform(-3.0, 3.0, 64)
    flat = potential[targets] - potential[sources]
    curvature = CellComplexAnalysis('curvature', space=space).check().prepare(flat).run()
    assert curvature.share_[0] == 0.0
    assert curvature.energy_[0] == pytest.approx(0.0, abs=1e-12)
    boundary = CellComplexSpace(LatticeSpace((8, 8), boundary='fixed')).check().prepare()
    corner = np.flatnonzero(np.diff(boundary.incidences_[1].T.tocsr().indptr) == 1)[0]
    theta = np.zeros(boundary.incidences_[0].shape[0])
    theta[corner] = np.pi
    state = CellComplexAnalysis('curvature', space=boundary).check().prepare(theta).run()
    assert state.share_[0] * boundary.incidences_[1].shape[0] == pytest.approx(1.0)
    assert state.energy_[0] == pytest.approx(2.0)


def test_curvature_of_a_trajectory_reads_the_group_and_its_period():
    """Test that a trajectory gives the complex of the group and one row per frame."""
    space = _square(4)
    links = GroupElement('links', element=AngleLinkElement, space=space, on='edges')
    rate = RateInteraction('rate', target='links', state='theta', rate=0.5)
    trajectory = (
        Simulation(System(elements=[links], interactions=[rate]), dt=0.1).check().prepare().run(until=1.0).trajectory_
    )
    curvature = CellComplexAnalysis('curvature', variables='links.theta').check().prepare(trajectory).run()
    assert curvature.share_.shape == trajectory.times.shape
    assert np.all(curvature.share_ == 0.0)


@pytest.mark.parametrize(
    ('analysis', 'source', 'message'),
    [
        (CellComplexAnalysis('hodge'), np.zeros(4), 'no complex'),
        (CellComplexAnalysis('hodge', space=_square(3)), np.zeros(4), 'edges'),
    ],
)
def test_prepare_rejects_a_source_without_a_complex_or_of_another_size(analysis, source, message):
    """Test that values need a complex of as many edges."""
    with pytest.raises(AnalysisError, match=message):
        analysis.check().prepare(source)


def test_check_rejects_an_unknown_kind_and_a_space_that_is_no_complex():
    """Test that an unknown kind or another space raise an error."""
    with pytest.raises(AnalysisError, match='kind'):
        CellComplexAnalysis('volume').check()
    with pytest.raises(AnalysisError, match='CellComplexSpace'):
        CellComplexAnalysis('betti', space=LatticeSpace((3, 3))).check()
