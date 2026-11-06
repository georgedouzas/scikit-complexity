"""Test the `_yang_mills` module of `skcomplex.systems.physics.waves`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import AnalysisError, ValidationError
from skcomplex.core.specs import MatrixGroupSpec
from skcomplex.systems import create_simulation, describe
from skcomplex.systems.physics.waves import GaugeAnalysis, lattice_yang_mills

ISOSPIN = MatrixGroupSpec('SU', 2)
COLOR = MatrixGroupSpec('SU', 3)


def _space(shape):
    """Return the prepared complex of a periodic lattice."""
    return CellComplexSpace(LatticeSpace(shape)).prepare()


def _random_links(space, group, seed=0):
    """Return random link matrices of a complex."""
    return group.uniform_of(np.random.default_rng(seed), space.incidences_[0].shape[0])


def _gauge_transform(links, space, group, seed=1):
    """Return link matrices after a random gauge transformation of the sites, `g_x U g_y^-1` on a link from x to y."""
    gauge = group.uniform_of(np.random.default_rng(seed), space.size_)
    sources, targets = space.edges_[None]
    return gauge[sources] @ links @ np.conj(np.swapaxes(gauge[targets], -1, -2))


def test_four_dimensional_su2_plaquette_is_near_the_known_value():
    """Test that SU(2) at `beta = 2.3` on a 4^4 lattice has a mean plaquette within 0.03 of 0.60."""
    series = create_simulation('lattice_yang_mills', size=4).prepare().run(until=100).trajectory_
    assert abs(series.values('mean_plaquette')[30:].mean() - 0.60) < 0.03


def test_polyakov_loop_orders_above_the_deconfinement_coupling():
    """Test that the SU(2) Polyakov loop with two time slices is three times larger at `beta = 2.6` than at 1.5."""
    loops = []
    for beta in (1.5, 2.6):
        simulation = create_simulation('lattice_yang_mills', size=4, extent=2, beta=beta).prepare()
        loops.append(simulation.run(until=100).trajectory_.values('polyakov_loop')[30:].mean())
    assert loops[1] > 3 * loops[0]


def test_su3_links_stay_in_the_group_and_order_from_a_cold_start():
    """Test that SU(3) links from the identity start at plaquette 1, fall at small beta, and stay in SU(3)."""
    space = _space((4, 4, 4))
    cold = np.broadcast_to(np.eye(3), (space.incidences_[0].shape[0], 3, 3))
    system = lattice_yang_mills(4, dimension=3, colors=3, beta=1.0, links=cold)
    observables = describe('lattice_yang_mills').observables_of(system)
    trajectory = Simulation(system, dt=1, observables=observables, random_state=0).prepare().run(until=10).trajectory_
    plaquettes = trajectory.values('mean_plaquette')
    assert plaquettes[0] == pytest.approx(1.0)
    assert plaquettes[-1] < 0.5
    matrices = COLOR.matrices_of(trajectory.values('links.U')[-1])
    assert np.allclose(COLOR.projection_of(matrices), matrices, atol=1e-12)


@pytest.mark.parametrize(('group', 'kind'), [(ISOSPIN, 'wilson'), (COLOR, 'wilson'), (COLOR, 'polyakov')])
def test_loops_of_matrix_links_are_gauge_invariant(group, kind):
    """Test that Wilson loops of 2 by 1 rectangles and Polyakov loops are unchanged by a gauge transformation."""
    space = _space((4, 4, 3))
    links = _random_links(space, group)
    values = [
        GaugeAnalysis(kind, space, extents=(2, 1), group=group).prepare(group.components_of(item)).run().value_[0]
        for item in (links, _gauge_transform(links, space, group))
    ]
    assert values[0] == pytest.approx(values[1], abs=1e-12)


def test_plaquette_loop_of_matrix_links_is_the_mean_plaquette():
    """Test that the 1 by 1 Wilson loop of random SU(3) links is the mean of `Re tr U_p / 3`."""
    space = _space((4, 4))
    links = _random_links(space, COLOR)
    plaquettes = COLOR.holonomy_of(links, space.boundaries_, space.boundary_signs_)
    expected = np.trace(plaquettes, axis1=1, axis2=2).real.mean() / 3
    loop = GaugeAnalysis('wilson', space, group=COLOR).prepare(COLOR.components_of(links)).run().value_[0]
    assert loop == pytest.approx(expected, abs=1e-12)


def test_polyakov_loop_of_a_uniform_field_is_its_trace():
    """Test that SU(2) links of `i sigma_3` along the last axis of extent 2 give a Polyakov loop of modulus 1."""
    space = _space((3, 2))
    links = np.broadcast_to(np.eye(2, dtype=complex), (space.incidences_[0].shape[0], 2, 2)).copy()
    links[space.link_axes_ == 1] = np.diag([1j, -1j])
    analysis = GaugeAnalysis('polyakov', space, group=ISOSPIN).prepare(ISOSPIN.components_of(links)[None])
    assert analysis.run().value_.tolist() == pytest.approx([1.0])


def test_analysis_reads_matrix_links_from_a_trajectory():
    """Test that a trajectory of matrix links gives one loop per frame."""
    system = lattice_yang_mills(3, dimension=2)
    trajectory = Simulation(system, dt=1, random_state=0).prepare().run(until=3).trajectory_
    analysis = GaugeAnalysis('wilson', _space((3, 3)), variables='links.U', group=ISOSPIN).prepare(trajectory)
    assert analysis.run().value_.shape == (4,)


@pytest.mark.parametrize('kind', ['charge', 'gauss'])
def test_analysis_of_matrix_links_takes_only_loops(kind):
    """Test that the charge and the Gauss residual raise an error for matrix links."""
    with pytest.raises(AnalysisError, match='Matrix links'):
        GaugeAnalysis(kind, _space((4, 4)), group=ISOSPIN).prepare()


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'colors': 4}, 'SU\\(2\\) or SU\\(3\\)'),
        ({'links': np.zeros((8, 2, 2))}, 'one per link'),
        ({'links': 2 * np.broadcast_to(np.eye(2), (8, 2, 2))}, 'one per link'),
    ],
)
def test_system_rejects_groups_and_links_it_cannot_hold(arguments, match):
    """Test that a group other than SU(2) and SU(3), or initial links that are not matrices of it, raise an error."""
    with pytest.raises(ValidationError, match=match):
        lattice_yang_mills(2, dimension=2, **arguments)
