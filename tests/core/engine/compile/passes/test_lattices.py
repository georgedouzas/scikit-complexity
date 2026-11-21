"""Test triangular lattices, lattice laws on graphs, neighbour sums, and colored schedules."""

import itertools
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import (
    LinearRuleSpec,
    ObjectiveLaw,
    OuterTotalisticSpec,
    TableSpec,
    TotalisticSpec,
    UpdateLaw,
)
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import (
    AdjacencyNeighborhood,
    LatticeNeighborhood,
    LatticeSpace,
    NetworkSpace,
    RadiusNeighborhood,
)
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.engine.viz import RerunObserver, SiteVisual
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import sweep
from skcomplex.core.systems import System
from skcomplex.core.types import Array, Role
from skcomplex.systems.physics.criticality import HexagonalForestFireSystem
from skcomplex.systems.physics.statistical_mechanics import GraphIsingSystem

LIFE = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})
MOORE = RadiusNeighborhood(1, metric='chebyshev')


def _local(label, state, rule, neighborhood, space, *, schedule=None, **values):
    """Return a group on a space whose value `state` follows a local rule of its neighbourhood."""
    local = Interaction(
        'rule',
        roles={'target': label},
        laws=lambda s: [UpdateLaw(getattr(s.target, state), rule, neighborhood=neighborhood, schedule=schedule)],
    )
    return System(elements=[Element(label, values={state: 0.0} | values, space=space)], interactions=[local])


def _ising_magnetization(trajectory):
    """Return the mean absolute magnetisation of the recorded frames."""
    return float(np.abs(trajectory.get_values('spins.spin').mean(axis=1)).mean())


class Recording:
    """Recording stand-in that keeps what is logged to it."""

    def __init__(self):
        self.entries = []

    def log(self, path, archetype, static=False):
        """Keep a logged archetype with its path."""
        self.entries.append((path, archetype, static))


def test_triangular_neighbours_wrap_on_torus_at_unit_distance():
    """Test that the six triangular neighbours of the corner of a periodic 10 x 10 lattice lie at distance 1."""
    space = LatticeSpace((10, 10), geometry='triangular').prepare()
    neighbours = [
        tuple(np.unravel_index(site, space.shape_))
        for site in space._get_neighbors(0, RadiusNeighborhood(1, metric='hexagonal'))
    ]
    assert len(set(neighbours)) == 6
    assert {(0, 9), (9, 0), (9, 1)} <= set(neighbours)
    positions = space.positions_.reshape(10, 10, 2)
    rows, columns = np.array([5.0, 10 * np.sqrt(3) / 2]), np.array([10.0, 0.0])
    images = [i * rows + j * columns for i, j in itertools.product((-1, 0, 1), repeat=2)]
    for cell in neighbours:
        distance = min(np.linalg.norm(positions[cell] + image - positions[0, 0]) for image in images)
        assert distance == pytest.approx(1.0)


def test_triangular_opposite_directions_return_to_start():
    """Test that a step in direction k and a step in direction k + 3 return to the starting site."""
    space = LatticeSpace((7, 9), geometry='triangular').prepare()
    table = space._get_index_table(RadiusNeighborhood(1, metric='hexagonal'))
    for site in range(space.size_):
        for k in range(6):
            assert table[table[site, k], (k + 3) % 6] == site


def test_triangular_geometry_needs_two_dimensions():
    """Test that a triangular lattice of another dimension is rejected."""
    with pytest.raises(ValidationError, match='two dimensions'):
        LatticeSpace((5, 5, 5), geometry='triangular').prepare()


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_triangular_rule_spreads_as_hexagons(backend):
    """Test that a fire on a periodic 40 x 40 triangular lattice covers the 3k(k + 1) + 1 sites within k steps."""
    fire = np.zeros((40, 40))
    fire[20, 20] = 1
    system = _local(
        'forest',
        'fire',
        lambda *values: np.max(values, axis=0),
        RadiusNeighborhood(1, metric='hexagonal', center=True),
        LatticeSpace((40, 40), geometry='triangular'),
        fire=fire,
    )
    burning = system.set_params(backend=backend()).prepare().simulate(until=10, dt=1).get_lattice('forest.fire')
    assert [int(item.sum()) for item in burning] == [3 * k * (k + 1) + 1 for k in range(11)]


def test_network_index_table_pads_with_size():
    """Test that the adjacency table of a star graph pads the leaves with the number of nodes."""
    space = NetworkSpace(nx.star_graph(3)).prepare()
    table = space._get_index_table(AdjacencyNeighborhood(center=True))
    assert table.tolist() == [[0, 1, 2, 3], [1, 0, 4, 4], [2, 0, 4, 4], [3, 0, 4, 4]]
    with pytest.raises(ValidationError, match='adjacency neighbourhood'):
        space._get_neighbors(0, RadiusNeighborhood(1, metric='chebyshev'))


def test_grid_graph_matches_square_lattice():
    """Test that a glider gives the same configurations on a periodic lattice and on its Moore graph for 40 steps."""
    lattice = LatticeSpace((20, 20)).prepare()
    table = lattice._get_index_table(RadiusNeighborhood(1, metric='chebyshev'))
    graph = nx.Graph()
    graph.add_nodes_from(range(lattice.size_))
    graph.add_edges_from((site, int(other)) for site, row in enumerate(table) for other in row)
    glider = np.zeros((20, 20))
    glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
    on_lattice = _local('cells', 'alive', LIFE, MOORE, lattice, alive=glider)
    on_graph = _local('cells', 'alive', LIFE, AdjacencyNeighborhood(), NetworkSpace(graph), alive=glider.ravel())
    expected = on_lattice.prepare().simulate(until=40, dt=1).get_values('cells.alive')
    found = on_graph.prepare().simulate(until=40, dt=1).get_values('cells.alive')
    assert np.array_equal(found, expected)
    assert expected[-1].sum() == 5


def test_totalistic_rule_counts_real_neighbours_on_star():
    """Test that a totalistic count on a star with 5 leaves gives every leaf 1 and the centre its live leaves."""

    values = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    rule = TotalisticSpec(lambda total: total)
    system = _local('star', 'value', rule, AdjacencyNeighborhood(), NetworkSpace(nx.star_graph(5)), value=values)
    assert system.prepare().simulate(until=1, dt=1).states[-1].tolist() == [2, 1, 1, 1, 1, 1]


def test_function_rule_reads_degrees_on_star():
    """Test that a function rule naming `degree` receives the number of neighbours of every node."""

    system = _local(
        'star',
        'value',
        lambda *values, degree: degree,
        AdjacencyNeighborhood(center=True),
        NetworkSpace(nx.star_graph(5)),
    )
    assert system.prepare().simulate(until=1, dt=1).states[-1].tolist() == [5, 1, 1, 1, 1, 1]


def test_lattice_groups_on_networks_reject_tables_and_capacities():
    """Test that a table of ordered patterns and a capacity are rejected on a network."""

    table = TableSpec({(1, 1): 1})
    system = _local('nodes', 'value', table, AdjacencyNeighborhood(), NetworkSpace(nx.cycle_graph(4)))
    with pytest.raises(ValidationError, match='totalistic'):
        system.prepare()
    system = _local('nodes', 'value', table, AdjacencyNeighborhood(), NetworkSpace(nx.cycle_graph(4), capacity=1))
    with pytest.raises(ValidationError, match='capacity'):
        system.prepare()


def test_neighbor_sum_ignores_padding():
    """Test that the XY neighbour-sum energy change on a star with 4 leaves sums over the real neighbours only."""

    def laws(s):
        """Return the XY energy of a node, sampled by rotations."""
        theta = s.target.theta
        return [ObjectiveLaw(-s.target.J * neighbors(cos(theta - s.target.other.theta), reduce='sum'), theta)]

    graph = nx.star_graph(4)
    angles = np.random.default_rng(0).uniform(0, 2 * np.pi, 5)
    xy = Element('xy', values={'theta': angles, 'J': 1.3}, space=NetworkSpace(graph))
    solver = MetropolisSolver(moves={'xy.theta': 'rotate'}, schedule='colored', neighborhood=AdjacencyNeighborhood())
    system = System(
        elements=[xy],
        interactions=[Interaction('energy', roles={'target': 'xy'}, laws=laws)],
        solver=solver,
    )
    (sampling,) = system.prepare()._problem.lattice
    proposal = angles + np.random.default_rng(1).uniform(-1, 1, 5)
    neighbours = np.append(angles, 0.0)[sampling.table]
    masks = (sampling.table != 5).astype(float)
    change = sampling.delta(0.0, angles, *neighbours.T, proposal, *masks.T, 1.3)[0]
    expected = [
        -1.3 * sum(np.cos(proposal[node] - angles[j]) - np.cos(angles[node] - angles[j]) for j in graph[node])
        for node in graph.nodes
    ]
    assert np.allclose(change, expected, rtol=0, atol=1e-12)


def test_ising_on_grid_graph_has_transition():
    """Test that the ensemble magnetisation on the periodic 64 x 64 grid graph falls below one half from 2.2 to 2.4."""
    graph = nx.grid_2d_graph(64, 64, periodic=True)
    temperatures = [1.5, 2.2, 2.4, 3.5]
    result = sweep(
        lambda temperature: GraphIsingSystem(graph, temperature=temperature, initial='up'),
        {'temperature': temperatures},
        replicates=4,
        until=1000,
        dt=50,
        warmup=200,
        outputs=[_ising_magnetization],
        random_state=0,
    )
    means = {row['temperature']: row['_ising_magnetization.mean'] for row in result.get_summary()}
    assert means[2.2] > 0.5 > means[2.4]
    assert means[1.5] > 0.95
    assert means[3.5] < 0.1


def _hexagonal_spins(space):
    """Return the Ising spins of a triangular lattice sampled in the classes of a proper coloring."""
    hexagonal = RadiusNeighborhood(1, metric='hexagonal')

    def laws(s):
        """Return the energy of a spin over the temperature."""
        spin = s.target.spin
        return [ObjectiveLaw(-spin * sum(neighbors(spin, hexagonal, reduce='each')) / s.target.temperature, spin)]

    spins = Element('spins', values={'spin': 1.0, 'temperature': 2.0}, space=space)
    solver = MetropolisSolver(moves={'spins.spin': 'flip'}, schedule='colored', neighborhood=hexagonal)
    return System(
        elements=[spins],
        interactions=[Interaction('ising', roles={'target': 'spins'}, laws=laws)],
        solver=solver,
    )


def test_colored_schedule_partitions_triangular_lattice():
    """Test that the colored classes of a 12 x 12 triangular lattice hold no neighbours and visit every site once."""
    space = LatticeSpace((12, 12), geometry='triangular')
    (sampling,) = _hexagonal_spins(space).prepare()._problem.lattice
    sites = np.concatenate(sampling.classes)
    assert np.array_equal(np.sort(sites), np.arange(144))
    for members in sampling.classes:
        assert not np.isin(sampling.table[members], members).any()
    _hexagonal_spins(space).prepare().simulate(until=2, dt=1)


@pytest.mark.parametrize('space', [LatticeSpace((6, 6), geometry='triangular'), NetworkSpace(nx.cycle_graph(6))])
def test_checkerboard_rejected_where_it_does_not_apply(space):
    """Test that the checkerboard schedule on a triangular lattice or a graph names the law and the colored schedule."""
    neighborhood = (
        RadiusNeighborhood(1, metric='hexagonal') if isinstance(space, LatticeSpace) else AdjacencyNeighborhood()
    )

    system = _local('spins', 'spin', lambda *values: -values[0], neighborhood, space, schedule='checkerboard', spin=1.0)
    with pytest.raises(ValidationError, match=r"of rule.*'colored'"):
        system.prepare()


def test_triangular_forest_fire_streams_hexagonal_cells(tmp_path):
    """Test that a triangular forest fire logs one colored cell per site at its position at every logged step."""
    trajectory = HexagonalForestFireSystem((12, 10)).prepare().simulate(until=1, dt=1)
    frame = FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], trajectory.problem)
    recording = Recording()
    (SiteVisual(path=trajectory.problem.elements[0].path, recording=recording).prepare(trajectory.problem).run(frame))
    ((path, points, _),) = recording.entries
    assert path == 'forest/cover'
    assert type(points).__name__ == 'Points2D'
    positions = np.asarray(points.positions.as_arrow_array().to_numpy(zero_copy_only=False).tolist())
    assert np.allclose(positions, LatticeSpace((12, 10), geometry='triangular').prepare().positions_)
    assert len(points.colors.as_arrow_array()) == 120
    observer = RerunObserver('forest', save_path=tmp_path / 'forest.rrd')
    HexagonalForestFireSystem((12, 10)).prepare().simulate(until=3, dt=1, observers=[observer])
    assert observer.logged_frames_ == 4
    assert (tmp_path / 'forest.rrd').stat().st_size > 0


def test_graph_lattices_stream_nodes_and_static_edges():
    """Test that a lattice group on a graph logs its edges once and its nodes at every frame."""
    trajectory = GraphIsingSystem(nx.cycle_graph(8)).prepare().simulate(until=1, dt=1)
    frame = FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], trajectory.problem)
    recording = Recording()
    visual = SiteVisual(low=-1, high=1, path=trajectory.problem.elements[0].path, recording=recording)
    visual.prepare(trajectory.problem)
    visual.run(frame)
    visual.run(frame)
    kinds = [(type(archetype).__name__, static) for _, archetype, static in recording.entries]
    assert kinds == [('GraphEdges', True), ('GraphNodes', False), ('GraphNodes', False)]


def test_linear_rule_runs_as_its_law_steps():
    """Test that a linear rule compiles to a function rule, and that a run agrees with the law of Life and rule 90."""
    start = np.array([0, 0, 1, 0, 0, 0])
    rule90 = LinearRuleSpec((1, 1), 'GF(2)')
    system = _local('cells', 'alive', rule90, LatticeNeighborhood(((-1,), (1,))), LatticeSpace(6), alive=start)
    trajectory = system.prepare().simulate(until=2, dt=1)
    values = trajectory.get_values('cells.alive')
    assert values[1].astype(int).tolist() == [0, 1, 0, 1, 0, 0]
    assert values[2].astype(int).tolist() == [1, 0, 0, 0, 1, 0]


def test_life_simulation_agrees_with_the_run_of_its_law():
    """Test that a simulation of Life and the run of its law give the same next configuration."""
    start = np.zeros(25)
    start[[1, 7, 10, 11, 12]] = 1
    rule = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})
    space = LatticeSpace((5, 5))
    system = _local('cells', 'alive', rule, RadiusNeighborhood(1, metric='chebyshev'), space, alive=start)
    trajectory = system.prepare().simulate(until=1, dt=1)
    law = UpdateLaw(var('alive'), rule, neighborhood=RadiusNeighborhood(1, metric='chebyshev'))
    image = law._get_value({'space': space, 'alive': start.astype(int)})
    np.testing.assert_array_equal(trajectory.get_values('cells.alive')[-1].astype(int), image)


class CellElement(Element):
    """Plain cell whose value `alive` a local rule changes."""

    def __init__(self, label: str, *, alive: float | Array = 0.0, space=None) -> None:
        super().__init__(label, space=space)
        self.alive = alive


class LifeInteraction(Interaction):
    """Game of Life written as an update with a neighbourhood."""

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context):
        """Set the life rule of the cells."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target.alive, LIFE, neighborhood=MOORE)]


def _life_step(alive):
    """Return the next configuration of Life on a periodic grid, from the counts of the eight neighbours."""
    around = sum(np.roll(alive, (row, column), axis=(0, 1)) for row in (-1, 0, 1) for column in (-1, 0, 1)) - alive
    return ((around == 3) | ((alive == 1) & (around == 2))).astype(float)


def test_update_with_a_neighborhood_lowers_to_the_kernel_of_a_local_rule():
    """Test that Life as an update with a neighbourhood compiles to a fused outer totalistic lookup and runs Life."""
    start = (np.random.default_rng(0).random(1600) < 0.3).astype(float)
    space = LatticeSpace((40, 40))
    new = System(elements=[CellElement('cells', alive=start, space=space)], interactions=[LifeInteraction('life')])
    (rule,) = new.prepare()._problem.lattice
    assert (rule.lookup.kind, rule.fused) == ('outer', True)
    expected = [start.reshape(40, 40)]
    for _ in range(20):
        expected.append(_life_step(expected[-1]))
    np.testing.assert_array_equal(
        new.simulate(until=20, dt=1).get_values('cells.alive'),
        np.reshape(expected, (21, 1600)),
    )
