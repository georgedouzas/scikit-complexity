"""Test triangular lattices, lattice laws on graphs, neighbour sums, colored schedules, and extremal updates."""

import itertools

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos

from skcomplex.core.base import ValidationError
from skcomplex.core.compile import CompiledExtremal, ExtremalRecord, compile_system
from skcomplex.core.experiments import Experiment, Grid
from skcomplex.core.model import (
    Adjacency,
    Element,
    Energy,
    Extremal,
    LatticeSpace,
    LocalRule,
    Move,
    Neighborhood,
    NetworkSpace,
    OuterTotalistic,
    Sampler,
    System,
    Table,
    Totalistic,
    adjacent,
    neighbor_sum,
    neighbors,
)
from skcomplex.core.runtime import Frame, Simulation
from skcomplex.core.viz import RerunObserver, SiteVisual
from skcomplex.models.physics.criticality import hexagonal_forest_fire
from skcomplex.models.physics.statistical_mechanics import graph_ising

Burn = Element.define(
    'Burn',
    states={'fire': 0.0},
    laws=lambda s: [LocalRule(s.fire, Neighborhood.triangular(center=True), lambda *values: np.max(values, axis=0))],
)


def _life(neighborhood):
    """Return a Game of Life element type on a neighbourhood."""
    return Element.define(
        'Life',
        states={'alive': 0.0},
        laws=lambda s: [LocalRule(s.alive, neighborhood, OuterTotalistic.life())],
    )


def _renew(value, random):
    """Return new uniform values."""
    return random


Fitness = Element.define(
    'Fitness',
    states={'value': 0.0},
    laws=lambda s: [Extremal(s.value, Neighborhood(((-1,), (1,))), _renew, repeat=7, label='update')],
)


def _ising_magnetization(trajectory):
    """Return the mean absolute magnetisation of the recorded frames."""
    return float(np.abs(trajectory['spins.spin'].mean(axis=1)).mean())


class Recording:
    """Recording stand-in that keeps what is logged to it."""

    def __init__(self):
        """Start with no entries."""
        self.entries = []

    def log(self, path, archetype, static=False):
        """Keep a logged archetype with its path."""
        self.entries.append((path, archetype, static))


def test_triangular_neighbours_wrap_on_torus_at_unit_distance():
    """Test that the six triangular neighbours of the corner of a periodic 10 x 10 lattice lie at distance 1."""
    space = LatticeSpace((10, 10), geometry='triangular')
    neighbours = space.neighbors((0, 0), Neighborhood.triangular())
    assert len(set(neighbours)) == 6
    assert {(0, 9), (9, 0), (9, 1)} <= set(neighbours)
    positions = space.positions.reshape(10, 10, 2)
    rows, columns = np.array([5.0, 10 * np.sqrt(3) / 2]), np.array([10.0, 0.0])
    images = [i * rows + j * columns for i, j in itertools.product((-1, 0, 1), repeat=2)]
    for cell in neighbours:
        distance = min(np.linalg.norm(positions[cell] + image - positions[0, 0]) for image in images)
        assert distance == pytest.approx(1.0)


def test_triangular_opposite_directions_return_to_start():
    """Test that a step in direction k and a step in direction k + 3 return to the starting site."""
    space = LatticeSpace((7, 9), geometry='triangular')
    offsets = Neighborhood.triangular().offsets
    for cell in itertools.product(range(7), range(9)):
        for k in range(6):
            assert space.neighbor(space.neighbor(cell, offsets[k]), offsets[(k + 3) % 6]) == cell


def test_triangular_geometry_needs_two_dimensions():
    """Test that a triangular lattice of another dimension is rejected."""
    with pytest.raises(ValueError, match='two dimensions'):
        LatticeSpace((5, 5, 5), geometry='triangular').check()


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_triangular_rule_spreads_as_hexagons(backend):
    """Test that a fire on a periodic 40 x 40 triangular lattice covers the 3k(k + 1) + 1 sites within k steps."""
    fire = np.zeros((40, 40))
    fire[20, 20] = 1
    system = System(components=[Burn.lattice('forest', LatticeSpace((40, 40), geometry='triangular'), fire=fire)])
    burning = Simulation(system, dt=1, backend=backend).run(until=10).trajectory_.lattice('forest.fire')
    assert [int(item.sum()) for item in burning] == [3 * k * (k + 1) + 1 for k in range(11)]


def test_network_index_table_pads_with_size():
    """Test that the adjacency table of a star graph pads the leaves with the number of nodes."""
    table = NetworkSpace(nx.star_graph(3)).index_table(Adjacency(center=True))
    assert table.tolist() == [[0, 1, 2, 3], [1, 0, 4, 4], [2, 0, 4, 4], [3, 0, 4, 4]]
    with pytest.raises(ValidationError, match='Adjacency'):
        NetworkSpace(nx.star_graph(3)).index_table(Neighborhood.moore())


def test_grid_graph_matches_square_lattice():
    """Test that a glider gives the same configurations on a periodic lattice and on its Moore graph for 40 steps."""
    lattice = LatticeSpace((20, 20))
    table = lattice.index_table(Neighborhood.moore())
    graph = nx.Graph()
    graph.add_nodes_from(range(lattice.size))
    graph.add_edges_from((site, int(other)) for site, row in enumerate(table) for other in row)
    glider = np.zeros((20, 20))
    glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
    on_lattice = System(components=[_life(Neighborhood.moore()).lattice('cells', lattice, alive=glider)])
    on_graph = System(components=[_life(Adjacency()).lattice('cells', NetworkSpace(graph), alive=glider.ravel())])
    expected = Simulation(on_lattice, dt=1).run(until=40).trajectory_['cells.alive']
    found = Simulation(on_graph, dt=1).run(until=40).trajectory_['cells.alive']
    assert np.array_equal(found, expected)
    assert expected[-1].sum() == 5


def test_totalistic_rule_counts_real_neighbours_on_star():
    """Test that a totalistic count on a star with 5 leaves gives every leaf 1 and the centre its live leaves."""
    Count = Element.define(
        'Count',
        states={'value': 0.0},
        laws=lambda s: [LocalRule(s.value, Adjacency(), Totalistic(lambda total: total))],
    )
    values = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
    system = System(components=[Count.lattice('star', NetworkSpace(nx.star_graph(5)), value=values)])
    assert Simulation(system, dt=1).run(until=1).state_.tolist() == [2, 1, 1, 1, 1, 1]


def test_function_rule_reads_degrees_on_star():
    """Test that a function rule naming `degree` receives the number of neighbours of every node."""
    Degree = Element.define(
        'Degree',
        states={'value': 0.0},
        laws=lambda s: [LocalRule(s.value, Adjacency(center=True), lambda *values, degree: degree)],
    )
    system = System(components=[Degree.lattice('star', NetworkSpace(nx.star_graph(5)))])
    assert Simulation(system, dt=1).run(until=1).state_.tolist() == [5, 1, 1, 1, 1, 1]


def test_lattice_groups_on_networks_reject_tables_and_capacities():
    """Test that a table of ordered patterns and a capacity are rejected on a network."""
    Ordered = Element.define(
        'Ordered',
        states={'value': 0.0},
        laws=lambda s: [LocalRule(s.value, Adjacency(), Table({(1, 1): 1}))],
    )
    with pytest.raises(ValidationError, match='totalistic'):
        compile_system(System(components=[Ordered.lattice('nodes', NetworkSpace(nx.cycle_graph(4)))]))
    with pytest.raises(ValidationError, match='capacity'):
        compile_system(System(components=[Ordered.lattice('nodes', NetworkSpace(nx.cycle_graph(4), capacity=1))]))


def test_neighbor_sum_ignores_padding():
    """Test that the XY neighbour-sum energy change on a star with 4 leaves sums over the real neighbours only."""
    XY = Element.define(
        'XY',
        states={'theta': 0.0},
        parameters={'J': 1.3, 'temperature': 0.5},
        laws=lambda s: [
            Energy(-s.J * neighbor_sum(cos(s.theta - adjacent(s.theta))), Adjacency()),
            Move.rotate(s.theta, 1.0),
            Sampler('metropolis', s.temperature, schedule='colored'),
        ],
    )
    graph = nx.star_graph(4)
    angles = np.random.default_rng(0).uniform(0, 2 * np.pi, 5)
    (sampling,) = compile_system(System(components=[XY.lattice('xy', NetworkSpace(graph), theta=angles)])).lattice
    proposal = angles + np.random.default_rng(1).uniform(-1, 1, 5)
    neighbours = np.append(angles, 0.0)[sampling.table]
    masks = (sampling.table != 5).astype(float)
    change = sampling.delta(0.0, angles, *neighbours.T, proposal, *masks.T, 1.3, 0.5)[0]
    expected = [
        -1.3 * sum(np.cos(proposal[node] - angles[j]) - np.cos(angles[node] - angles[j]) for j in graph[node])
        for node in graph.nodes
    ]
    assert np.allclose(change, expected, rtol=0, atol=1e-12)


def test_ising_on_grid_graph_has_transition():
    """Test that the ensemble magnetisation on the periodic 64 x 64 grid graph falls below one half from 2.2 to 2.4."""
    graph = nx.grid_2d_graph(64, 64, periodic=True)
    temperatures = [1.5, 2.2, 2.4, 3.5]
    result = Experiment(
        lambda temperature: graph_ising(graph, temperature=temperature, initial='up'),
        Grid(temperature=temperatures),
        replicates=4,
        until=1000,
        dt=50,
        warmup=200,
        outputs=[_ising_magnetization],
        random_state=0,
    ).run()
    means = {row['temperature']: row['_ising_magnetization.mean'] for row in result.summary()}
    assert means[2.2] > 0.5 > means[2.4]
    assert means[1.5] > 0.95
    assert means[3.5] < 0.1


def test_colored_schedule_partitions_triangular_lattice():
    """Test that the colored classes of a 12 x 12 triangular lattice hold no neighbours and visit every site once."""
    Spin = Element.define(
        'Spin',
        states={'spin': 1.0},
        parameters={'temperature': 2.0},
        laws=lambda s: [
            Energy(-s.spin * sum(neighbors(s.spin, Neighborhood.triangular())), Neighborhood.triangular()),
            Move.flip(s.spin),
            Sampler('metropolis', s.temperature, schedule='colored'),
        ],
    )
    space = LatticeSpace((12, 12), geometry='triangular')
    (sampling,) = compile_system(System(components=[Spin.lattice('spins', space)])).lattice
    sites = np.concatenate(sampling.classes)
    assert np.array_equal(np.sort(sites), np.arange(144))
    for members in sampling.classes:
        assert not np.isin(sampling.table[members], members).any()
    Simulation(System(components=[Spin.lattice('spins', space)]), dt=1).run(until=2)


@pytest.mark.parametrize('space', [LatticeSpace((6, 6), geometry='triangular'), NetworkSpace(nx.cycle_graph(6))])
def test_checkerboard_rejected_where_it_does_not_apply(space):
    """Test that the checkerboard schedule on a triangular lattice or a graph names the law and the colored schedule."""
    neighborhood = Neighborhood.triangular() if isinstance(space, LatticeSpace) else Adjacency()
    Flip = Element.define(
        'Flip',
        states={'spin': 1.0},
        laws=lambda s: [LocalRule(s.spin, neighborhood, lambda *values: -values[0], schedule='checkerboard')],
    )
    with pytest.raises(ValidationError, match=r"local rule of spins.*'colored'"):
        compile_system(System(components=[Flip.lattice('spins', space)]))


def test_extremal_changes_only_the_neighbourhood():
    """Test that one extremal update on a ring of 100 changes the minimum and its two neighbours only."""
    Once = Element.define(
        'Once',
        states={'value': 0.0},
        laws=lambda s: [Extremal(s.value, Neighborhood.radius(), _renew)],
    )
    values = np.random.default_rng(0).random(100)
    simulation = Simulation(System(components=[Once.lattice('ring', LatticeSpace(100), value=values)]), dt=1)
    simulation.run(until=1)
    site = int(np.argmin(values))
    changed = np.flatnonzero(simulation.state_ != values)
    assert sorted(changed.tolist()) == sorted([(site - 1) % 100, site, (site + 1) % 100])
    (event,) = simulation.trajectory_.events
    assert event == ExtremalRecord('ring', 1.0, site, float(values[site]))


def test_extremal_maximum_and_ties():
    """Test that an extremal law on the largest value breaks ties among equal values with the random stream."""
    Top = Element.define(
        'Top',
        states={'value': 0.0},
        laws=lambda s: [Extremal(s.value, Adjacency(), _renew, extremum='max', repeat=20)],
    )
    system = System(components=[Top.lattice('nodes', NetworkSpace(nx.path_graph(10)), value=np.full(10, 2.0))])
    events = Simulation(system, dt=1, random_state=0).run(until=1).trajectory_.events
    assert events[0].value == 2.0
    assert all(event.value <= 2.0 for event in events)
    with pytest.raises(ValueError, match='extremum'):
        Extremal(None, Adjacency(), _renew, extremum='median')


def test_extremal_runs_are_reproducible_across_backends():
    """Test that two runs on the NumPy and two on the Numba backend give identical lattices and events."""
    values = np.random.default_rng(1).random(50)
    runs = []
    for backend in ('numpy', 'numpy', 'numba', 'numba'):
        system = System(components=[Fitness.lattice('ring', LatticeSpace(50), value=values)])
        simulation = Simulation(system, dt=1, backend=backend, random_state=7).run(until=20)
        runs.append((simulation.trajectory_['ring.value'], simulation.trajectory_.events))
    system = System(components=[Fitness.lattice('ring', LatticeSpace(50), value=values)])
    compiled = compile_system(system, backend='numba')
    assert isinstance(compiled.lattice[0], CompiledExtremal)
    assert compiled.lattice[0].loop is not None
    for states, events in runs[1:]:
        assert np.array_equal(states, runs[0][0])
        assert events == runs[0][1]
    assert len(runs[0][1]) == 140


def test_triangular_forest_fire_streams_hexagonal_cells(tmp_path):
    """Test that a triangular forest fire logs one colored cell per site at its position at every logged step."""
    simulation = Simulation(hexagonal_forest_fire((12, 10)), dt=1).run(until=1)
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    recording = Recording()
    SiteVisual().log(recording, simulation.problem_.elements[0], frame)
    ((path, points, _),) = recording.entries
    assert path == 'forest/cover'
    assert type(points).__name__ == 'Points2D'
    positions = np.asarray(points.positions.as_arrow_array().to_numpy(zero_copy_only=False).tolist())
    assert np.allclose(positions, LatticeSpace((12, 10), geometry='triangular').positions)
    assert len(points.colors.as_arrow_array()) == 120
    observer = RerunObserver('forest', save_path=tmp_path / 'forest.rrd')
    Simulation(hexagonal_forest_fire((12, 10)), dt=1, observers=[observer]).run(until=3)
    assert observer.logged_frames_ == 4
    assert (tmp_path / 'forest.rrd').stat().st_size > 0


def test_graph_lattices_stream_nodes_and_static_edges():
    """Test that a lattice group on a graph logs its edges once and its nodes at every frame."""
    simulation = Simulation(graph_ising(nx.cycle_graph(8)), dt=1).run(until=1)
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    visual, recording = SiteVisual(low=-1, high=1), Recording()
    visual.log(recording, simulation.problem_.elements[0], frame)
    visual.log(recording, simulation.problem_.elements[0], frame)
    kinds = [(type(archetype).__name__, static) for _, archetype, static in recording.entries]
    assert kinds == [('GraphEdges', True), ('GraphNodes', False), ('GraphNodes', False)]
