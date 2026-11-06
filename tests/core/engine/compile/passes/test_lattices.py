"""Test triangular lattices, lattice laws on graphs, neighbour sums, colored schedules, and extremal updates."""

import itertools
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import cos, var
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.laws import ExtremalLaw, LocalRuleLaw, SampleLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import (
    AdjacencyNeighborhood,
    BoxNeighborhood,
    LatticeNeighborhood,
    LatticeSpace,
    MooreNeighborhood,
    NetworkSpace,
    TriangularNeighborhood,
)
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.problems._extremal_kernel import ExtremalKernel
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver, SiteVisual
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.core.specs import (
    ExtremalRecordSpec,
    LinearRuleSpec,
    MoveSpec,
    ObjectiveSpec,
    OuterTotalisticSpec,
    ParameterSpec,
    SamplerSpec,
    StateSpec,
    TableSpec,
    TotalisticSpec,
    VisualSpec,
)
from skcomplex.systems.physics.criticality import hexagonal_forest_fire
from skcomplex.systems.physics.statistical_mechanics import graph_ising


class BurnElement(Element):
    """Burn element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        fire: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.fire = fire

    def _get_laws(self, s):
        """Return the laws of the burn element."""
        return [LocalRuleLaw(s.fire, TriangularNeighborhood(center=True), lambda *values: np.max(values, axis=0))]


class LifeElement(Element):
    """Cell of the Game of Life on a square lattice."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alive: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.alive = alive

    def _get_laws(self, s):
        """Return the Game of Life rule on the Moore neighbourhood."""
        return [LocalRuleLaw(s.alive, MooreNeighborhood(), OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1}))]


class GraphLifeElement(LifeElement):
    """Cell of the Game of Life on a graph."""

    def _get_laws(self, s):
        """Return the Game of Life rule on the adjacent nodes."""
        return [LocalRuleLaw(s.alive, AdjacencyNeighborhood(), OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1}))]


def _renew(value, random):
    """Return new uniform values."""
    return random


class FitnessElement(Element):
    """Fitness element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value

    def _get_laws(self, s):
        """Return the laws of the fitness element."""
        return [ExtremalLaw(s.value, LatticeNeighborhood(((-1,), (1,))), _renew, repeat=7, label='update')]


def _ising_magnetization(trajectory):
    """Return the mean absolute magnetisation of the recorded frames."""
    return float(np.abs(trajectory.values('spins.spin').mean(axis=1)).mean())


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
        tuple(np.unravel_index(site, space.shape_)) for site in space._get_neighbors(0, TriangularNeighborhood())
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
    table = space._index_table(TriangularNeighborhood())
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
    system = System(
        elements=[
            LatticeElement(
                'forest',
                element=BurnElement,
                space=LatticeSpace((40, 40), geometry='triangular'),
                values={'fire': fire},
            ),
        ],
    )
    burning = Simulation(system, dt=1, backend=backend()).prepare().run(until=10).trajectory_.lattice('forest.fire')
    assert [int(item.sum()) for item in burning] == [3 * k * (k + 1) + 1 for k in range(11)]


def test_network_index_table_pads_with_size():
    """Test that the adjacency table of a star graph pads the leaves with the number of nodes."""
    space = NetworkSpace(nx.star_graph(3)).prepare()
    table = space._index_table(AdjacencyNeighborhood(center=True))
    assert table.tolist() == [[0, 1, 2, 3], [1, 0, 4, 4], [2, 0, 4, 4], [3, 0, 4, 4]]
    with pytest.raises(ValidationError, match='adjacency neighbourhood'):
        space._get_neighbors(0, MooreNeighborhood())


def test_grid_graph_matches_square_lattice():
    """Test that a glider gives the same configurations on a periodic lattice and on its Moore graph for 40 steps."""
    lattice = LatticeSpace((20, 20)).prepare()
    table = lattice._index_table(MooreNeighborhood())
    graph = nx.Graph()
    graph.add_nodes_from(range(lattice.size_))
    graph.add_edges_from((site, int(other)) for site, row in enumerate(table) for other in row)
    glider = np.zeros((20, 20))
    glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
    on_lattice = System(
        elements=[
            LatticeElement('cells', element=LifeElement, space=lattice, values={'alive': glider}),
        ],
    )
    on_graph = System(
        elements=[
            LatticeElement(
                'cells',
                element=GraphLifeElement,
                space=NetworkSpace(graph),
                values={'alive': glider.ravel()},
            ),
        ],
    )
    expected = Simulation(on_lattice, dt=1).prepare().run(until=40).trajectory_.values('cells.alive')
    found = Simulation(on_graph, dt=1).prepare().run(until=40).trajectory_.values('cells.alive')
    assert np.array_equal(found, expected)
    assert expected[-1].sum() == 5


def test_totalistic_rule_counts_real_neighbours_on_star():
    """Test that a totalistic count on a star with 5 leaves gives every leaf 1 and the centre its live leaves."""

    class CountElement(Element):
        """Count element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            value: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.value = value

        def _get_laws(self, s):
            """Return the laws of the count element."""
            return [LocalRuleLaw(s.value, AdjacencyNeighborhood(), TotalisticSpec(lambda total: total))]

    values = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
    system = System(
        elements=[
            LatticeElement(
                'star',
                element=CountElement,
                space=NetworkSpace(nx.star_graph(5)),
                values={'value': values},
            ),
        ],
    )
    assert Simulation(system, dt=1).prepare().run(until=1).state_.tolist() == [2, 1, 1, 1, 1, 1]


def test_function_rule_reads_degrees_on_star():
    """Test that a function rule naming `degree` receives the number of neighbours of every node."""

    class DegreeElement(Element):
        """Degree element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            value: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.value = value

        def _get_laws(self, s):
            """Return the laws of the degree element."""
            return [LocalRuleLaw(s.value, AdjacencyNeighborhood(center=True), lambda *values, degree: degree)]

    system = System(elements=[LatticeElement('star', element=DegreeElement, space=NetworkSpace(nx.star_graph(5)))])
    assert Simulation(system, dt=1).prepare().run(until=1).state_.tolist() == [5, 1, 1, 1, 1, 1]


def test_lattice_groups_on_networks_reject_tables_and_capacities():
    """Test that a table of ordered patterns and a capacity are rejected on a network."""

    class OrderedElement(Element):
        """Ordered element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            value: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.value = value

        def _get_laws(self, s):
            """Return the laws of the ordered element."""
            return [LocalRuleLaw(s.value, AdjacencyNeighborhood(), TableSpec({(1, 1): 1}))]

    system = System(
        elements=[LatticeElement('nodes', element=OrderedElement, space=NetworkSpace(nx.cycle_graph(4)))],
    )
    with pytest.raises(ValidationError, match='totalistic'):
        PassCompiler().prepare(system).run()
    system = System(
        elements=[
            LatticeElement('nodes', element=OrderedElement, space=NetworkSpace(nx.cycle_graph(4), capacity=1)),
        ],
    )
    with pytest.raises(ValidationError, match='capacity'):
        PassCompiler().prepare(system).run()


def test_neighbor_sum_ignores_padding():
    """Test that the XY neighbour-sum energy change on a star with 4 leaves sums over the real neighbours only."""

    class XYElement(Element):
        """X y element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            theta: Annotated[object, StateSpec()] = 0.0,
            J: Annotated[object, ParameterSpec()] = 1.3,
            temperature: Annotated[object, ParameterSpec()] = 0.5,
        ) -> None:
            super().__init__(label, visual=visual)
            self.theta = theta
            self.J = J
            self.temperature = temperature

        def _get_laws(self, s):
            """Return the laws of the x y element."""
            return [
                SampleLaw(
                    ObjectiveSpec(
                        -s.J * neighbors(cos(s.theta - s.other.theta), reduce='sum'),
                        AdjacencyNeighborhood(),
                    ),
                    MoveSpec(s.theta, 'rotate', width=1.0),
                    SamplerSpec('metropolis', 1 / s.temperature, schedule='colored'),
                ),
            ]

    graph = nx.star_graph(4)
    angles = np.random.default_rng(0).uniform(0, 2 * np.pi, 5)
    system = System(
        elements=[LatticeElement('xy', element=XYElement, space=NetworkSpace(graph), values={'theta': angles})],
    )
    (sampling,) = PassCompiler().prepare(system).run().problem_.lattice
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
    result = (
        Experiment(
            lambda temperature: graph_ising(graph, temperature=temperature, initial='up'),
            GridDesign(),
            {'temperature': temperatures},
            replicates=4,
            until=1000,
            dt=50,
            warmup=200,
            outputs=[_ising_magnetization],
            random_state=0,
        )
        .prepare()
        .run()
        .result_
    )
    means = {row['temperature']: row['_ising_magnetization.mean'] for row in result.summary()}
    assert means[2.2] > 0.5 > means[2.4]
    assert means[1.5] > 0.95
    assert means[3.5] < 0.1


def test_colored_schedule_partitions_triangular_lattice():
    """Test that the colored classes of a 12 x 12 triangular lattice hold no neighbours and visit every site once."""

    class SpinElement(Element):
        """Spin element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            spin: Annotated[object, StateSpec()] = 1.0,
            temperature: Annotated[object, ParameterSpec()] = 2.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.spin = spin
            self.temperature = temperature

        def _get_laws(self, s):
            """Return the laws of the spin element."""
            return [
                SampleLaw(
                    ObjectiveSpec(
                        -s.spin * sum(neighbors(s.spin, TriangularNeighborhood(), reduce='each')),
                        TriangularNeighborhood(),
                    ),
                    MoveSpec(s.spin),
                    SamplerSpec('metropolis', 1 / s.temperature, schedule='colored'),
                ),
            ]

    space = LatticeSpace((12, 12), geometry='triangular')
    system = System(elements=[LatticeElement('spins', element=SpinElement, space=space)])
    (sampling,) = PassCompiler().prepare(system).run().problem_.lattice
    sites = np.concatenate(sampling.classes)
    assert np.array_equal(np.sort(sites), np.arange(144))
    for members in sampling.classes:
        assert not np.isin(sampling.table[members], members).any()
    Simulation(
        System(elements=[LatticeElement('spins', element=SpinElement, space=space)]),
        dt=1,
    ).prepare().run(until=2)


@pytest.mark.parametrize('space', [LatticeSpace((6, 6), geometry='triangular'), NetworkSpace(nx.cycle_graph(6))])
def test_checkerboard_rejected_where_it_does_not_apply(space):
    """Test that the checkerboard schedule on a triangular lattice or a graph names the law and the colored schedule."""
    neighborhood = TriangularNeighborhood() if isinstance(space, LatticeSpace) else AdjacencyNeighborhood()

    class FlipElement(Element):
        """Flip element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            spin: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.spin = spin

        def _get_laws(self, s):
            """Return the laws of the flip element."""
            return [LocalRuleLaw(s.spin, neighborhood, lambda *values: -values[0], schedule='checkerboard')]

    with pytest.raises(ValidationError, match=r"LocalRuleLaw of spins.*'colored'"):
        PassCompiler().prepare(System(elements=[LatticeElement('spins', element=FlipElement, space=space)])).run()


def test_extremal_changes_only_the_neighbourhood():
    """Test that one extremal update on a ring of 100 changes the minimum and its two neighbours only."""

    class OnceElement(Element):
        """Once element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            value: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.value = value

        def _get_laws(self, s):
            """Return the laws of the once element."""
            return [ExtremalLaw(s.value, BoxNeighborhood(), _renew)]

    values = np.random.default_rng(0).random(100)
    simulation = Simulation(
        System(
            elements=[
                LatticeElement('ring', element=OnceElement, space=LatticeSpace(100), values={'value': values}),
            ],
        ),
        dt=1,
    ).prepare()
    simulation.run(until=1)
    site = int(np.argmin(values))
    changed = np.flatnonzero(simulation.state_ != values)
    assert sorted(changed.tolist()) == sorted([(site - 1) % 100, site, (site + 1) % 100])
    (event,) = simulation.trajectory_.events
    assert event == ExtremalRecordSpec('ring', 1.0, site, float(values[site]))


def test_extremal_maximum_and_ties():
    """Test that an extremal law on the largest value breaks ties among equal values with the random stream."""

    class TopElement(Element):
        """Top element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            value: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.value = value

        def _get_laws(self, s):
            """Return the laws of the top element."""
            return [ExtremalLaw(s.value, AdjacencyNeighborhood(), _renew, extremum='max', repeat=20)]

    system = System(
        elements=[
            LatticeElement(
                'nodes',
                element=TopElement,
                space=NetworkSpace(nx.path_graph(10)),
                values={'value': np.full(10, 2.0)},
            ),
        ],
    )
    events = Simulation(system, dt=1, random_state=0).prepare().run(until=1).trajectory_.events
    assert events[0].value == 2.0
    assert all(event.value <= 2.0 for event in events)
    with pytest.raises(ValidationError, match='extremum'):
        ExtremalLaw(None, AdjacencyNeighborhood(), _renew, extremum='median').prepare()


def test_extremal_runs_are_reproducible_across_backends():
    """Test that two runs on the NumPy and two on the Numba backend give identical lattices and events."""
    values = np.random.default_rng(1).random(50)
    runs = []
    for backend in (NumpyBackend, NumpyBackend, NumbaBackend, NumbaBackend):
        system = System(
            elements=[
                LatticeElement('ring', element=FitnessElement, space=LatticeSpace(50), values={'value': values}),
            ],
        )
        simulation = Simulation(system, dt=1, backend=backend(), random_state=7).prepare().run(until=20)
        runs.append((simulation.trajectory_.values('ring.value'), simulation.trajectory_.events))
    system = System(
        elements=[LatticeElement('ring', element=FitnessElement, space=LatticeSpace(50), values={'value': values})],
    )
    compiled = PassCompiler(backend=NumbaBackend()).prepare(system).run().problem_
    assert isinstance(compiled.lattice[0], ExtremalKernel)
    assert compiled.lattice[0].loop is not None
    for states, events in runs[1:]:
        assert np.array_equal(states, runs[0][0])
        assert events == runs[0][1]
    assert len(runs[0][1]) == 140


def test_triangular_forest_fire_streams_hexagonal_cells(tmp_path):
    """Test that a triangular forest fire logs one colored cell per site at its position at every logged step."""
    simulation = Simulation(hexagonal_forest_fire((12, 10)), dt=1).prepare().run(until=1)
    frame = FrameSpec(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    recording = Recording()
    (SiteVisual(path=simulation.problem_.elements[0].path, recording=recording).prepare(simulation.problem_).run(frame))
    ((path, points, _),) = recording.entries
    assert path == 'forest/cover'
    assert type(points).__name__ == 'Points2D'
    positions = np.asarray(points.positions.as_arrow_array().to_numpy(zero_copy_only=False).tolist())
    assert np.allclose(positions, LatticeSpace((12, 10), geometry='triangular').prepare().positions_)
    assert len(points.colors.as_arrow_array()) == 120
    observer = RerunObserver('forest', save_path=tmp_path / 'forest.rrd')
    Simulation(hexagonal_forest_fire((12, 10)), dt=1, observers=[observer]).prepare().run(until=3)
    assert observer.logged_frames_ == 4
    assert (tmp_path / 'forest.rrd').stat().st_size > 0


def test_graph_lattices_stream_nodes_and_static_edges():
    """Test that a lattice group on a graph logs its edges once and its nodes at every frame."""
    simulation = Simulation(graph_ising(nx.cycle_graph(8)), dt=1).prepare().run(until=1)
    frame = FrameSpec(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    recording = Recording()
    visual = SiteVisual(low=-1, high=1, path=simulation.problem_.elements[0].path, recording=recording)
    visual.prepare(simulation.problem_)
    visual.run(frame)
    visual.run(frame)
    kinds = [(type(archetype).__name__, static) for _, archetype, static in recording.entries]
    assert kinds == [('GraphEdges', True), ('GraphNodes', False), ('GraphNodes', False)]


class OnOffElement(Element):
    """Cell with the values 0 and 1."""

    def __init__(self, label, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive


def test_linear_rule_runs_as_its_law_steps():
    """Test that a linear rule compiles to a function rule, and that a run agrees with the law of Life and rule 90."""
    start = np.array([0, 0, 1, 0, 0, 0])
    cells = LatticeElement('cells', element=OnOffElement, space=LatticeSpace(6), values={'alive': start})
    rule90 = LocalRuleInteraction(
        'rule',
        target='cells',
        state='alive',
        rule=LinearRuleSpec((1, 1), 'GF(2)'),
        neighborhood=LatticeNeighborhood(((-1,), (1,))),
    )
    trajectory = Simulation(System(elements=[cells], interactions=[rule90]), dt=1).prepare().run(until=2).trajectory_
    values = trajectory.values('cells.alive')
    assert values[1].astype(int).tolist() == [0, 1, 0, 1, 0, 0]
    assert values[2].astype(int).tolist() == [1, 0, 0, 0, 1, 0]


def test_life_simulation_agrees_with_the_run_of_its_law():
    """Test that a simulation of Life and the run of its law give the same next configuration."""
    start = np.zeros(25)
    start[[1, 7, 10, 11, 12]] = 1
    rule = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})
    space = LatticeSpace((5, 5))
    cells = LatticeElement('cells', element=OnOffElement, space=space, values={'alive': start})
    life = LocalRuleInteraction('life', target='cells', state='alive', rule=rule, neighborhood=MooreNeighborhood())
    trajectory = Simulation(System(elements=[cells], interactions=[life]), dt=1).prepare().run(until=1).trajectory_
    law = LocalRuleLaw(var('alive'), MooreNeighborhood(), rule).prepare()
    image = law.run({'space': space, 'alive': start.astype(int)}).value_
    np.testing.assert_array_equal(trajectory.values('cells.alive')[-1].astype(int), image)
