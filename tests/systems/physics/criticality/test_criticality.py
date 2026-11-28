"""Test the `_criticality` module of `skcomplex.systems.physics.criticality`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import distribution, fit_law
from skcomplex.core.errors import ValidationError
from skcomplex.systems.physics.criticality import (
    ForestFireSystem,
    HexagonalForestFireSystem,
    MannaSandpileSystem,
    SandpileSystem,
)

from ...conftest import EventsObserver


@pytest.mark.slow
def test_sandpile_avalanches_follow_power_law():
    """Test that the 40 x 40 sandpile stays stable after every step and its sizes follow a power law over two decades.

    The exponent is the slope of the log-binned density of the avalanche sizes of 22,000 grains after a transient of
    8,000 grains, more than twice the grains that fill the pile to its critical density, between sizes 1 and 100.
    """
    observer = EventsObserver(lambda frame: (frame.get_lattice('pile.height') < 4).all())
    SandpileSystem((40, 40)).prepare().simulate(until=30_000, dt=1000, random_state=0, observers=[observer])
    sizes = np.asarray([event.size for event in observer.events_ if event.time > 8_000])
    assert observer.passed_
    assert sizes.max() > 1000
    assert {event.label for event in observer.events_} == {'avalanche'}
    histogram = distribution(sizes, kind='histogram', bins=20, log=True)
    edges, density = histogram.edges, histogram.density
    centers = np.sqrt(edges[1:] * edges[:-1])
    decades = (centers >= 1) & (centers <= 100) & (density > 0)
    exponent = -np.polyfit(np.log(centers[decades]), np.log(density[decades]), 1)[0]
    assert 1.0 <= exponent <= 1.3


def test_forest_fire_burns_graph_by_distance():
    """Test that a fire on a forested Watts-Strogatz graph reaches every node at the step of its graph distance."""
    graph = nx.connected_watts_strogatz_graph(500, 4, 0.1, seed=0)
    cover = np.ones(500)
    cover[0] = 2
    system = HexagonalForestFireSystem(graph=graph, growth=0.0, lightning=0.0, cover=cover)
    covers = system.prepare().simulate(until=40, dt=1).get_values('forest.cover')
    distances = nx.single_source_shortest_path_length(graph, 0)
    assert (covers == 2).any(axis=0).all()
    assert np.array_equal(np.argmax(covers == 2, axis=0), [distances[node] for node in graph.nodes])


def test_manna_sandpile_settles_below_two_grains():
    """Test that the Manna pile records avalanches, stays stable between grains, and fills to a critical density."""
    trajectory = MannaSandpileSystem((16, 16)).prepare().simulate(until=3000, dt=100, random_state=0)
    heights = trajectory.get_lattice('pile.height')
    assert heights.max() < 2
    assert trajectory.events
    assert 0.55 < heights[-1].mean() < 0.75


def test_forest_fire_records_one_fire_from_a_corner():
    """Test that a fire from a corner of a full fixed forest is one event that burns every tree in diamonds."""
    cover = np.ones(100)
    cover[0] = 2
    system = ForestFireSystem((10, 10), boundary='fixed', growth=0.0, lightning=0.0).prepare()
    (fire,) = system.simulate(until=40, dt=1, initial_state={'forest.cover': cover}).events
    assert (fire.label, fire.time, fire.size, fire.duration, fire.area) == ('fire', 0.0, 100, 19, 100)
    assert fire.profile == (*range(1, 11), *range(9, 0, -1))


def test_forest_fire_boundary_of_a_strip():
    """Test that a fire at column 0 reaches column 19 at step 1 on a periodic strip and at step 19 on a fixed one."""
    rows = np.arange(5) * 20
    for boundary, step in (('periodic', 1), ('fixed', 19)):
        cover = np.ones(100)
        cover[rows] = 2
        system = ForestFireSystem((5, 20), boundary=boundary, growth=0.0, lightning=0.0).prepare()
        covers = system.simulate(until=25, dt=1, initial_state={'forest.cover': cover}).get_values('forest.cover')
        assert np.argmax(covers[:, rows + 19] == 2, axis=0).tolist() == [step] * 5
    with pytest.raises(ValidationError, match='boundary'):
        ForestFireSystem(boundary='open').prepare()


def _space_time_fires(burning: np.ndarray, side: int) -> list[int]:
    """Return the sizes of the clusters of burning sites in space and time that end before the last frame."""
    lattice = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side, periodic=True), ordering='sorted')
    graph = nx.Graph()
    for step, sites in enumerate(burning):
        graph.add_nodes_from((step, int(site)) for site in np.flatnonzero(sites))
    for step, site in list(graph.nodes):
        for other in (site, *lattice[site]):
            graph.add_edges_from(((step, site), (later, other)) for later in (step, step + 1) if (later, other) in graph)
    last = len(burning) - 1
    return [len(fire) for fire in nx.connected_components(graph) if max(step for step, _ in fire) < last]


def test_forest_fire_events_of_a_driven_forest():
    """Test that the fires of a driven forest are the clusters of burning sites in space and time that end in the run."""
    system = ForestFireSystem((32, 32), growth=0.01, lightning=1e-3).prepare()
    trajectory = system.simulate(until=500, dt=1, random_state=0)
    burning = trajectory.get_values('forest.cover') == 2
    assert trajectory.events
    assert sorted(event.size for event in trajectory.events) == sorted(_space_time_fires(burning, 32))


@pytest.mark.slow
def test_sandpile_with_random_neighbors_has_mean_field_avalanches():
    """Test that a sandpile with random neighbours stays stable and has the mean-field size exponent 3/2."""
    system = SandpileSystem((32, 32), neighbors='random', conservation=0.999).prepare()
    trajectory = system.simulate(until=110_000, dt=1000, random_state=0)
    sizes = np.asarray([event.size for event in trajectory.events if event.time > 10_000])
    fitted = fit_law(sizes, law='power', minimum=1, maximum=300, discrete=True, bootstrap=0)
    assert trajectory.get_lattice('pile.height').max() < 4
    assert fitted.exponent == pytest.approx(1.5, abs=0.15)


def test_sandpile_with_random_neighbors_needs_a_loss():
    """Test that a sandpile with random neighbours and no loss of grains is refused, since it never stabilises."""
    with pytest.raises(ValidationError, match='never stabilises'):
        SandpileSystem(neighbors='random', conservation=1.0).prepare()
    with pytest.raises(ValidationError, match='neighbours'):
        SandpileSystem(neighbors='ring').prepare()
