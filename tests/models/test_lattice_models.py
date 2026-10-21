"""Test the lattice models of the library and their images in Rerun."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import histogram
from skcomplex.core.experiments import Experiment, Grid
from skcomplex.core.runtime import Frame, Observer, Simulation
from skcomplex.core.viz import ImageVisual, RerunObserver
from skcomplex.models.biology.evolution import bak_sneppen
from skcomplex.models.chemistry.kinetics import hodgepodge
from skcomplex.models.computer_science.automata import langtons_ant
from skcomplex.models.earth.seismology import earthquakes
from skcomplex.models.engineering.traffic import nagel_schreckenberg
from skcomplex.models.physics.criticality import hexagonal_forest_fire, sandpile
from skcomplex.models.physics.fluids import fhp_lattice_gas
from skcomplex.models.physics.statistical_mechanics import graph_ising, ising, potts, xy
from skcomplex.models.social.opinion_dynamics import majority_rule

CRITICAL_TEMPERATURE = 2 / np.log(1 + np.sqrt(2))


class Events(Observer):
    """Observer that keeps the events of every frame and checks a condition on the frames."""

    def __init__(self, check=None):
        """Start with no events."""
        self.events = []
        self.check = check
        self.passed = True

    def on_frame(self, frame):
        """Keep the frame's events and check the condition."""
        self.events += list(frame.events)
        if self.check is not None:
            self.passed &= bool(self.check(frame))


class Images:
    """Recording stand-in that keeps the images logged to it."""

    def __init__(self):
        """Start with no images."""
        self.images = []

    def log(self, path, archetype):
        """Keep a logged image with its path."""
        self.images.append((path, archetype))


def magnetization(trajectory):
    """Return the mean absolute magnetisation of the recorded frames."""
    return float(np.abs(trajectory['spins.spin'].mean(axis=1)).mean())


def test_sandpile_avalanches_follow_power_law():
    """Test that the 50 x 50 sandpile stays stable after every step and its sizes follow a power law over two decades.

    The exponent is the slope of the log-binned density of 100,000 avalanche sizes after a transient of 20,000 grains,
    between sizes 1 and 100.
    """
    observer = Events(lambda frame: (frame.lattice('pile.height') < 4).all())
    simulation = Simulation(sandpile((50, 50)), dt=1000, record=False, random_state=0, observers=[observer])
    simulation.run(until=120_000)
    sizes = np.asarray([event.size for event in observer.events if event.time > 20_000])
    assert observer.passed
    assert sizes.max() > 1000
    assert {event.label for event in observer.events} == {'avalanche'}
    edges, density = histogram(sizes, 20, log=True)
    centers = np.sqrt(edges[1:] * edges[:-1])
    decades = (centers >= 1) & (centers <= 100) & (density > 0)
    exponent = -np.polyfit(np.log(centers[decades]), np.log(density[decades]), 1)[0]
    assert 1.0 <= exponent <= 1.3


def test_ising_transition():
    """Test that the ensemble magnetisation of a 64 x 64 Ising lattice drops below one half between 2.2 and 2.4."""
    temperatures = [1.5, 2.0, 2.2, 2.4, 3.0, 3.5]
    result = Experiment(
        lambda temperature: ising(64, temperature=temperature, initial='up'),
        Grid(temperature=temperatures),
        replicates=4,
        until=1000,
        dt=50,
        warmup=200,
        outputs=[magnetization],
        random_state=0,
    ).run()
    means = {row['temperature']: row['magnetization.mean'] for row in result.summary()}
    assert means[2.2] > 0.5 > means[2.4]
    assert means[1.5] > 0.95
    assert means[3.5] < 0.1
    assert 2.2 < CRITICAL_TEMPERATURE < 2.4


def test_traffic_jams_lower_flow_and_move_backward():
    """Test that dense traffic flows less than near the critical density and its jams move backward."""
    flows, velocities = {}, {}
    for density in (0.15, 0.5):
        trajectory = Simulation(nagel_schreckenberg(1000, density=density), dt=1, random_state=0).run(600).trajectory_
        velocities[density] = trajectory['road.velocity'][100:]
        flows[density] = np.where(velocities[density] >= 0, velocities[density], 0).sum(axis=1).mean() / 1000
    assert flows[0.5] < flows[0.15]
    stopped = (velocities[0.5] == 0).astype(float)
    backward = np.mean(stopped[1:] * np.roll(stopped[:-1], -1, axis=1))
    forward = np.mean(stopped[1:] * np.roll(stopped[:-1], 1, axis=1))
    assert backward > 1.5 * forward


def test_traffic_conserves_cars():
    """Test that the number of cars on the ring never changes."""
    trajectory = Simulation(nagel_schreckenberg(200, density=0.3), dt=1, random_state=1).run(100).trajectory_
    assert np.all((trajectory['road.velocity'] >= 0).sum(axis=1) == 60)


def test_langtons_ant_moves_and_flips():
    """Test that Langton's ant keeps one ant, flips one cell per step, and builds its highway after 10,000 steps."""
    trajectory = Simulation(langtons_ant((100, 100)), dt=100).run(until=11_000).trajectory_
    ants = trajectory.lattice('cells.ant')
    assert np.all((ants >= 0).sum(axis=(1, 2)) == 1)
    black = trajectory.lattice('cells.color').sum(axis=(1, 2))
    assert black[1] <= 100
    position = np.argwhere(ants[-1] >= 0)[0]
    assert np.abs(position - 50).max() > 15


def test_majority_rule_settles():
    """Test that synchronous majority voting settles into domains that are fixed or blink with period two."""
    trajectory = Simulation(majority_rule((50, 50)), dt=1).run(until=50).trajectory_
    opinions = trajectory['voters.opinion']
    changes = np.abs(np.diff(opinions, axis=0)).sum(axis=1)
    assert changes[0] > 100
    assert changes[-1] < 25
    assert np.array_equal(opinions[-1], opinions[-3])


def test_hodgepodge_stays_in_range():
    """Test that the hodgepodge machine keeps every level between healthy and ill, with waves of infection."""
    levels = Simulation(hodgepodge((50, 50)), dt=1, random_state=0).run(until=100).trajectory_['cells.level']
    assert levels.min() >= 0
    assert levels.max() <= 100
    assert np.ptp(levels[-1]) > 50


def test_hodgepodge_keeps_waves():
    """Test that the default hodgepodge machine keeps travelling waves, with more than a tenth of the cells ill."""
    levels = Simulation(hodgepodge(), dt=1, random_state=0).run(until=300).trajectory_['cells.level']
    assert (levels[-1] == 100).mean() > 0.1


def test_earthquakes_have_broad_sizes():
    """Test that the earthquake model records earthquakes of sizes spanning two decades."""
    observer = Events()
    Simulation(earthquakes((30, 30)), dt=100, record=False, random_state=0, observers=[observer]).run(until=5000)
    sizes = np.asarray([event.size for event in observer.events])
    assert sizes.min() == 1
    assert sizes.max() > 100


@pytest.mark.parametrize('factory', [potts, xy])
def test_cold_spin_models_order(factory):
    """Test that the Potts and XY models order at low temperature."""
    simulation = Simulation(factory(16, temperature=0.2), dt=100, random_state=0).run(until=300)
    values = simulation.state_
    if factory is potts:
        assert np.bincount(values.astype(int)).max() > 0.8 * values.size
    else:
        assert np.abs(np.exp(1j * values).mean()) > 0.8


def test_lattices_log_images():
    """Test that a lattice state is logged as an image of the lattice shape."""
    recording = Images()
    simulation = Simulation(ising(8, initial='up'), dt=1).run(until=1)
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    ImageVisual(low=-1, high=1).log(recording, simulation.problem_.elements[0], frame)
    ((path, image),) = recording.images
    assert path == 'spins/spin'
    assert type(image).__name__ == 'Image'


def test_lattice_recording_is_saved(tmp_path):
    """Test that a lattice model streams to a recording with one image per frame."""
    observer = RerunObserver('ising', save_path=tmp_path / 'ising.rrd')
    Simulation(ising(16), dt=1, observers=[observer]).run(until=5)
    assert observer.logged_frames_ == 6
    assert (tmp_path / 'ising.rrd').stat().st_size > 0


def test_fhp_lattice_gas_conserves_mass_and_momentum():
    """Test that the FHP gas on a periodic 64 x 64 lattice keeps its particle number and momentum for 500 steps."""
    trajectory = Simulation(fhp_lattice_gas((64, 64)), dt=1, random_state=0).run(until=500).trajectory_
    occupations = np.stack([trajectory[f'gas.n{direction}'].astype(float).sum(axis=1) for direction in range(6)])
    angles = np.arange(6) * np.pi / 3
    momentum = np.stack([np.cos(angles) @ occupations, np.sin(angles) @ occupations])
    assert len(trajectory) == 501
    assert np.all(occupations.sum(axis=0) == occupations.sum(axis=0)[0])
    assert np.allclose(momentum, momentum[:, :1], rtol=0, atol=1e-9)
    assert np.array_equal(trajectory['gas.density'].sum(axis=1), occupations.sum(axis=0))


def test_graph_ising_orders_at_low_temperature():
    """Test that the Ising model on a random 4-regular graph of 1000 nodes orders at 1.0 and not at 6.0."""
    graph = nx.random_regular_graph(4, 1000, seed=0)
    result = Experiment(
        lambda temperature: graph_ising(graph, temperature=temperature, initial='up'),
        Grid(temperature=[1.0, 6.0]),
        replicates=4,
        until=400,
        dt=20,
        warmup=100,
        outputs=[magnetization],
        random_state=0,
    ).run()
    means = {row['temperature']: row['magnetization.mean'] for row in result.summary()}
    assert means[1.0] > 0.9
    assert means[6.0] < 0.1


def test_forest_fire_burns_graph_by_distance():
    """Test that a fire on a forested Watts-Strogatz graph reaches every node at the step of its graph distance."""
    graph = nx.connected_watts_strogatz_graph(500, 4, 0.1, seed=0)
    cover = np.ones(500)
    cover[0] = 2
    system = hexagonal_forest_fire(graph=graph, growth=0.0, lightning=0.0, cover=cover)
    covers = Simulation(system, dt=1).run(until=40).trajectory_['forest.cover']
    distances = nx.single_source_shortest_path_length(graph, 0)
    assert (covers == 2).any(axis=0).all()
    assert np.array_equal(np.argmax(covers == 2, axis=0), [distances[node] for node in graph.nodes])


@pytest.mark.benchmark
def test_bak_sneppen_self_organises():
    """Test that Bak-Sneppen on a ring of 200 gathers its fitnesses above 0.75 with largest minima near 0.667."""
    observer = Events()
    system = bak_sneppen(200, updates=10_000)
    simulation = Simulation(system, dt=100, record=False, backend='numba', random_state=0, observers=[observer])
    simulation.run(until=200)
    minima = np.asarray([event.value for event in observer.events if event.time > 100])
    assert minima.size == 1_000_000
    assert np.median(simulation.state_) > 0.75
    assert 0.6 < minima.max() < 0.72
