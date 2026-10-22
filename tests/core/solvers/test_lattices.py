"""Test lattices, neighbourhoods, local rules, schedules, relaxation, and Monte Carlo sampling."""

import numpy as np
import pytest

from skcomplex.core.base import Domain, SolverError, ValidationError, state
from skcomplex.core.compile import compile_system
from skcomplex.core.model import (
    Drive,
    Element,
    Energy,
    Group,
    LatticeSpace,
    LocalRule,
    Move,
    Neighborhood,
    OuterTotalistic,
    Relaxation,
    Sample,
    Sampler,
    System,
    Table,
    Totalistic,
    neighbors,
)
from skcomplex.core.runtime import Simulation
from skcomplex.models.computer_science.automata import elementary_automaton, game_of_life
from skcomplex.models.physics.criticality import forest_fire
from skcomplex.models.physics.statistical_mechanics import ising, xy

Shift = Element.define(
    'Shift',
    states={'value': 0.0},
    laws=lambda s: [LocalRule(s.value, Neighborhood(((-1,),)), lambda left: left)],
)


def _shift(schedule):
    """Return a ring of five cells whose rule copies the left neighbour on a schedule."""
    Copy = Element.define(
        'Copy',
        states={'value': 0.0},
        laws=lambda s: [LocalRule(s.value, Neighborhood(((-1,),)), lambda left: left, schedule=schedule)],
    )
    return System(components=[Copy.lattice('ring', LatticeSpace(6), value=[1, 2, 3, 4, 5, 6])])


def _rule_30(n_steps, size):
    """Return the space-time diagram of rule 30 computed with `left xor (centre or right)`."""
    row = np.zeros(size, dtype=int)
    row[size // 2] = 1
    rows = [row]
    for _ in range(n_steps):
        row = np.roll(row, 1) ^ (row | np.roll(row, -1))
        rows.append(row)
    return np.asarray(rows)


def test_moore_neighbourhood_wraps_on_torus():
    """Test that the Moore neighbours of a corner of a periodic 10 x 10 lattice wrap across both edges."""
    neighbours = LatticeSpace((10, 10)).neighbors((0, 0), Neighborhood.moore())
    assert sorted(neighbours) == [(0, 1), (0, 9), (1, 0), (1, 1), (1, 9), (9, 0), (9, 1), (9, 9)]


def test_fixed_and_reflecting_edges():
    """Test that fixed edges read the fill value and reflecting edges read the mirror image."""
    assert LatticeSpace((5,), boundary='fixed').neighbors((0,), Neighborhood.radius()) == [None, (0,), (1,)]
    assert LatticeSpace((5,), boundary='reflecting').neighbors((0,), Neighborhood.radius(2))[:2] == [(1,), (0,)]
    Edge = Element.define(
        'Edge',
        states={'value': 0.0},
        laws=lambda s: [LocalRule(s.value, Neighborhood(((-1,),)), lambda left: left)],
    )
    fixed = System(components=[Edge.lattice('cells', LatticeSpace(3, boundary='fixed', fill=9), value=[1, 2, 3])])
    assert Simulation(fixed, dt=1).run(until=1).state_.tolist() == [9, 1, 2]
    mirror = System(components=[Edge.lattice('cells', LatticeSpace(3, boundary='reflecting'), value=[1, 2, 3])])
    assert Simulation(mirror, dt=1).run(until=1).state_.tolist() == [1, 1, 2]


def test_neighborhoods_are_ordered():
    """Test the order and the sizes of the built-in neighbourhoods."""
    assert Neighborhood.moore(center=True).offsets[0] == (0, 0)
    assert len(Neighborhood.moore(2)) == 24
    assert len(Neighborhood.von_neumann(2)) == 12
    assert len(Neighborhood.moore(n_dim=3)) == 26
    with pytest.raises(ValueError, match='same number of dimensions'):
        Neighborhood(((0, 1), (1,)))


def test_rule_30_matches_published_pattern():
    """Test that rule 30 from a single live cell reproduces its space-time diagram for 15 steps."""
    trajectory = Simulation(elementary_automaton(30, size=41), dt=1).run(until=15).trajectory_
    assert np.array_equal(trajectory.lattice('cells.alive').astype(int), _rule_30(15, 41))


def test_glider_moves_diagonally():
    """Test that a glider on a periodic 20 x 20 lattice has the same shape one cell further after 4 steps."""
    glider = np.zeros((20, 20))
    glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
    trajectory = Simulation(game_of_life((20, 20), initial=glider), dt=1).run(until=4).trajectory_
    cells = trajectory.lattice('cells.alive')
    assert np.array_equal(cells[-1], np.roll(glider, (1, 1), axis=(0, 1)))
    assert not np.array_equal(cells[2], cells[0])


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_lookup_rules_agree_across_backends(backend):
    """Test that the fused and the NumPy lookup rules give the same Game of Life."""
    trajectory = Simulation(game_of_life((30, 30)), dt=1, backend=backend).run(until=20).trajectory_
    reference = Simulation(game_of_life((30, 30)), dt=1, backend='numpy').run(until=20).trajectory_
    assert np.array_equal(trajectory.states, reference.states)


def test_table_and_totalistic_rules():
    """Test that tables, totalistic, and outer totalistic rules give their documented next values."""
    Parity = Element.define(
        'Parity',
        states={'bit': 0.0},
        laws=lambda s: [LocalRule(s.bit, Neighborhood.radius(), Totalistic({1: 1, 3: 1}))],
    )
    ring = System(components=[Parity.lattice('ring', LatticeSpace(5), bit=[0, 1, 0, 0, 0])])
    assert Simulation(ring, dt=1).run(until=1).state_.tolist() == [1, 1, 1, 0, 0]
    Swap = Element.define(
        'Swap',
        states={'bit': 0.0},
        laws=lambda s: [LocalRule(s.bit, Neighborhood(((0,),)), Table({(0,): 1, (1,): 0}))],
    )
    flip = System(components=[Swap.lattice('ring', LatticeSpace(3), bit=[0, 1, 0])])
    assert Simulation(flip, dt=1).run(until=1).state_.tolist() == [1, 0, 1]
    Count = Element.define(
        'Count',
        states={'bit': 0.0},
        laws=lambda s: [
            LocalRule(s.bit, Neighborhood(((-1,), (1,))), OuterTotalistic(lambda center, total: (center + total) % 2)),
        ],
    )
    count = System(components=[Count.lattice('ring', LatticeSpace(4), bit=[1, 0, 0, 0])])
    assert Simulation(count, dt=1).run(until=1).state_.tolist() == [1, 1, 0, 1]


def test_synchronous_rule_shifts_whole_configuration():
    """Test that one synchronous step of a rule that copies the left neighbour shifts every cell by one."""
    assert Simulation(_shift('synchronous'), dt=1).run(until=1).state_.tolist() == [6, 1, 2, 3, 4, 5]


@pytest.mark.parametrize('schedule', ['random-sequential', 'random-site'])
def test_asynchronous_schedules_are_reproducible(schedule):
    """Test that asynchronous schedules copy values from neighbours and repeat with the same seed."""
    first = Simulation(_shift(schedule), dt=1, random_state=3).run(until=2).state_
    second = Simulation(_shift(schedule), dt=1, random_state=3).run(until=2).state_
    assert first.tolist() == second.tolist()
    assert set(first) <= {1, 2, 3, 4, 5, 6}
    assert first.tolist() != [5, 6, 1, 2, 3, 4]


def test_checkerboard_schedule_updates_sublattices_in_turn():
    """Test that a checkerboard update of the copy rule on a ring moves even sites first, then odd sites."""
    assert Simulation(_shift('checkerboard'), dt=1).run(until=1).state_.tolist() == [6, 6, 2, 2, 4, 4]


def test_checkerboard_needs_bipartite_neighbourhood():
    """Test that a checkerboard schedule on a Moore neighbourhood is rejected."""
    Cell = Element.define(
        'Cell',
        states={'alive': 0.0},
        laws=lambda s: [LocalRule(s.alive, Neighborhood.moore(), lambda *values: values[0], schedule='checkerboard')],
    )
    with pytest.raises(ValidationError, match='random-sequential'):
        compile_system(System(components=[Cell.lattice('cells', LatticeSpace((4, 4)))]))


def test_unknown_schedule_is_rejected():
    """Test that an unknown schedule names the known ones."""
    with pytest.raises(ValueError, match='synchronous'):
        LocalRule(None, Neighborhood.radius(), lambda *values: values[0], schedule='parallel')


def test_forest_fire_is_reproducible():
    """Test that two forest fires with the same seed give identical lattices at every step."""
    first = Simulation(forest_fire((30, 30)), dt=1, random_state=7).run(until=30).trajectory_.states
    second = Simulation(forest_fire((30, 30)), dt=1, random_state=7).run(until=30).trajectory_.states
    third = Simulation(forest_fire((30, 30)), dt=1, random_state=8).run(until=30).trajectory_.states
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_rule_reads_parameters_by_keyword():
    """Test that a function rule receives the parameters it names, which change live."""
    Growth = Element.define(
        'Growth',
        states={'value': 0.0},
        parameters={'rate': 1.0},
        laws=lambda s: [LocalRule(s.value, Neighborhood(((0,),)), lambda value, *, rate: value + rate)],
    )
    simulation = Simulation(System(components=[Growth.lattice('cells', LatticeSpace(2))]), dt=1).run(until=2)
    assert simulation.state_.tolist() == [2.0, 2.0]
    simulation.update_parameters({'cells.rate': 5.0})
    assert simulation.parameters_.tolist() == [5.0, 5.0]


def test_relaxation_records_avalanches():
    """Test that a relaxation records the size, duration, and area of a toppling avalanche."""
    Pile = Element.define(
        'Pile',
        states={'height': 0.0},
        laws=lambda s: [
            Relaxation(
                drive=[Drive(s.height, lambda height, *, random: height + (np.arange(height.size) == 2))],
                relax=[
                    LocalRule(
                        s.height,
                        Neighborhood.radius(),
                        lambda left, height, right: height - 2 * (height >= 2) + (left >= 2) + (right >= 2),
                    ),
                ],
                stable=lambda height: bool((height < 2).all()),
                activity=lambda height: height >= 2,
            ),
        ],
    )
    space = LatticeSpace(5, boundary='fixed')
    trajectory = Simulation(System(components=[Pile.lattice('pile', space, height=[1, 1, 1, 1, 1])]), dt=1).run(1)
    (record,) = trajectory.trajectory_.events
    assert (record.label, record.time) == ('pile', 1.0)
    assert (record.size, record.duration, record.area) == (9, 5, 5)
    assert np.all(trajectory.state_ < 2)


def test_relaxation_without_end_raises():
    """Test that a relaxation that never becomes stable raises an error after its largest number of iterations."""
    Restless = Element.define(
        'Restless',
        states={'value': 0.0},
        laws=lambda s: [
            Relaxation([], [LocalRule(s.value, Neighborhood(((0,),)), lambda value: value + 1)], lambda _: False, 10),
        ],
    )
    with pytest.raises(SolverError, match='not stable after 10 iterations'):
        Simulation(System(components=[Restless.lattice('cells', LatticeSpace(2))]), dt=1).run(until=1)


def test_local_rule_needs_lattice_group():
    """Test that a local rule on a state outside a lattice group is rejected."""
    with pytest.raises(ValidationError, match='LocalRule law of cells needs a lattice'):
        compile_system(System(components=[Group('cells', element=Shift, size=3)]))


def test_monte_carlo_energy_change_is_symbolic():
    """Test that a flip move of an Ising energy changes the energy by `2 J s sum(neighbours)`."""
    nearest = Neighborhood.von_neumann()
    Spin = Element.define(
        'Spin',
        states={'spin': state(1.0, domain=Domain.INTEGER)},
        parameters={'J': 1.0, 'temperature': 1.0},
        laws=lambda s: [Sample(Energy(-s.J * s.spin * sum(neighbors(s.spin, nearest)), nearest), Move.flip(s.spin))],
    )
    problem = compile_system(System(components=[Spin.lattice('spins', LatticeSpace((4, 4)))]))
    sampling = problem.lattice[0]
    change = sampling.delta(0.0, np.ones(3), *np.ones((4, 3)), -np.ones(3), 1.0, 1.0)[0]
    assert np.allclose(change, 8.0)
    assert isinstance(sampling.sampler, Sampler)
    assert sampling.sampler.kind == 'metropolis'


def test_monte_carlo_needs_one_energy_and_move():
    """Test that two Monte Carlo laws of one component are rejected."""
    nearest = Neighborhood.von_neumann()
    Lonely = Element.define(
        'Lonely',
        states={'spin': 1.0},
        parameters={'temperature': 1.0},
        laws=lambda s: [
            Sample(Energy(-s.spin * sum(neighbors(s.spin, nearest)), nearest), Move.flip(s.spin)),
            Sample(Energy(-s.spin * sum(neighbors(s.spin, nearest)), nearest), Move.flip(s.spin)),
        ],
    )
    with pytest.raises(ValidationError, match='one energy, one move'):
        compile_system(System(components=[Lonely.lattice('spins', LatticeSpace((4, 4)))]))


@pytest.mark.parametrize('sampler', ['metropolis', 'glauber', 'heat-bath'])
def test_samplers_order_cold_ising(sampler):
    """Test that every sampler keeps a cold Ising lattice ordered and disorders a hot one."""
    cold = Simulation(ising(16, temperature=0.5, initial='up', sampler=sampler), dt=50, random_state=0).run(200)
    hot = Simulation(ising(16, temperature=10.0, initial='up', sampler=sampler), dt=50, random_state=0).run(200)
    assert abs(cold.state_.mean()) > 0.95
    assert abs(hot.state_.mean()) < 0.3


def test_temperature_changes_live():
    """Test that raising the temperature of an ordered lattice during a run disorders it."""
    simulation = Simulation(
        ising(16, temperature=0.5, initial='up'),
        dt=50,
        random_state=0,
        schedule={'spins.temperature': lambda time: 0.5 if time < 100 else 20.0},
    ).run(until=200)
    magnetization = np.abs(simulation.trajectory_['spins.spin'].mean(axis=1))
    assert magnetization[2] > 0.95
    assert magnetization[-1] < 0.3


def test_monte_carlo_wraps_angles_after_every_sweep():
    """Test that Monte Carlo rotations of the XY model keep every angle in (-π, π]."""
    values = Simulation(xy(16)).run(until=50).trajectory_['spins.theta']
    assert np.all((values > -np.pi) & (values <= np.pi))


def _pile(activity, evaluated):
    """Return a 32 x 32 sandpile whose relaxation declares an activity or none, counting the sites it topples."""

    def topple(height, *around):
        evaluated.append(np.size(height))
        return height - 4 * (height >= 4) + sum(item >= 4 for item in around)

    def drop(heights, *, random):
        following = heights.copy()
        following[tuple(random.integers(size) for size in heights.shape)] += 1
        return following

    Pile = Element.define(
        'Pile',
        states={'height': state(0.0)},
        laws=lambda s: [
            Relaxation(
                drive=[Drive(s.height, drop)],
                relax=[LocalRule(s.height, Neighborhood.von_neumann(center=True), topple)],
                stable=lambda heights: bool((heights < 4).all()),
                activity=activity,
                label='avalanche',
            ),
        ],
    )
    heights = np.random.default_rng(0).integers(0, 4, size=(32, 32)).astype(float)
    space = LatticeSpace((32, 32), boundary='fixed')
    return System(components=[Pile.lattice('pile', space, height=heights)])


def test_relaxation_over_active_sites_matches_the_full_update():
    """Test that relaxing only active sites and their neighbours gives the heights and durations of full updates."""
    sparse_sites, full_sites = [], []
    sparse = Simulation(_pile(lambda heights: heights >= 4, sparse_sites), dt=5000, random_state=0)
    sparse = sparse.run(until=5000).trajectory_
    full = Simulation(_pile(None, full_sites), dt=5000, random_state=0).run(until=5000).trajectory_
    np.testing.assert_array_equal(sparse['pile.height'][-1], full['pile.height'][-1])
    assert [event.duration for event in sparse.events] == [event.duration for event in full.events]
    assert [event.time for event in sparse.events] == [event.time for event in full.events]
    assert sum(sparse_sites) < sum(full_sites) / 10
