"""Test the `LatticeSolver` class."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import (
    ObjectiveLaw,
    OuterTotalisticSpec,
    SamplerSpec,
    TableSpec,
    TotalisticSpec,
    UpdateLaw,
)
from skcomplex.core.components.laws.utils import neighbors, relax
from skcomplex.core.components.spaces import (
    LatticeNeighborhood,
    LatticeSpace,
    RadiusNeighborhood,
)
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.errors import SolverError, ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem, GameOfLifeSystem
from skcomplex.systems.physics.criticality import ForestFireSystem
from skcomplex.systems.physics.statistical_mechanics import IsingSystem, XYSystem


def _cells(space, laws, label='cells', **values):
    """Return cells at the sites of a lattice whose laws are a function of the namespace of the role `target`."""
    rule = Interaction('rule', roles={'target': label}, laws=laws)
    return System(elements=[Element(label, values=values, space=space)], interactions=[rule])


def _symbolic(s):
    """Return the target of a namespace with its states as symbols, which the rules inside a relaxation read."""
    return Interaction._get_symbolic(s.target, s.t)


def _copy(s, schedule=None):
    """Return the rule that copies the value of the left neighbour."""
    return [UpdateLaw(s.target.value, lambda left: left, neighborhood=LatticeNeighborhood(((-1,),)), schedule=schedule)]


def _shift(schedule):
    """Return a ring of six cells whose rule copies the left neighbour on a schedule."""
    return _cells(LatticeSpace(6), lambda s: _copy(s, schedule), label='ring', value=[1, 2, 3, 4, 5, 6])


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
    neighbours = LatticeSpace((10, 10)).prepare()._get_neighbors(0, RadiusNeighborhood(1, metric='chebyshev').prepare())
    assert sorted(neighbours.tolist()) == [1, 9, 10, 11, 19, 90, 91, 99]


def test_fixed_and_reflecting_edges():
    """Test that fixed edges read the fill value and reflecting edges read the mirror image."""
    fixed_space = LatticeSpace((5,), boundary='fixed').prepare()
    assert fixed_space._get_neighbors(0, LatticeNeighborhood(((-1,), (0,), (1,))).prepare()).tolist() == [0, 1]
    reflecting_space = LatticeSpace((5,), boundary='reflecting').prepare()
    assert reflecting_space._get_neighbors(0, LatticeNeighborhood(((-2,), (-1,), (0,), (1,), (2,))).prepare())[
        :2
    ].tolist() == [
        1,
        0,
    ]

    fixed = _cells(LatticeSpace(3, boundary='fixed', fill=9), _copy, value=[1, 2, 3])
    assert fixed.prepare().simulate(until=1, dt=1).states[-1].tolist() == [9, 1, 2]
    mirror = _cells(LatticeSpace(3, boundary='reflecting'), _copy, value=[1, 2, 3])
    assert mirror.prepare().simulate(until=1, dt=1).states[-1].tolist() == [1, 1, 2]


def test_neighborhoods_are_ordered():
    """Test the order and the sizes of the built-in neighbourhoods."""
    assert RadiusNeighborhood(1, metric='chebyshev', center=True)._get_offsets(2)[0] == (0, 0)
    assert len(RadiusNeighborhood(2, metric='chebyshev')._get_offsets(2)) == 24
    assert len(RadiusNeighborhood(2, metric='manhattan')._get_offsets(2)) == 12
    assert len(RadiusNeighborhood(1, metric='chebyshev')._get_offsets(3)) == 26
    with pytest.raises(ValidationError, match='same number of dimensions'):
        LatticeNeighborhood(((0, 1), (1,))).prepare()


def test_rule_30_matches_published_pattern():
    """Test that rule 30 from a single live cell reproduces its space-time diagram for 15 steps."""
    trajectory = ElementaryAutomatonSystem(30, size=41).prepare().simulate(until=15, dt=1)
    assert np.array_equal(trajectory.get_lattice('cells.alive').astype(int), _rule_30(15, 41))


def test_glider_moves_diagonally():
    """Test that a glider on a periodic 20 x 20 lattice has the same shape one cell further after 4 steps."""
    glider = np.zeros((20, 20))
    glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
    trajectory = GameOfLifeSystem((20, 20), initial=glider).prepare().simulate(until=4, dt=1)
    cells = trajectory.get_lattice('cells.alive')
    assert np.array_equal(cells[-1], np.roll(glider, (1, 1), axis=(0, 1)))
    assert not np.array_equal(cells[2], cells[0])


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_lookup_rules_agree_across_backends(backend):
    """Test that the fused and the NumPy lookup rules give the same Game of Life."""
    trajectory = GameOfLifeSystem((30, 30), backend=backend()).prepare().simulate(until=20, dt=1)
    reference = GameOfLifeSystem((30, 30), backend=NumpyBackend()).prepare().simulate(until=20, dt=1)
    assert np.array_equal(trajectory.states, reference.states)


def test_table_and_totalistic_rules():
    """Test that tables, totalistic, and outer totalistic rules give their documented next values."""

    three = LatticeNeighborhood(((-1,), (0,), (1,)))
    ring = _cells(
        LatticeSpace(5),
        lambda s: [UpdateLaw(s.target.bit, TotalisticSpec({1: 1, 3: 1}), neighborhood=three)],
        label='ring',
        bit=[0, 1, 0, 0, 0],
    )
    assert ring.prepare().simulate(until=1, dt=1).states[-1].tolist() == [1, 1, 1, 0, 0]
    swap = TableSpec({(0,): 1, (1,): 0})
    flip = _cells(
        LatticeSpace(3),
        lambda s: [UpdateLaw(s.target.bit, swap, neighborhood=LatticeNeighborhood(((0,),)))],
        label='ring',
        bit=[0, 1, 0],
    )
    assert flip.prepare().simulate(until=1, dt=1).states[-1].tolist() == [1, 0, 1]
    odd = OuterTotalisticSpec(lambda center, total: (center + total) % 2)
    count = _cells(
        LatticeSpace(4),
        lambda s: [UpdateLaw(s.target.bit, odd, neighborhood=LatticeNeighborhood(((-1,), (1,))))],
        label='ring',
        bit=[1, 0, 0, 0],
    )
    assert count.prepare().simulate(until=1, dt=1).states[-1].tolist() == [1, 1, 0, 1]


def test_synchronous_rule_shifts_whole_configuration():
    """Test that one synchronous step of a rule that copies the left neighbour shifts every cell by one."""
    assert _shift('synchronous').prepare().simulate(until=1, dt=1).states[-1].tolist() == [6, 1, 2, 3, 4, 5]


@pytest.mark.parametrize('schedule', ['random-sequential', 'random-site'])
def test_asynchronous_schedules_are_reproducible(schedule):
    """Test that asynchronous schedules copy values from neighbours and repeat with the same seed."""
    first = _shift(schedule).prepare().simulate(until=2, dt=1, random_state=3).states[-1]
    second = _shift(schedule).prepare().simulate(until=2, dt=1, random_state=3).states[-1]
    assert first.tolist() == second.tolist()
    assert set(first) <= {1, 2, 3, 4, 5, 6}
    assert first.tolist() != [5, 6, 1, 2, 3, 4]


def test_checkerboard_schedule_updates_sublattices_in_turn():
    """Test that a checkerboard update of the copy rule on a ring moves even sites first, then odd sites."""
    assert _shift('checkerboard').prepare().simulate(until=1, dt=1).states[-1].tolist() == [6, 6, 2, 2, 4, 4]


def test_checkerboard_needs_bipartite_neighbourhood():
    """Test that a checkerboard schedule on a Moore neighbourhood is rejected."""

    moore = RadiusNeighborhood(1, metric='chebyshev')

    def laws(s):
        """Return a rule that keeps the first neighbour, on a checkerboard."""
        return [UpdateLaw(s.target.alive, lambda *values: values[0], neighborhood=moore, schedule='checkerboard')]

    system = _cells(LatticeSpace((4, 4)), laws, alive=0.0)
    with pytest.raises(ValidationError, match='random-sequential'):
        system.prepare()


def test_unknown_schedule_is_rejected():
    """Test that an unknown schedule names the known ones."""
    with pytest.raises(ValidationError, match='synchronous'):
        UpdateLaw(
            var('x'),
            lambda *values: values[0],
            neighborhood=LatticeNeighborhood(((-1,), (0,), (1,))),
            schedule='parallel',
        ).prepare()


def test_forest_fire_is_reproducible():
    """Test that two forest fires with the same seed give identical lattices at every step."""
    first = ForestFireSystem((30, 30)).prepare().simulate(until=30, dt=1, random_state=7).states
    second = ForestFireSystem((30, 30)).prepare().simulate(until=30, dt=1, random_state=7).states
    third = ForestFireSystem((30, 30)).prepare().simulate(until=30, dt=1, random_state=8).states
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_rule_reads_parameters_by_keyword():
    """Test that a function rule receives the parameters it names, which change live."""

    def grow(s):
        """Return the rule that adds the rate to the value."""
        return [
            UpdateLaw(s.target.value, lambda value, *, rate: value + rate, neighborhood=LatticeNeighborhood(((0,),))),
        ]

    trajectory = (
        _cells(LatticeSpace(2), grow, value=[0.0, 0.0], rate=[1.0, 1.0])
        .prepare()
        .simulate(until=3, dt=1, schedule={'cells.rate': lambda t: 1.0 if t < 2 else 5.0})
    )
    assert trajectory.states[2].tolist() == [2.0, 2.0]
    assert trajectory.parameters[-1].tolist() == [5.0, 5.0]
    assert trajectory.states[-1].tolist() == [7.0, 7.0]


def test_relaxation_records_avalanches():
    """Test that a relaxation records the size, duration, and area of a toppling avalanche."""

    three = LatticeNeighborhood(((-1,), (0,), (1,)))

    def laws(s):
        """Return the relaxation of the pile after one grain falls on its middle site."""
        height = _symbolic(s).height
        topple = UpdateLaw(
            height,
            lambda left, height, right: height - 2 * (height >= 2) + (left >= 2) + (right >= 2),
            neighborhood=three,
        )
        drop = UpdateLaw(height, lambda z, *, random: z + (np.arange(z.size) == 2), arrays=True)
        value = relax(
            height,
            topple,
            lambda heights: bool((heights < 2).all()),
            drive=drop,
            activity=lambda heights: heights >= 2,
        )
        return [UpdateLaw(height, value)]

    pile = _cells(LatticeSpace(5, boundary='fixed'), laws, label='pile', height=[1, 1, 1, 1, 1])
    trajectory = pile.prepare().simulate(until=1, dt=1)
    (record,) = trajectory.events
    assert (record.label, record.time) == ('rule', 1.0)
    assert (record.size, record.duration, record.area) == (9, 5, 5)
    assert np.all(trajectory.states[-1] < 2)


def test_relaxation_without_end_raises():
    """Test that a relaxation that never becomes stable raises an error after its largest number of iterations."""

    def laws(s):
        """Return a relaxation that grows every value and is never stable."""
        value = _symbolic(s).value
        grow = UpdateLaw(value, lambda value: value + 1, neighborhood=LatticeNeighborhood(((0,),)))
        return [UpdateLaw(value, relax(value, grow, lambda _: False, max_iterations=10))]

    with pytest.raises(SolverError, match='not stable after 10 iterations'):
        _cells(LatticeSpace(2), laws, value=0.0).prepare().simulate(until=1, dt=1)


def test_local_rule_needs_lattice_group():
    """Test that a local rule on a state outside a lattice group is rejected."""
    cells = Element('cells', values={'value': [0.0, 0.0, 0.0]}, size=3)
    system = System(elements=[cells], interactions=[Interaction('rule', roles={'target': 'cells'}, laws=_copy)])
    with pytest.raises(ValidationError, match='update law of rule needs a LatticeSpace'):
        system.prepare()


def test_monte_carlo_energy_change_is_symbolic():
    """Test that a flip move of an Ising energy changes the energy by `2 J s sum(neighbours)`."""
    nearest = LatticeNeighborhood(((-1, 0), (0, -1), (0, 1), (1, 0)))

    def laws(s):
        """Return the Ising energy of a spin over the temperature."""
        spin = s.target.spin
        energy = -s.target.J * spin * sum(neighbors(spin, nearest, reduce='each'))
        return [ObjectiveLaw(energy / s.target.T, spin)]

    spins = Element('spins', values={'spin': 1, 'J': 1.0, 'T': 1.0}, space=LatticeSpace((4, 4)))
    solver = MetropolisSolver(moves={'spins.spin': 'flip'})
    system = System(
        elements=[spins],
        interactions=[Interaction('ising', roles={'target': 'spins'}, laws=laws)],
        solver=solver,
    )
    problem = system.prepare()._problem
    sampling = problem.lattice[0]
    change = sampling.delta(0.0, np.ones(3), *np.ones((4, 3)), -np.ones(3), 1.0, 1.0)[0]
    assert np.allclose(change, 8.0)
    assert isinstance(sampling.sampler, SamplerSpec)
    assert sampling.sampler.kind == 'metropolis'


@pytest.mark.parametrize('sampler', ['metropolis', 'glauber', 'heat-bath'])
def test_samplers_order_cold_ising(sampler):
    """Test that every sampler keeps a cold Ising lattice ordered and disorders a hot one."""
    cold = (
        IsingSystem(16, temperature=0.5, initial='up', sampler=sampler)
        .prepare()
        .simulate(
            until=200,
            dt=50,
            random_state=0,
        )
    )
    hot = (
        IsingSystem(16, temperature=10.0, initial='up', sampler=sampler)
        .prepare()
        .simulate(
            until=200,
            dt=50,
            random_state=0,
        )
    )
    assert abs(cold.states[-1].mean()) > 0.95
    assert abs(hot.states[-1].mean()) < 0.3


def test_temperature_changes_live():
    """Test that raising the temperature of an ordered lattice during a run disorders it."""
    trajectory = (
        IsingSystem(16, temperature=0.5, initial='up')
        .prepare()
        .simulate(
            until=200,
            dt=50,
            random_state=0,
            schedule={'spins.temperature': lambda time: 0.5 if time < 100 else 20.0},
        )
    )
    magnetization = np.abs(trajectory.get_values('spins.spin').mean(axis=1))
    assert magnetization[2] > 0.95
    assert magnetization[-1] < 0.3


def test_monte_carlo_wraps_angles_after_every_sweep():
    """Test that Monte Carlo rotations of the XY system keep every angle in (-π, π]."""
    values = XYSystem(16).prepare().simulate(until=50).get_values('spins.theta')
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

    def laws(s):
        """Return the relaxation of the pile after one grain falls on a random site."""
        height = _symbolic(s).height
        cross = RadiusNeighborhood(1, metric='manhattan', center=True)
        rules = UpdateLaw(height, topple, neighborhood=cross)
        drive = UpdateLaw(height, drop, arrays=True)
        value = relax(height, rules, lambda heights: bool((heights < 4).all()), drive=drive, activity=activity)
        return [UpdateLaw(height, value, label='avalanche')]

    heights = np.random.default_rng(0).integers(0, 4, size=(32, 32)).astype(float)
    return _cells(LatticeSpace((32, 32), boundary='fixed'), laws, label='pile', height=heights)


def test_relaxation_over_active_sites_matches_the_full_update():
    """Test that relaxing only active sites and their neighbours gives the heights and durations of full updates."""
    sparse_sites, full_sites = [], []
    sparse = _pile(lambda heights: heights >= 4, sparse_sites).prepare().simulate(until=5000, dt=5000, random_state=0)
    full = _pile(None, full_sites).prepare().simulate(until=5000, dt=5000, random_state=0)
    np.testing.assert_array_equal(sparse.get_values('pile.height')[-1], full.get_values('pile.height')[-1])
    assert [event.duration for event in sparse.events] == [event.duration for event in full.events]
    assert [event.time for event in sparse.events] == [event.time for event in full.events]
    assert sum(sparse_sites) < sum(full_sites) / 10
