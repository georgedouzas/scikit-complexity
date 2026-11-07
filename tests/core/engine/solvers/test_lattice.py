"""Test the `LatticeSolver` class."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.laws import CascadeLaw, DriveLaw, LocalRuleLaw, SampleLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import (
    BoxNeighborhood,
    LatticeNeighborhood,
    LatticeSpace,
    MooreNeighborhood,
    VonNeumannNeighborhood,
)
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import SolverError, ValidationError
from skcomplex.core.specs import (
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
from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem, GameOfLifeSystem
from skcomplex.systems.physics.criticality import ForestFireSystem
from skcomplex.systems.physics.statistical_mechanics import IsingSystem, XYSystem


class ShiftElement(Element):
    """Shift element."""

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
        """Return the laws of the shift element."""
        return [LocalRuleLaw(s.value, LatticeNeighborhood(((-1,),)), lambda left: left)]


def _shift(schedule):
    """Return a ring of five cells whose rule copies the left neighbour on a schedule."""

    class CopyElement(Element):
        """Copy element."""

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
            """Return the laws of the copy element."""
            return [LocalRuleLaw(s.value, LatticeNeighborhood(((-1,),)), lambda left: left, schedule=schedule)]

    return System(
        elements=[
            LatticeElement('ring', element=CopyElement, space=LatticeSpace(6), values={'value': [1, 2, 3, 4, 5, 6]}),
        ],
    )


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
    neighbours = LatticeSpace((10, 10)).prepare()._get_neighbors(0, MooreNeighborhood().prepare())
    assert sorted(neighbours.tolist()) == [1, 9, 10, 11, 19, 90, 91, 99]


def test_fixed_and_reflecting_edges():
    """Test that fixed edges read the fill value and reflecting edges read the mirror image."""
    fixed_space = LatticeSpace((5,), boundary='fixed').prepare()
    assert fixed_space._get_neighbors(0, BoxNeighborhood().prepare()).tolist() == [0, 1]
    reflecting_space = LatticeSpace((5,), boundary='reflecting').prepare()
    assert reflecting_space._get_neighbors(0, BoxNeighborhood(2).prepare())[:2].tolist() == [1, 0]

    class EdgeElement(Element):
        """Edge element."""

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
            """Return the laws of the edge element."""
            return [LocalRuleLaw(s.value, LatticeNeighborhood(((-1,),)), lambda left: left)]

    fixed = System(
        elements=[
            LatticeElement(
                'cells',
                element=EdgeElement,
                space=LatticeSpace(3, boundary='fixed', fill=9),
                values={'value': [1, 2, 3]},
            ),
        ],
    )
    assert Simulation(fixed, dt=1).prepare().run(until=1).state_.tolist() == [9, 1, 2]
    mirror = System(
        elements=[
            LatticeElement(
                'cells',
                element=EdgeElement,
                space=LatticeSpace(3, boundary='reflecting'),
                values={'value': [1, 2, 3]},
            ),
        ],
    )
    assert Simulation(mirror, dt=1).prepare().run(until=1).state_.tolist() == [1, 1, 2]


def test_neighborhoods_are_ordered():
    """Test the order and the sizes of the built-in neighbourhoods."""
    assert MooreNeighborhood(center=True).prepare().offsets_[0] == (0, 0)
    assert len(MooreNeighborhood(2).prepare().offsets_) == 24
    assert len(VonNeumannNeighborhood(2).prepare().offsets_) == 12
    assert len(MooreNeighborhood(n_dim=3).prepare().offsets_) == 26
    with pytest.raises(ValidationError, match='same number of dimensions'):
        LatticeNeighborhood(((0, 1), (1,))).prepare()


def test_rule_30_matches_published_pattern():
    """Test that rule 30 from a single live cell reproduces its space-time diagram for 15 steps."""
    trajectory = (
        Simulation(ElementaryAutomatonSystem(30, size=41).prepare().system_, dt=1).prepare().run(until=15).trajectory_
    )
    assert np.array_equal(trajectory.lattice('cells.alive').astype(int), _rule_30(15, 41))


def test_glider_moves_diagonally():
    """Test that a glider on a periodic 20 x 20 lattice has the same shape one cell further after 4 steps."""
    glider = np.zeros((20, 20))
    glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
    trajectory = (
        Simulation(GameOfLifeSystem((20, 20), initial=glider).prepare().system_, dt=1)
        .prepare()
        .run(until=4)
        .trajectory_
    )
    cells = trajectory.lattice('cells.alive')
    assert np.array_equal(cells[-1], np.roll(glider, (1, 1), axis=(0, 1)))
    assert not np.array_equal(cells[2], cells[0])


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_lookup_rules_agree_across_backends(backend):
    """Test that the fused and the NumPy lookup rules give the same Game of Life."""
    trajectory = (
        Simulation(GameOfLifeSystem((30, 30)).prepare().system_, dt=1, backend=backend())
        .prepare()
        .run(until=20)
        .trajectory_
    )
    reference = (
        Simulation(GameOfLifeSystem((30, 30)).prepare().system_, dt=1, backend=NumpyBackend())
        .prepare()
        .run(until=20)
        .trajectory_
    )
    assert np.array_equal(trajectory.states, reference.states)


def test_table_and_totalistic_rules():
    """Test that tables, totalistic, and outer totalistic rules give their documented next values."""

    class ParityElement(Element):
        """Parity element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            bit: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.bit = bit

        def _get_laws(self, s):
            """Return the laws of the parity element."""
            return [LocalRuleLaw(s.bit, BoxNeighborhood(), TotalisticSpec({1: 1, 3: 1}))]

    ring = System(
        elements=[
            LatticeElement('ring', element=ParityElement, space=LatticeSpace(5), values={'bit': [0, 1, 0, 0, 0]}),
        ],
    )
    assert Simulation(ring, dt=1).prepare().run(until=1).state_.tolist() == [1, 1, 1, 0, 0]

    class SwapElement(Element):
        """Swap element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            bit: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.bit = bit

        def _get_laws(self, s):
            """Return the laws of the swap element."""
            return [LocalRuleLaw(s.bit, LatticeNeighborhood(((0,),)), TableSpec({(0,): 1, (1,): 0}))]

    flip = System(
        elements=[LatticeElement('ring', element=SwapElement, space=LatticeSpace(3), values={'bit': [0, 1, 0]})],
    )
    assert Simulation(flip, dt=1).prepare().run(until=1).state_.tolist() == [1, 0, 1]

    class CountElement(Element):
        """Count element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            bit: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.bit = bit

        def _get_laws(self, s):
            """Return the laws of the count element."""
            return [
                LocalRuleLaw(
                    s.bit,
                    LatticeNeighborhood(((-1,), (1,))),
                    OuterTotalisticSpec(lambda center, total: (center + total) % 2),
                ),
            ]

    count = System(
        elements=[LatticeElement('ring', element=CountElement, space=LatticeSpace(4), values={'bit': [1, 0, 0, 0]})],
    )
    assert Simulation(count, dt=1).prepare().run(until=1).state_.tolist() == [1, 1, 0, 1]


def test_synchronous_rule_shifts_whole_configuration():
    """Test that one synchronous step of a rule that copies the left neighbour shifts every cell by one."""
    assert Simulation(_shift('synchronous'), dt=1).prepare().run(until=1).state_.tolist() == [6, 1, 2, 3, 4, 5]


@pytest.mark.parametrize('schedule', ['random-sequential', 'random-site'])
def test_asynchronous_schedules_are_reproducible(schedule):
    """Test that asynchronous schedules copy values from neighbours and repeat with the same seed."""
    first = Simulation(_shift(schedule), dt=1, random_state=3).prepare().run(until=2).state_
    second = Simulation(_shift(schedule), dt=1, random_state=3).prepare().run(until=2).state_
    assert first.tolist() == second.tolist()
    assert set(first) <= {1, 2, 3, 4, 5, 6}
    assert first.tolist() != [5, 6, 1, 2, 3, 4]


def test_checkerboard_schedule_updates_sublattices_in_turn():
    """Test that a checkerboard update of the copy rule on a ring moves even sites first, then odd sites."""
    assert Simulation(_shift('checkerboard'), dt=1).prepare().run(until=1).state_.tolist() == [6, 6, 2, 2, 4, 4]


def test_checkerboard_needs_bipartite_neighbourhood():
    """Test that a checkerboard schedule on a Moore neighbourhood is rejected."""

    class CellElement(Element):
        """Cell element."""

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
            """Return the laws of the cell element."""
            return [
                LocalRuleLaw(s.alive, MooreNeighborhood(), lambda *values: values[0], schedule='checkerboard'),
            ]

    system = System(elements=[LatticeElement('cells', element=CellElement, space=LatticeSpace((4, 4)))])
    with pytest.raises(ValidationError, match='random-sequential'):
        PassCompiler().prepare(system).run()


def test_unknown_schedule_is_rejected():
    """Test that an unknown schedule names the known ones."""
    with pytest.raises(ValidationError, match='synchronous'):
        LocalRuleLaw(None, BoxNeighborhood(), lambda *values: values[0], schedule='parallel').prepare()


def test_forest_fire_is_reproducible():
    """Test that two forest fires with the same seed give identical lattices at every step."""
    first = (
        Simulation(ForestFireSystem((30, 30)).prepare().system_, dt=1, random_state=7)
        .prepare()
        .run(until=30)
        .trajectory_.states
    )
    second = (
        Simulation(ForestFireSystem((30, 30)).prepare().system_, dt=1, random_state=7)
        .prepare()
        .run(until=30)
        .trajectory_.states
    )
    third = (
        Simulation(ForestFireSystem((30, 30)).prepare().system_, dt=1, random_state=8)
        .prepare()
        .run(until=30)
        .trajectory_.states
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_rule_reads_parameters_by_keyword():
    """Test that a function rule receives the parameters it names, which change live."""

    class GrowthElement(Element):
        """Growth element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            value: Annotated[object, StateSpec()] = 0.0,
            rate: Annotated[object, ParameterSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.value = value
            self.rate = rate

        def _get_laws(self, s):
            """Return the laws of the growth element."""
            return [LocalRuleLaw(s.value, LatticeNeighborhood(((0,),)), lambda value, *, rate: value + rate)]

    simulation = (
        Simulation(
            System(elements=[LatticeElement('cells', element=GrowthElement, space=LatticeSpace(2))]),
            dt=1,
        )
        .prepare()
        .run(until=2)
    )
    assert simulation.state_.tolist() == [2.0, 2.0]
    simulation.update_parameters({'cells.rate': 5.0})
    assert simulation.parameters_.tolist() == [5.0, 5.0]


def test_relaxation_records_avalanches():
    """Test that a relaxation records the size, duration, and area of a toppling avalanche."""

    class PileElement(Element):
        """Pile element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            height: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.height = height

        def _get_laws(self, s):
            """Return the laws of the pile element."""
            return [
                CascadeLaw(
                    drive=[DriveLaw(s.height, lambda height, *, random: height + (np.arange(height.size) == 2))],
                    relax=[
                        LocalRuleLaw(
                            s.height,
                            BoxNeighborhood(),
                            lambda left, height, right: height - 2 * (height >= 2) + (left >= 2) + (right >= 2),
                        ),
                    ],
                    stable=lambda height: bool((height < 2).all()),
                    activity=lambda height: height >= 2,
                ),
            ]

    space = LatticeSpace(5, boundary='fixed')
    trajectory = (
        Simulation(
            System(
                elements=[LatticeElement('pile', element=PileElement, space=space, values={'height': [1, 1, 1, 1, 1]})],
            ),
            dt=1,
        )
        .prepare()
        .run(1)
    )
    (record,) = trajectory.trajectory_.events
    assert (record.label, record.time) == ('pile', 1.0)
    assert (record.size, record.duration, record.area) == (9, 5, 5)
    assert np.all(trajectory.state_ < 2)


def test_relaxation_without_end_raises():
    """Test that a relaxation that never becomes stable raises an error after its largest number of iterations."""

    class RestlessElement(Element):
        """Restless element."""

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
            """Return the laws of the restless element."""
            return [
                CascadeLaw(
                    [],
                    [LocalRuleLaw(s.value, LatticeNeighborhood(((0,),)), lambda value: value + 1)],
                    lambda _: False,
                    10,
                ),
            ]

    with pytest.raises(SolverError, match='not stable after 10 iterations'):
        Simulation(
            System(elements=[LatticeElement('cells', element=RestlessElement, space=LatticeSpace(2))]),
            dt=1,
        ).prepare().run(until=1)


def test_local_rule_needs_lattice_group():
    """Test that a local rule on a state outside a lattice group is rejected."""
    with pytest.raises(ValidationError, match='LocalRuleLaw of cells needs a LatticeSpace'):
        PassCompiler().prepare(System(elements=[GroupElement('cells', element=ShiftElement, size=3)])).run()


def test_monte_carlo_energy_change_is_symbolic():
    """Test that a flip move of an Ising energy changes the energy by `2 J s sum(neighbours)`."""
    nearest = VonNeumannNeighborhood()

    class SpinElement(Element):
        """Spin element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            spin: Annotated[object, StateSpec(domain=ZZ)] = 1.0,
            J: Annotated[object, ParameterSpec()] = 1.0,
            intensity: Annotated[object, ParameterSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.spin = spin
            self.J = J
            self.intensity = intensity

        def _get_laws(self, s):
            """Return the laws of the spin element."""
            return [
                SampleLaw(
                    ObjectiveSpec(-s.J * s.spin * sum(neighbors(s.spin, nearest, reduce='each')), nearest),
                    MoveSpec(s.spin),
                ),
            ]

    system = System(elements=[LatticeElement('spins', element=SpinElement, space=LatticeSpace((4, 4)))])
    problem = PassCompiler().prepare(system).run().problem_
    sampling = problem.lattice[0]
    change = sampling.delta(0.0, np.ones(3), *np.ones((4, 3)), -np.ones(3), 1.0, 1.0)[0]
    assert np.allclose(change, 8.0)
    assert isinstance(sampling.sampler, SamplerSpec)
    assert sampling.sampler.kind == 'metropolis'


@pytest.mark.parametrize('sampler', ['metropolis', 'glauber', 'heat-bath'])
def test_samplers_order_cold_ising(sampler):
    """Test that every sampler keeps a cold Ising lattice ordered and disorders a hot one."""
    cold = (
        Simulation(
            IsingSystem(16, temperature=0.5, initial='up', sampler=sampler).prepare().system_,
            dt=50,
            random_state=0,
        )
        .prepare()
        .run(200)
    )
    hot = (
        Simulation(
            IsingSystem(16, temperature=10.0, initial='up', sampler=sampler).prepare().system_,
            dt=50,
            random_state=0,
        )
        .prepare()
        .run(200)
    )
    assert abs(cold.state_.mean()) > 0.95
    assert abs(hot.state_.mean()) < 0.3


def test_temperature_changes_live():
    """Test that raising the temperature of an ordered lattice during a run disorders it."""
    simulation = (
        Simulation(
            IsingSystem(16, temperature=0.5, initial='up').prepare().system_,
            dt=50,
            random_state=0,
            schedule={'spins.temperature': lambda time: 0.5 if time < 100 else 20.0},
        )
        .prepare()
        .run(until=200)
    )
    magnetization = np.abs(simulation.trajectory_.values('spins.spin').mean(axis=1))
    assert magnetization[2] > 0.95
    assert magnetization[-1] < 0.3


def test_monte_carlo_wraps_angles_after_every_sweep():
    """Test that Monte Carlo rotations of the XY system keep every angle in (-π, π]."""
    values = Simulation(XYSystem(16).prepare().system_).prepare().run(until=50).trajectory_.values('spins.theta')
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

    class PileElement(Element):
        """Pile element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            height: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.height = height

        def _get_laws(self, s):
            """Return the laws of the pile element."""
            return [
                CascadeLaw(
                    drive=[DriveLaw(s.height, drop)],
                    relax=[LocalRuleLaw(s.height, VonNeumannNeighborhood(center=True), topple)],
                    stable=lambda heights: bool((heights < 4).all()),
                    activity=activity,
                    label='avalanche',
                ),
            ]

    heights = np.random.default_rng(0).integers(0, 4, size=(32, 32)).astype(float)
    space = LatticeSpace((32, 32), boundary='fixed')
    return System(elements=[LatticeElement('pile', element=PileElement, space=space, values={'height': heights})])


def test_relaxation_over_active_sites_matches_the_full_update():
    """Test that relaxing only active sites and their neighbours gives the heights and durations of full updates."""
    sparse_sites, full_sites = [], []
    sparse = Simulation(_pile(lambda heights: heights >= 4, sparse_sites), dt=5000, random_state=0).prepare()
    sparse = sparse.run(until=5000).trajectory_
    full = Simulation(_pile(None, full_sites), dt=5000, random_state=0).prepare().run(until=5000).trajectory_
    np.testing.assert_array_equal(sparse.values('pile.height')[-1], full.values('pile.height')[-1])
    assert [event.duration for event in sparse.events] == [event.duration for event in full.events]
    assert [event.time for event in sparse.events] == [event.time for event in full.events]
    assert sum(sparse_sites) < sum(full_sites) / 10
