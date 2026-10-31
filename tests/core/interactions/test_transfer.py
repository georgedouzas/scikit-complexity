"""Test the `_transfer` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import TransferInteraction
from skcomplex.core.laws import Sink, Source
from skcomplex.core.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec


class StockElement(Element):
    """Holder of a level with a capacity and a clock."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[float, StateSpec()] = 0.0,
        capacity: Annotated[float, StateSpec()] = 1.0,
        clock: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
        self.capacity = capacity
        self.clock = clock


def _run(system: System, until: float) -> Trajectory:
    """Return the trajectory of a system run to a time."""
    return Simulation(system, random_state=0).check().prepare().run(until=until).trajectory_


def _levels(trajectory: Trajectory, frame: int, group: str = 'cells') -> list[float]:
    """Return the levels of the members of a group at a frame."""
    return trajectory.agents(group, frame).values['level'].tolist()


def _refill(until: int, values: dict[str, object], amount: object, legs: tuple = ()) -> Trajectory:
    """Return the trajectory of two cells whose level a transfer from the environment refills."""
    cells = GroupElement('cells', element=StockElement, size=2, values=values)
    refill = TransferInteraction(
        'refill',
        source='cells',
        target='cells',
        state=Source(),
        into='level',
        amount=amount,
        legs=legs,
        form='agent',
    )
    return _run(System(elements=[cells], interactions=[refill]), until)


def test_transfer_conserves_the_total_of_a_continuous_flow():
    """Test that a symbolic flow between two holders keeps their total and relaxes them to equal levels."""
    exchange = TransferInteraction(
        'exchange',
        source='upper',
        target='lower',
        state='level',
        amount=lambda s: s.source.level - s.target.level,
    )
    system = System(elements=[StockElement('upper', level=4.0), StockElement('lower')], interactions=[exchange])
    trajectory = _run(system, 5.0)
    upper, lower = trajectory.values('upper.level'), trajectory.values('lower.level')
    assert np.allclose(upper + lower, 4.0)
    assert upper[-1] == pytest.approx(2.0 + 2.0 * np.exp(-10.0), abs=1e-4)


def test_transfer_moves_between_the_members_of_groups_and_the_environment():
    """Test that member transfers with a source and a sink change one end only, member by member."""
    tanks = GroupElement('tanks', element=StockElement, size=2, values={'level': [1.0, 3.0]})
    inflow = TransferInteraction('inflow', source='tanks', target='tanks', state=Source(), into='level', amount=1.0)
    outflow = TransferInteraction(
        'outflow',
        source='tanks',
        target='tanks',
        state='level',
        into=Sink(),
        amount=lambda s: s.source.level / 2,
    )
    trajectory = _run(System(elements=[tanks], interactions=[inflow, outflow]), 20.0)
    assert np.asarray(trajectory.values('tanks.level'))[-1] == pytest.approx([2.0, 2.0], abs=1e-3)


def test_transfer_guard_stops_a_flow_where_it_does_not_hold():
    """Test that a guarded flow stops once its guard fails."""
    drain = TransferInteraction(
        'drain',
        source='tank',
        target='tank',
        state='level',
        into=Sink(),
        amount=1.0,
        guard=lambda s: s.source.level > 1,
    )
    trajectory = Simulation(System(elements=[StockElement('tank', level=3.0)], interactions=[drain]), dt=0.01)
    level = trajectory.check().prepare().run(until=4.0).trajectory_.values('tank.level')[-1]
    assert level == pytest.approx(1.0, abs=0.02)


def test_transfer_rate_moves_units_by_jumps_and_conserves_the_total():
    """Test that a transfer at a rate moves whole units by jumps and keeps the total."""
    hop = TransferInteraction(
        'hop',
        source='left',
        target='right',
        state='level',
        amount=1.0,
        rate=lambda s: s.source.level,
    )
    system = System(elements=[StockElement('left', level=50.0), StockElement('right')], interactions=[hop])
    trajectory = _run(system, 1.0)
    left, right = trajectory.values('left.level'), trajectory.values('right.level')
    assert np.allclose(left + right, 50.0)
    assert np.allclose(right, np.round(right))
    assert 20 < right[-1] < 40


@pytest.mark.parametrize('pairing', ['random', 'pairs', 'neighbors'])
def test_transfer_agent_pairings_conserve_the_total(pairing):
    """Test that random, paired, and neighbouring agent transfers keep the total of the level."""
    cells = [(row, column) for row in range(4) for column in range(4)]
    traders = GroupElement('traders', element=StockElement, size=16, values={'level': 5.0}, location=cells)
    give = TransferInteraction(
        'give',
        source='traders',
        target='traders',
        state='level',
        amount=lambda source, target, model: model.rng.random(),
        pairing=pairing,
        form='agent',
        guard=lambda source, target, model: source.level >= 1,
    )
    space = LatticeSpace((4, 4), neighborhood='von_neumann')
    trajectory = _run(System(elements=[traders], interactions=[give], space=space), 30)
    levels = np.asarray(_levels(trajectory, 30, 'traders'))
    assert levels.sum() == pytest.approx(80.0)
    assert levels.min() >= 0
    assert levels.std() > 0


def test_transfer_regrows_linearly_up_to_the_capacity():
    """Test that a refill of a constant amount bounded by the capacity regrows a level linearly."""
    values = {'level': [0.0, 0.0], 'capacity': [1.0, 4.0]}
    trajectory = _refill(3, values, lambda cell, _, model: min(1.5, cell.capacity - cell.level))
    assert _levels(trajectory, 1) == [1.0, 1.5]
    assert _levels(trajectory, 3) == [1.0, 4.0]


def test_transfer_regrows_logistically_towards_the_capacity():
    """Test that a logistic refill follows `r' = r + rate r (1 - r / K)` and approaches the capacity."""
    values = {'level': [0.1, 0.0], 'capacity': [2.0, 2.0]}
    trajectory = _refill(60, values, lambda cell, _, model: 0.5 * cell.level * (1 - cell.level / cell.capacity))
    expected = 0.1
    for _ in range(5):
        expected += 0.5 * expected * (1 - expected / 2.0)
    assert _levels(trajectory, 5)[0] == pytest.approx(expected)
    assert _levels(trajectory, 60) == pytest.approx([2.0, 0.0])


def test_transfer_legs_restore_the_capacity_after_a_delay():
    """Test that a clock leg and a level leg restore a level when the clock ends, and restart the clock."""
    delay = 5.0

    def ticks(cell, _, model):
        """Return how much the clock falls, its restart to the delay when it ends below the capacity."""
        if cell.level >= cell.capacity:
            return 0.0
        return cell.clock - delay if cell.clock - 1 <= 0 else 1.0

    def restores(cell, _, model):
        """Return the missing level when the clock ends."""
        return cell.capacity - cell.level if cell.level < cell.capacity and cell.clock - 1 <= 0 else 0.0

    values = {'level': [0.0, 1.0], 'clock': [3.0, 3.0]}
    trajectory = _refill(3, values, restores, legs=(('clock', Sink(), ticks),))
    assert _levels(trajectory, 2) == [0.0, 1.0]
    assert _levels(trajectory, 3) == [1.0, 1.0]
    assert trajectory.agents('cells', 3).values['clock'].tolist() == [5.0, 3.0]


def test_transfer_moves_every_leg_between_two_agents():
    """Test that `transfer` moves the amounts of every leg read before any of them moves."""

    class AgentStub:
        """Agent with a level and a clock."""

        def __init__(self, level: float, clock: float) -> None:
            self.level = level
            self.clock = clock

    source, target = AgentStub(4.0, 2.0), AgentStub(0.0, 0.0)
    transfer = TransferInteraction(
        'move',
        source='a',
        target='b',
        state='level',
        amount=lambda first, second, model: first.level / 2,
        legs=(('clock', 'clock', lambda first, second, model: first.level / 4),),
        form='agent',
    )
    transfer.check().transfer(source, target, None)
    assert (source.level, target.level, source.clock, target.clock) == (2.0, 2.0, 1.0, 1.0)


def test_transfer_symbolic_flows_run_in_discrete_time():
    """Test that a symbolic transfer in discrete time moves its amount every step."""
    flow = TransferInteraction('flow', source='upper', target='lower', state='level', amount=1.0)
    system = System(
        elements=[StockElement('upper', level=3.0), StockElement('lower')],
        interactions=[flow],
        time=DiscreteTime(),
    )
    trajectory = Simulation(system, dt=1, random_state=0).check().prepare().run(until=2).trajectory_
    assert trajectory.values('lower.level').tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    'arguments',
    [
        {'pairing': 'chain'},
        {'legs': (('level',),)},
        {'state': Source(), 'into': Sink()},
        {'into': 3},
        {'amount': 'all'},
        {'rate': 1.0, 'form': 'agent'},
        {'pairing': 'random'},
        {'form': 'rule'},
        {'guard': 1},
        {'form': 'agent', 'schedule': 'parallel'},
        {'form': 'agent', 'actor': 'both'},
        {'form': 'agent', 'remove': 'both'},
        {'form': 'agent', 'schedule': 'member', 'pairing': 'pairs'},
        {'schedule': 'member'},
        {'form': 'agent', 'schedule': 'member', 'pairing': 'cell', 'partners': 0},
        {'form': 'agent', 'pairing': 'site', 'actor': 'target'},
    ],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown form or pairing, malformed legs and amounts, mismatched options, and a bad guard fail."""
    settings = {'state': 'level', 'amount': 1.0, **arguments}
    with pytest.raises(ValidationError):
        TransferInteraction('flow', source='a', target='b', **settings).check()


def test_transfer_actors_take_one_partner_of_their_cell_and_remove_it():
    """Test that every actor takes one random partner of its cell, gains its amount, and removes it."""
    prey = GroupElement('prey', element=StockElement, size=3, location=[(0, 0)] * 3, variable=True)
    hunters = GroupElement('hunters', element=StockElement, size=2, location=[(0, 0), (1, 1)], variable=True)
    eating = TransferInteraction(
        'eating',
        source='prey',
        target='hunters',
        state=Source(),
        into='level',
        amount=5.0,
        pairing='cell',
        form='agent',
        schedule='member',
        actor='target',
        partners=1,
        remove='source',
    )
    system = System(elements=[prey, hunters], interactions=[eating], space=LatticeSpace((2, 2)))
    trajectory = _run(system, 2)
    assert len(trajectory.agents('prey', 1).values['level']) == 2
    assert len(trajectory.agents('prey', 2).values['level']) == 1
    assert trajectory.agents('hunters', 2).values['level'].tolist() == [10.0, 0.0]
