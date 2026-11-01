"""Test the `_transfer` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, FieldElement, GroupElement
from skcomplex.core.interactions import TransferInteraction
from skcomplex.core.laws import BaseLaw, RateLaw, RuleLaw, Sink, Source
from skcomplex.core.laws.utils import bernoulli
from skcomplex.core.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec


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


class MortalElement(StockElement):
    """Holder of a level that leaves the system in the phase `die` of its first step."""

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that removes the holder."""
        return [RuleLaw(lambda agent, model: agent.remove(), phase='die')]


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


@pytest.mark.parametrize('pairing', ['random', 'pairs', 'neighbors', 'neighbor-pairs'])
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


def test_transfer_reports_named_flows_under_the_element_they_move_within():
    """Test that named flows within one element report under it, and the others under the interaction."""
    within = TransferInteraction(
        'within',
        source='tank',
        target='tank',
        state='level',
        into='capacity',
        amount=1.0,
        flow='filling',
        legs=(('capacity', Sink(), 0.5, 'spill'), ('clock', Sink(), 0.25)),
    )
    between = TransferInteraction('between', source='tank', target='other', state='level', amount=0.5, flow='leak')
    system = System(elements=[StockElement('tank', level=10.0), StockElement('other')], interactions=[within, between])
    trajectory = _run(system, 1.0)
    assert {'tank.filling', 'tank.spill', 'within.flow_2', 'between.leak'} <= set(trajectory.names)
    assert trajectory.values('tank.filling')[-1] == pytest.approx(1.0)
    assert trajectory.values('between.leak')[-1] == pytest.approx(0.5)
    nested = System(systems=[System('plant', elements=[StockElement('tank', level=10.0)], interactions=[within])])
    assert {'plant/tank.filling', 'plant/within.flow_2'} <= set(_run(nested, 1.0).names)


def test_transfer_neighbor_pairs_are_disjoint_neighbours():
    """Test that neighbour pairs share no agent and pair agents of neighbouring cells only."""
    cells = [(row, 0) for row in range(6)]
    traders = GroupElement('traders', element=StockElement, size=6, values={'level': 1.0}, location=cells)
    give = TransferInteraction(
        'give',
        source='traders',
        target='traders',
        state='level',
        into='clock',
        amount=1.0,
        pairing='neighbor-pairs',
        form='agent',
    )
    space = LatticeSpace((6, 1), neighborhood='von_neumann')
    trajectory = _run(System(elements=[traders], interactions=[give], space=space), 1)
    levels = trajectory.agents('traders', 1).values['level'].tolist()
    clocks = trajectory.agents('traders', 1).values['clock'].tolist()
    assert sorted(set(levels)) <= [0.0, 1.0]
    assert levels.count(0.0) == sum(clocks) >= 1
    assert all(level + clock <= 1.0 + 1.0 for level, clock in zip(levels, clocks, strict=True))


def test_transfer_fallback_receives_when_no_target_lives():
    """Test that the fallback group receives the transfers of the steps in which no target member lives."""
    payers = GroupElement('payers', element=StockElement, size=2, values={'level': 3.0})
    payees = GroupElement('payees', element=MortalElement, size=1, variable=True)
    others = GroupElement('others', element=StockElement, size=1)
    pay = TransferInteraction(
        'pay',
        source='payers',
        target='payees',
        state='level',
        amount=1.0,
        pairing='random',
        fallback='others',
        form='agent',
        phase='pay',
    )
    time = DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('pay'), PhaseSpec('die')]))
    trajectory = _run(System(elements=[payers, payees, others], interactions=[pay], time=time), 2)
    assert _levels(trajectory, 2, 'others') == [2.0]
    assert _levels(trajectory, 2, 'payers') == [1.0, 1.0]


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


class ClientElement(StockElement):
    """Holder of a level with a reference to a holder of the group `holders`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[float, StateSpec()] = 0.0,
        holder: Annotated[float, StateSpec(reference='holders')] = -1.0,
    ) -> None:
        super().__init__(label, visual=visual, level=level)
        self.holder = holder


def test_transfer_leg_end_moves_the_state_of_a_referenced_member():
    """Test that a leg end `holder.level` moves the level of the member the reference state names."""
    clients = GroupElement('clients', element=ClientElement, size=3, values={'level': 1.0, 'holder': [0, 1, 1]})
    holders = GroupElement('holders', element=StockElement, size=2)
    deposit = TransferInteraction(
        'deposit',
        source='clients',
        target='clients',
        state='level',
        into='holder.level',
        amount=1.0,
        form='agent',
    )
    trajectory = _run(System(elements=[clients, holders], interactions=[deposit]), 1)
    assert _levels(trajectory, 1, 'clients') == [0.0, 0.0, 0.0]
    assert _levels(trajectory, 1, 'holders') == [1.0, 2.0]


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
        {'flow': 3},
        {'flow': 'out', 'legs': (('capacity', 'capacity', 1.0, 'out'),)},
        {'flow': 'out', 'form': 'agent'},
        {'fallback': 'c'},
        {'into': 'holder.level'},
        {'reverse_rate': 1.0},
        {'rate': 1.0, 'reverse_rate': 'back'},
        {'offset': 1},
        {'offset': 1.5},
        {'boundary': 'reflecting'},
    ],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that unknown forms or pairings, malformed legs, amounts, or flow names, and mismatched options fail."""
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


class LevelElement(Element):
    """Field element whose level stays as it is."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u

    def _get_laws(self, s):
        """Return the still rate of the level."""
        return [RateLaw(s.u, 0 * s.u)]


@pytest.mark.parametrize('schedule', ['random-sequential', 'member'])
def test_transfer_site_deposits_from_the_agents_into_the_field_at_their_cells(schedule):
    """Test that every source agent moves its amount into the site of the target field at its cell."""
    markers = GroupElement('markers', element=StockElement, size=2, values={'level': 10.0}, location=[(1, 2), (0, 0)])
    marking = TransferInteraction(
        'marking',
        source='walkers/markers',
        target='ground/level',
        state='level',
        into='u',
        amount=1.0,
        pairing='site',
        form='agent',
        schedule=schedule,
    )
    system = System(
        systems=[
            System('walkers', elements=[markers], space=LatticeSpace((4, 3))),
            System('ground', elements=[FieldElement('level', element=LevelElement, space=LatticeSpace((4, 3)))]),
        ],
        interactions=[marking],
    )
    trajectory = Simulation(system, dt=1).check().prepare().run(until=3).trajectory_
    level = trajectory.lattice('ground/level.u')[-1]
    assert (level[1, 2], level[0, 0], level.sum()) == (3.0, 3.0, 6.0)
    assert trajectory.agents('walkers/markers', 3).values['level'].tolist() == [7.0, 7.0]


class RelayInteraction(TransferInteraction):
    """Transfer at a parameter rate from a first holder through a middle holder to a last holder."""

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        middle: Annotated[str, RoleSpec()],
        last: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(
            label,
            source=first,
            target=middle,
            state='first.level',
            into='middle.level',
            amount=lambda s: s.k * s.first.level,
            flow='inner',
            legs=[('middle.level', 'last.level', lambda s: s.k * s.middle.level, 'outer')],
        )
        self.first = first
        self.middle = middle
        self.last = last
        self.k = k

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the first and the middle holder."""
        return [(namespace.first, namespace.middle)]


def test_transfer_legs_move_between_the_holders_of_several_roles_at_a_parameter_rate():
    """Test that legs naming roles move amounts along a chain of three holders, reported under the interaction."""
    relay = RelayInteraction('relay', first='a', middle='b', last='c', k=0.5)
    holders = [StockElement('a', level=2.0), StockElement('b'), StockElement('c')]
    trajectory = _run(System(elements=holders, interactions=[relay]), 40.0)
    levels = [trajectory.values(f'{label}.level')[-1] for label in 'abc']
    assert levels == pytest.approx([0.0, 0.0, 2.0], abs=1e-3)
    assert trajectory.values('relay.inner')[0] == pytest.approx(1.0)


def test_transfer_offset_moves_along_the_group_order_in_discrete_time():
    """Test that a transfer by an offset moves the amount of every member to the next one, around a ring."""
    cells = GroupElement('cells', element=StockElement, size=4, values={'level': [4.0, 0.0, 0.0, 0.0]})
    shift = TransferInteraction(
        'shift',
        source='cells',
        target='cells',
        state='level',
        amount=lambda s: s.source.level,
        offset=1,
    )
    system = System(elements=[cells], interactions=[shift], time=DiscreteTime())
    trajectory = Simulation(system, dt=1, random_state=0).check().prepare().run(until=4).trajectory_
    levels = np.asarray(trajectory.values('cells.level'))[1:].tolist()
    assert levels == [[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 4.0], [4.0, 0.0, 0.0, 0.0]]


def test_transfer_offset_moves_nothing_past_a_fixed_boundary():
    """Test that a flow along a line with a fixed boundary keeps the total, with nothing leaving the last member."""
    cells = GroupElement('cells', element=StockElement, size=3, values={'level': [3.0, 0.0, 0.0]})
    flow = TransferInteraction(
        'flow',
        source='cells',
        target='cells',
        state='level',
        amount=lambda s: s.source.level,
        offset=1,
        boundary='fixed',
    )
    trajectory = _run(System(elements=[cells], interactions=[flow]), 30.0)
    levels = np.asarray(trajectory.values('cells.level'))
    assert np.allclose(levels.sum(axis=1), 3.0)
    assert levels[-1] == pytest.approx([0.0, 0.0, 3.0], abs=1e-3)


def test_transfer_offset_shares_one_draw_per_edge():
    """Test that both ends of an edge read one draw, so units are kept and hop at the current of a parallel TASEP."""
    cells = GroupElement('cells', element=StockElement, size=50, values={'level': np.tile([1.0, 0.0], 25)})
    hops = TransferInteraction(
        'hops',
        source='cells',
        target='cells',
        state='level',
        amount=lambda s: s.source.level * (1 - s.target.level) * bernoulli('hop', 0.5),
        flow='hops',
        offset=1,
    )
    system = System(elements=[cells], interactions=[hops], time=DiscreteTime())
    trajectory = Simulation(system, dt=1, random_state=0).check().prepare().run(until=50).trajectory_
    levels = np.asarray(trajectory.values('cells.level'))
    assert set(np.unique(levels)) == {0.0, 1.0}
    assert set(levels.sum(axis=1)) == {25.0}
    current = (1 - np.sqrt(1 - 4 * 0.5 * 0.5 * 0.5)) / 2
    assert np.asarray(trajectory.values('cells.hops'))[25:].mean() == pytest.approx(current, rel=0.2)


def test_transfer_reverse_rate_returns_units_to_the_environment():
    """Test that reverse jumps of a leg from `Source()` return units to the environment, balancing the entries."""
    bath = TransferInteraction(
        'bath',
        source='tank',
        target='tank',
        state=Source('bath'),
        into='level',
        amount=1.0,
        rate=2.0,
        reverse_rate=lambda s: s.target.level,
    )
    trajectory = _run(System(elements=[StockElement('tank')], interactions=[bath]), 400.0)
    levels = np.asarray(trajectory.values('tank.level'))
    assert np.allclose(levels, np.round(levels))
    assert levels[50:].mean() == pytest.approx(2.0, rel=0.25)


class GiftInteraction(TransferInteraction):
    """Gift of one unit of level from every donor to a receiver, a transfer whose roles have names of their own."""

    def __init__(
        self,
        label: str,
        *,
        donors: Annotated[str, RoleSpec()],
        receivers: Annotated[str, RoleSpec()],
        pairing: str = 'members',
        form: str = 'symbolic',
        schedule: str = 'random-sequential',
    ) -> None:
        super().__init__(
            label,
            source=donors,
            target=receivers,
            state='level',
            amount=1.0,
            pairing=pairing,
            form=form,
            schedule=schedule,
            actor='target',
        )
        self.donors = donors
        self.receivers = receivers

    def _get_role(self, role: str) -> str:
        """Return the donors as the source and the receivers as the target."""
        return {'source': 'donors', 'target': 'receivers'}.get(role, role)


@pytest.mark.parametrize(
    ('form', 'pairing', 'schedule'),
    [
        ('symbolic', 'members', 'random-sequential'),
        ('agent', 'members', 'random-sequential'),
        ('agent', 'random', 'member'),
    ],
)
def test_transfer_subclass_resolves_its_own_role_names_in_both_forms(form, pairing, schedule):
    """Test that a subclass that names its roles otherwise transfers in the symbolic and the agent form."""
    variable = form == 'agent'
    donors = GroupElement('givers', element=StockElement, size=1, values={'level': 4.0}, variable=variable)
    receivers = GroupElement('takers', element=StockElement, size=1, variable=variable)
    gift = GiftInteraction('gift', donors='givers', receivers='takers', pairing=pairing, form=form, schedule=schedule)
    trajectory = _run(System(elements=[donors, receivers], interactions=[gift]), 1)
    if variable:
        assert _levels(trajectory, 1, 'givers') + _levels(trajectory, 1, 'takers') == [3.0, 1.0]
    else:
        assert np.asarray(trajectory.values('givers.level'))[-1] == pytest.approx([3.0])
        assert np.asarray(trajectory.values('takers.level'))[-1] == pytest.approx([1.0])
