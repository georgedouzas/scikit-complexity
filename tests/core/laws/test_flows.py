"""Test the `_flows` module of `skcomplex.core.laws`."""

import functools
import pickle  # nosec B403: round-trips the pickles the tests themselves make
from operator import attrgetter

import numpy as np
import pytest
from scipy import stats

from skcomplex.core.base import ConservationError, LabelError, ValidationError, parameter, role, state
from skcomplex.core.components import ElementComponent
from skcomplex.core.interactions import BaseInteraction
from skcomplex.core.laws import (
    Auxiliary,
    BoundaryLaw,
    Invariant,
    RateLaw,
    Sink,
    Source,
    TransferLaw,
    UpdateLaw,
    distributed_delay,
    lookup,
    step,
    transfer,
    where,
)
from skcomplex.core.space import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.analysis import conservation, loop_dominance, loops
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import ConservationObserver, Simulation
from skcomplex.formalisms.agents import RuleLaw


class TanksComponent(ElementComponent):
    """Two stocks with a transfer between them and the invariant of their total."""

    a: float = state(3.0)
    b: float = state(1.0)

    def laws(self, s):
        """Return the transfer and the invariant."""
        return [TransferLaw(s.a, s.b, s.a / 4, label='flow'), Invariant('total', s.a + s.b)]


def _tub_laws(s):
    """Return a faucet of 5 and a drain proportional to the level."""
    return [
        TransferLaw(Source(), s.water, 5.0, label='faucet'),
        TransferLaw(s.water, Sink(), s.k * s.water, label='drain'),
    ]


def _population_laws(s):
    """Return births through an auxiliary and deaths proportional to the population."""
    births = Auxiliary('birth_rate', s.b * s.n)
    return [
        births,
        TransferLaw(Source(), s.n, births.symbol, label='births'),
        TransferLaw(s.n, Sink(), s.d * s.n, label='deaths'),
    ]


def _clipped_laws(s):
    """Return an inflow and a boundary law that clips the stock, which destroys the quantity."""
    return [TransferLaw(Source(), s.water, 1.0, label='faucet'), BoundaryLaw(s.water, where(s.water > 2, 2, s.water))]


def _ledger_laws(s):
    """Return a per-step payment between two holders and an untagged counter updated every step."""
    return [
        TransferLaw(s.a, s.b, 0.1 * s.a, label='pay'),
        TransferLaw(Source(), s.a, 1.0, label='lend'),
        UpdateLaw(s.c, s.c + 1),
    ]


def _fertility_laws(s):
    """Return an auxiliary that reads a table of food per person."""
    return [Auxiliary('fertility', lookup(s.food, [(0, 0.0), (1, 2.0), (2, 3.0)])), RateLaw(s.food, 1.0)]


TubComponent = ElementComponent.define(
    'TubComponent',
    states={'water': state(0.0, quantity='water')},
    parameters={'k': 0.5},
    laws=_tub_laws,
)


PopulationComponent = ElementComponent.define(
    'PopulationComponent',
    states={'n': state(10.0, quantity='people')},
    parameters={'b': 0.03, 'd': 0.01},
    laws=_population_laws,
)


ClippedComponent = ElementComponent.define(
    'ClippedComponent',
    states={'water': state(0.0, quantity='water')},
    laws=_clipped_laws,
)


LedgerComponent = ElementComponent.define(
    'LedgerComponent',
    states={'a': state(10.0, quantity='money'), 'b': state(0.0, quantity='money'), 'c': state(0.0)},
    laws=_ledger_laws,
)


FertilityComponent = ElementComponent.define('FertilityComponent', states={'food': state(-1.0)}, laws=_fertility_laws)


def _delayed(kind):
    """Return an element whose output stock integrates a delay of a unit step at time 1 with delay time 6."""
    conserved, order = kind.startswith('material'), int(kind[-1])
    function = functools.partial(distributed_delay, order=order, conserved=conserved)
    return ElementComponent.define(
        f'Delayed_{kind}',
        states={'out': state(0.0)},
        parameters={'T': 6.0},
        laws=lambda s: [RateLaw(s.out, function(step(s.t, 1.0), s.T))],
    )


def _balance(agent):
    """Return assets minus liabilities minus equity."""
    return agent.assets - agent.liabilities - agent.equity


class BankComponent(ElementComponent):
    """Bank whose rule lowers its assets alone."""

    assets: float = state(10.0)
    liabilities: float = state(8.0)
    equity: float = state(2.0)

    def laws(self, s):
        """Return the rule that loses assets and the invariant of the balance."""
        return [RuleLaw(self.lose), Invariant('assets == liabilities + equity', _balance)]

    def lose(self, agent, model):
        """Lower the assets by 1."""
        agent.assets -= 1.0


class LenderComponent(ElementComponent):
    """Lender holding claims on firms, positive loans."""

    loans: float = state(0.0, quantity='loans')
    money: float = state(0.0, quantity='money')


class BorrowerComponent(ElementComponent):
    """Firm that owes its loans, a negative holding, and goes bankrupt when its money runs out."""

    loans: float = state(0.0, quantity='loans')
    money: float = state(0.0, quantity='money')
    settles: float = state(1.0)

    def laws(self, s):
        """Return the rule that removes bankrupt firms."""
        return [RuleLaw(self.default)]

    def default(self, agent, model):
        """Remove the firm when its money runs out, after writing off its debt when it settles."""
        if agent.money > 0:
            return
        if agent.settles:
            lender = model.agents('lenders')[0]
            transfer(agent, lender, agent.loans, 'loans')
        agent.remove()


def _credit_system(settles):
    """Return a lender and two bankrupt borrowers that each owe 5."""
    return System(
        components=[
            LenderComponent.group('lenders', 1, loans=10.0, money=4.0),
            BorrowerComponent.group('firms', 2, variable=True, loans=-5.0, money=0.0, settles=float(settles)),
        ],
    )


def _total_money(model):
    """Return the money of every holder bound to the payments."""
    return sum(agent.money for agent in model.agents('holders'))


class PaymentsInteraction(BaseInteraction):
    """Payments between holders, which create money when asked, under the invariant of the total."""

    holders: str = role()
    creates: float = parameter(0.0)

    def laws(self, s):
        """Return the rule that moves money and the invariant of the total."""
        return [RuleLaw(self.move), Invariant('total money', _total_money)]

    def move(self, model):
        """Move one unit from the first holder to the second, and create money when the payments create it."""
        first, second = model.agents('holders')[:2]
        transfer(first, second, 1.0, 'money')
        first.money += model.parameter('creates')


@pytest.mark.parametrize('time', [None, DiscreteTime()])
def test_transfer_conserves_across_time_classes(time):
    """Test that a transfer keeps the total of two stocks in continuous and in discrete time."""
    trajectory = Simulation(System(components=[TanksComponent('t')], time=time), dt=1.0).run(until=5).trajectory_
    np.testing.assert_allclose(trajectory['t.a'] + trajectory['t.b'], 4.0, atol=1e-12)
    assert trajectory['t.a'][-1] < 3.0


def test_transfer_with_the_environment_takes_none():
    """Test that a transfer from `None` draws the quantity from the environment."""
    FilledComponent = ElementComponent.define(
        'FilledComponent',
        states={'n': 0.0},
        laws=lambda s: [TransferLaw(None, s.n, 1.0, label='inflow')],
    )
    trajectory = Simulation(System(components=[FilledComponent('f')]), dt=1.0).run(until=3).trajectory_
    assert trajectory['f.n'][-1] == pytest.approx(3.0)


def test_bathtub_approaches_the_balance_of_its_flows_and_records_them():
    """Test that the stock approaches inflow / k = 10, with both flows recorded and the inflow accumulated."""
    trajectory = Simulation(System(components=[TubComponent('tub')]), dt=1).run(30).trajectory_
    assert trajectory['tub.water'][-1] == pytest.approx(10.0, rel=1e-3)
    assert trajectory['tub.faucet'][-1] == pytest.approx(5.0)
    assert trajectory['tub.drain'][-1] == pytest.approx(5.0, rel=1e-3)
    assert trajectory['__boundary.water'][-1] == pytest.approx(trajectory['tub.water'][-1], rel=1e-6)


def test_flow_between_quantities_is_rejected():
    """Test that a flow between holders of different quantities raises a conservation error."""
    mixed = ElementComponent.define(
        'MixedComponent',
        states={'x': state(1.0, quantity='mass'), 'y': state(0.0, quantity='energy')},
        laws=lambda s: [TransferLaw(s.x, s.y, s.x)],
    )
    with pytest.raises(ConservationError, match='different quantities'):
        compile_system(System(components=[mixed('m')]))


def test_flow_holders_and_labels_are_checked():
    """Test that flows reject parameters as holders, environment to environment, and duplicate labels."""
    with pytest.raises(ValidationError, match='both of its holders'):
        TransferLaw(Source(), Sink(), 1.0)
    twice = ElementComponent.define(
        'TwiceComponent',
        states={'x': state(1.0)},
        laws=lambda s: [TransferLaw(Source(), s.x, 1.0, label='f'), TransferLaw(s.x, Sink(), 1.0, label='f')],
    )
    with pytest.raises(LabelError, match="'f'"):
        compile_system(System(components=[twice('twice')]))
    reads = ElementComponent.define(
        'ReadsComponent',
        states={'x': state(1.0)},
        parameters={'k': 1.0},
        laws=lambda s: [TransferLaw(s.k, s.x, 1)],
    )
    with pytest.raises(ValidationError, match='not a state'):
        compile_system(System(components=[reads('reads')]))


def test_discrete_flows_are_per_step_and_join_the_updates():
    """Test that in a discrete map a flow moves its amount once per step and money changes only by lending."""
    trajectory = Simulation(System(components=[LedgerComponent('ledger')]), dt=1).run(5).trajectory_
    assert trajectory['ledger.a'].tolist() == pytest.approx([10.0] * 6)
    assert trajectory['ledger.b'].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert trajectory['__boundary.money'][-1] == pytest.approx(5.0)
    assert trajectory['ledger.c'][-1] == pytest.approx(5.0)


def test_jump_flows_move_their_size_at_random_times():
    """Test that a flow with a rate becomes a jump that conserves the total of its holders."""
    jumping = ElementComponent.define(
        'JumpingComponent',
        states={'a': state(50.0, quantity='coins'), 'b': state(0.0, quantity='coins')},
        laws=lambda s: [TransferLaw(s.a, s.b, 1, rate=0.1 * s.a, label='move')],
    )
    trajectory = Simulation(System(components=[jumping('j')]), dt=1, random_state=0).run(10).trajectory_
    assert np.all(trajectory['j.a'] + trajectory['j.b'] == 50.0)
    assert trajectory['j.b'][-1] > 0


def test_table_function_interpolates_and_stays_flat():
    """Test that the fertility auxiliary interpolates the table linearly and holds its ends."""
    trajectory = Simulation(System(components=[FertilityComponent('f')]), dt=0.5).run(4).trajectory_
    food = trajectory['f.food']
    expected = np.interp(food, [0, 1, 2], [0.0, 2.0, 3.0])
    assert trajectory['f.fertility'] == pytest.approx(expected)
    assert trajectory['f.fertility'][0] == 0.0
    assert trajectory['f.fertility'][-1] == 3.0


def test_third_order_delay_follows_the_erlang_step_response():
    """Test that the output of a third-order delay of a step reaches half the step 5.35 after it."""
    trajectory = Simulation(System(components=[_delayed('material_delay3')('d')]), dt=0.01).run(14).trajectory_
    times = np.asarray(trajectory.times)
    output = trajectory['d.material_delay3_0'][:, 2] / 2.0
    assert times[np.argmax(output >= 0.5)] - 1.0 == pytest.approx(stats.gamma(3, scale=2).median(), abs=0.02)
    assert stats.gamma(3, scale=2).median() == pytest.approx(5.35, abs=0.01)


@pytest.mark.parametrize(('kind', 'stages'), [('material_delay1', 1), ('smoothing_delay1', 1), ('smoothing_delay3', 3)])
def test_delays_and_smoothings_add_hidden_stocks(kind, stages):
    """Test that delays and smoothings add one hidden stock per stage that tends to the step."""
    trajectory = Simulation(System(components=[_delayed(kind)('d')]), dt=1).run(80).trajectory_
    hidden = np.atleast_2d(trajectory[f'd.{kind}_0'].T).T
    assert hidden.shape[1] == stages
    level = 6.0 / stages if kind == 'material_delay1' else 1.0
    assert hidden[-1, -1] == pytest.approx(level, rel=1e-3)


def test_population_has_one_reinforcing_and_one_balancing_loop():
    """Test that births close a reinforcing loop and deaths a balancing loop, and births dominate."""
    system = System(components=[PopulationComponent('pop')])
    found = {loop.nodes: loop.polarity for loop in loops(system).loops}
    assert found == {('pop.n', 'pop.deaths'): 'balancing', ('pop.n', 'pop.birth_rate', 'pop.births'): 'reinforcing'}
    trajectory = Simulation(system, dt=1).run(5).trajectory_
    dominance = loop_dominance(trajectory)
    assert dominance.gains[-1] == pytest.approx([-0.01, 0.03])
    assert dominance.loops[int(dominance.dominant[-1])].polarity == 'reinforcing'


def test_conservation_returns_the_boundary_flows():
    """Test that the net inflow of people is births minus deaths."""
    boundary = conservation(System(components=[PopulationComponent('pop')]))
    assert str(boundary['people']) == 'b_pop*n_pop - d_pop*n_pop'


def test_monitor_detects_drift_beyond_the_flows():
    """Test that clipping a stock with a boundary law destroys water, which the monitor reports."""
    with pytest.raises(ConservationError, match="'water' drifted"):
        Simulation(System(components=[ClippedComponent('c')]), dt=1, observers=[ConservationObserver()]).run(5)


def test_member_invariant_violation_names_the_agent_the_invariant_and_the_change():
    """Test that lowering assets alone breaks the invariant of every member after the step."""
    with pytest.raises(ConservationError, match=r"'assets == liabilities \+ equity' of agent \d+ of banks.*-1"):
        Simulation(System(components=[BankComponent.group('banks', 1)])).run(2)


def test_bankrupt_firm_writes_off_its_debt_against_its_lender():
    """Test that removed firms settle their debt with transfers and total loans stay zero."""
    trajectory = Simulation(_credit_system(settles=True)).run(1).trajectory_
    assert trajectory.agents('lenders', 1).values['loans'].tolist() == [0.0]
    assert len(trajectory.agents('firms', 1).values['loans']) == 0


def test_removing_an_unsettled_agent_destroys_its_holdings():
    """Test that removing a firm that still owes raises a conservation error."""
    with pytest.raises(ConservationError, match='Settle its holdings'):
        Simulation(_credit_system(settles=False)).run(1)


def test_transfer_moves_between_states_and_checks_names():
    """Test that a transfer can move between differently named states and rejects missing ones."""
    lender, borrower = LenderComponent('l'), BorrowerComponent('b')
    transfer(lender, borrower, 3.0, 'money')
    assert (lender.money, borrower.money) == (-3.0, 3.0)
    transfer(borrower, lender, 1.0, 'money', 'loans')
    assert (borrower.money, lender.loans) == (2.0, 1.0)
    with pytest.raises(ValidationError, match='cash'):
        transfer(lender, borrower, 1.0, 'cash')


def test_member_invariant_rejects_lambdas_and_pickles_with_functions():
    """Test that invariants of members take module-level functions, which pickle."""
    with pytest.raises(ValidationError, match='cannot be pickled'):
        Invariant('lambda', lambda agent: 0.0)
    data = pickle.dumps(Invariant('equity', attrgetter('equity')))
    invariant = pickle.loads(data)  # noqa: S301  # nosec B301: loads only what it dumped
    assert invariant.label == 'equity'


@pytest.mark.parametrize(('creates', 'violated'), [(0.0, False), (0.5, True)])
def test_interaction_invariant_checks_the_total_of_its_agents(creates, violated):
    """Test that an invariant of an interaction holds under transfers and names the interaction when money appears."""
    system = System(
        components=[LenderComponent.group('holders', 3, variable=True, money=5.0)],
        interactions=[PaymentsInteraction('payments', holders='holders', creates=creates)],
    )
    if violated:
        with pytest.raises(ConservationError, match="'total money' of payments"):
            Simulation(system).run(3)
        return
    trajectory = Simulation(system).run(3).trajectory_
    assert trajectory.agents('holders', 3).values['money'].sum() == pytest.approx(15.0)
