"""Test the `_imitation` module of `skcomplex.systems.general.agents`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from scipy.special import expit

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime, LatticeSpace, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import LayerSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.systems.general.agents import ImitationInteraction

SIZE = 50
RUNS = 200
FRACTION = 0.3
DISCRETE_SIZE = 20
DISCRETE_RUNS = 100
SEED = 7


class PlayerElement(Element):
    """Player with a discrete strategy and a payoff."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        state: Annotated[object, StateSpec()] = 0.0,
        payoff: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.state = state
        self.payoff = payoff


class PeerImitationInteraction(ImitationInteraction):
    """Imitation that binds its players as `peers` and calls its probability of imitation `openness`."""

    def __init__(
        self,
        label: str,
        *,
        peers: Annotated[str, RoleSpec()],
        openness: Annotated[float, ParameterSpec()] = 1.0,
        **arguments: object,
    ) -> None:
        super().__init__(label, target=peers, **arguments)
        self.peers = peers
        self.openness = openness

    def _get_role(self, role: str) -> str:
        """Return `peers` for the target."""
        return 'peers' if role == 'target' else role

    def _get_parameter(self, name: str) -> str:
        """Return `openness` for the probability of imitation."""
        return 'openness' if name == 'probability' else name


def _grid_players() -> GroupElement:
    """Return players on a 3 by 3 grid whose middle player holds option 1 with the largest payoff."""
    cells = [(x, y) for x in range(3) for y in range(3)]
    payoffs = np.array([4.0 if cell == (1, 1) else 0.0 for cell in cells])
    values = {'state': (payoffs > 0).astype(float), 'payoff': payoffs}
    return GroupElement('players', element=PlayerElement, size=len(cells), values=values, location=cells)


def _final(space, values, rule, until, seed=0, **arguments):
    """Return the strategies after imitation by a rule for some time, in continuous or in discrete time."""
    discrete = arguments.get('form') == 'probability'
    system = CompositeSystem(
        elements=[GroupElement('players', element=PlayerElement, space=space, values=values, on='sites')],
        interactions=[ImitationInteraction('imitation', target='players', rule=rule, **arguments)],
        time=DiscreteTime() if discrete else ContinuousTime(),
    )
    return simulate(system, until=until, dt=1.0 if discrete else until, random_state=seed).values('players.state')[-1]


def _alternating(size):
    """Return strategies and payoffs alternating between option 0 of payoff 0 and option 1 of payoff 1."""
    values = (np.arange(size) % 2).astype(float)
    return {'state': values, 'payoff': values}


def test_voters_on_a_ring_reach_consensus_with_the_initial_fraction_as_probability():
    """Test that voters on a ring of 50 always reach consensus, winning with the probability of their fraction."""
    initial = (np.arange(SIZE) < FRACTION * SIZE).astype(float)
    finals = [_final(LatticeSpace(SIZE), {'state': initial}, 'random', 5000.0, seed) for seed in range(RUNS)]
    assert all(np.ptp(final) == 0.0 for final in finals)
    wins = np.mean([final[0] for final in finals])
    assert wins == pytest.approx(FRACTION, abs=3 * np.sqrt(FRACTION * (1 - FRACTION) / RUNS))


def test_majority_flips_a_member_whose_neighbours_agree():
    """Test that a member whose two neighbours on a ring both hold the other option takes it, and blocks stay."""
    initial = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    final = _final(LatticeSpace(12), {'state': initial}, 'majority', 50.0)
    assert final.tolist() == [1.0] * 7 + [0.0] * 5


def test_best_copies_the_neighbour_of_largest_payoff_up_to_a_local_maximum():
    """Test that every member ends with the strategy of the local payoff maximum it climbs to through its neighbours."""
    graph = nx.gnp_random_graph(12, 0.3, seed=5)
    rng = np.random.default_rng(4)
    strategies, payoffs = rng.integers(0, 2, 12).astype(float), rng.normal(size=12)
    final = _final(NetworkSpace(graph), {'state': strategies, 'payoff': payoffs}, 'best', 200.0)

    def summit(node):
        """Return the node reached by moving to the neighbour of largest payoff while it pays more."""
        best = max(graph.neighbors(node), key=lambda other: payoffs[other], default=node)
        return summit(best) if payoffs[best] > payoffs[node] else node

    assert final.tolist() == [strategies[summit(node)] for node in graph]


def test_voters_in_discrete_time_reach_consensus_with_the_initial_fraction_as_probability():
    """Test that voters on a ring of 20 imitating with probability one half per step reach consensus."""
    initial = (np.arange(DISCRETE_SIZE) < FRACTION * DISCRETE_SIZE).astype(float)
    finals = [
        _final(
            LatticeSpace(DISCRETE_SIZE),
            {'state': initial},
            'random',
            2000.0,
            seed,
            form='probability',
            probability=0.5,
        )
        for seed in range(DISCRETE_RUNS)
    ]
    assert all(np.ptp(final) == 0.0 for final in finals)
    wins = np.mean([final[0] for final in finals])
    assert wins == pytest.approx(FRACTION, abs=3 * np.sqrt(FRACTION * (1 - FRACTION) / DISCRETE_RUNS))


def test_fermi_copies_with_the_probability_of_the_payoff_difference():
    """Test that in one step a member copies a neighbour with probability `p / (1 + exp(-beta (P_j - P_i)))`."""
    values = _alternating(20000)
    final = _final(LatticeSpace(20000), values, 'fermi', 1.0, form='probability', probability=0.8, intensity=1.0)
    switched = final != values['state']
    assert switched[::2].mean() == pytest.approx(0.8 * expit(1.0), abs=0.015)
    assert switched[1::2].mean() == pytest.approx(0.8 * expit(-1.0), abs=0.015)


def _fermi_switching(strategies, final):
    """Return the switching frequency and the exact Fermi probability of the members of each state on a torus."""
    differing = sum(np.roll(strategies, shift, axis) != strategies for shift in (1, -1) for axis in (0, 1))
    switched = final != strategies
    exact = differing / 4 * expit(np.where(strategies == 0, 1.0, -1.0))
    return [(switched[strategies == own].mean(), exact[strategies == own].mean()) for own in (0.0, 1.0)]


def test_fermi_copies_a_random_neighbour_with_the_fermi_probability_from_either_state():
    """Test that in one step members of both states on a mixed lattice switch with the exact Fermi probability."""
    strategies = (np.random.default_rng(SEED).random((150, 150)) < 0.5).astype(float)
    system = CompositeSystem(
        elements=[
            GroupElement(
                'players',
                element=PlayerElement,
                space=LatticeSpace((150, 150), metric='manhattan'),
                values={'state': strategies.ravel(), 'payoff': strategies.ravel()},
                on='sites',
            ),
        ],
        interactions=[ImitationInteraction('imitation', target='players', rule='fermi', form='probability')],
        time=DiscreteTime(),
    )
    trajectory = simulate(system, until=1.0, dt=1.0, random_state=SEED)
    final = trajectory.values('players.state')[-1].reshape(strategies.shape)
    for switched, exact in _fermi_switching(strategies, final):
        assert switched == pytest.approx(exact, abs=0.01)


def test_agents_copy_a_random_neighbour_with_the_fermi_probability_from_either_state():
    """Test that in one step agents of both states on a mixed grid switch with the exact Fermi probability."""
    size = 60
    strategies = (np.random.default_rng(SEED).random((size, size)) < 0.5).astype(float)
    cells = [(x, y) for x in range(size) for y in range(size)]
    group = GroupElement(
        'players',
        element=PlayerElement,
        size=len(cells),
        values={'state': strategies.ravel(), 'payoff': strategies.ravel()},
        location=cells,
    )
    system = CompositeSystem(
        elements=[group],
        interactions=[ImitationInteraction('imitation', target='players', rule='fermi', form='agent')],
        space=LatticeSpace((size, size), boundary='periodic', metric='manhattan'),
        time=DiscreteTime(),
    )
    trajectory = simulate(system, until=1, random_state=SEED)
    final = np.asarray(trajectory.agents('players', 1).values['state'], dtype=float).reshape(strategies.shape)
    for switched, exact in _fermi_switching(strategies, final):
        assert switched == pytest.approx(exact, abs=0.03)


def test_fermi_copies_at_the_rate_times_the_fermi_probability():
    """Test that in a short time a member copies a neighbour at the rate times its Fermi probability."""
    values = _alternating(20000)
    final = _final(LatticeSpace(20000), values, 'fermi', 0.1, intensity=1.0)
    switched = final != values['state']
    assert switched[::2].mean() == pytest.approx(1 - np.exp(-0.1 * expit(1.0)), abs=0.01)
    assert switched[1::2].mean() == pytest.approx(1 - np.exp(-0.1 * expit(-1.0)), abs=0.01)


def test_majority_in_discrete_time_flips_a_member_whose_neighbours_agree():
    """Test that in discrete time a member whose two neighbours on a ring hold the other option takes it."""
    initial = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    final = _final(LatticeSpace(12), {'state': initial}, 'majority', 3.0, form='probability')
    assert final.tolist() == [1.0] * 7 + [0.0] * 5


def test_best_in_discrete_time_climbs_to_the_local_payoff_maximum():
    """Test that in discrete time every member ends with the strategy of the payoff maximum it climbs to."""
    payoffs = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 5.0])
    strategies = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    final = _final(LatticeSpace(8), {'state': strategies, 'payoff': payoffs}, 'best', 10.0, form='probability')
    assert final.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize('rule', ['best', 'better'])
def test_agents_copy_the_neighbour_of_larger_payoff_in_the_agent_form(rule):
    """Test that agents on a grid copy only a neighbour of larger payoff, so the option spreads to its neighbours."""
    cells = [(x, y) for x in range(3) for y in range(3)]
    payoffs = np.array([4.0 if cell == (1, 1) else 0.0 for cell in cells])
    group = GroupElement(
        'players',
        element=PlayerElement,
        size=len(cells),
        values={'state': (payoffs > 0).astype(float), 'payoff': payoffs},
        location=cells,
    )
    imitation = ImitationInteraction('imitation', target='players', rule=rule, form='agent')
    system = CompositeSystem(
        elements=[group],
        interactions=[imitation],
        space=LatticeSpace((3, 3), boundary='periodic', metric='manhattan'),
        time=DiscreteTime(),
    )
    trajectory = simulate(system, until=30, random_state=0)
    final = dict(zip(cells, trajectory.agents('players', 30).values['state'], strict=True))
    assert {cell for cell, state in final.items() if state} == {(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)}


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'rule': 'copy'}, 'Unknown rule'),
        ({'form': 'events'}, 'form of imitation'),
        ({'options': (1.0, 1.0)}, 'two distinct'),
        ({'rule': 'better'}, 'agent form only'),
        ({'schedule': 'member'}, 'only the agent form'),
        ({'ties': 'random'}, 'only the agent form'),
        ({'ties': 'nearest', 'form': 'agent'}, 'Unknown ties'),
        ({'schedule': 'parallel', 'form': 'agent'}, 'Unknown schedule'),
    ],
)
def test_check_rejects_unknown_rules_and_equal_options(arguments, match):
    """Test that an unknown rule or options that are not distinct raise an error."""
    with pytest.raises(ValidationError, match=match):
        ImitationInteraction('imitation', target='players', **arguments).prepare()


@pytest.mark.parametrize(('rule', 'form'), [('random', 'rate'), ('best', 'rate'), ('fermi', 'probability')])
def test_get_laws_reads_a_renamed_target_and_probability(rule, form):
    """Test that an imitation that binds `peers` and calls its probability `openness` gives the plain trajectory."""
    discrete = form == 'probability'
    finals = []
    for imitation in (
        ImitationInteraction('imitation', target='players', rule=rule, form=form, probability=0.5),
        PeerImitationInteraction('imitation', peers='players', rule=rule, form=form, openness=0.5),
    ):
        system = CompositeSystem(
            elements=[
                GroupElement(
                    'players',
                    element=PlayerElement,
                    space=LatticeSpace(10),
                    values=_alternating(10),
                    on='sites',
                ),
            ],
            interactions=[imitation],
            time=DiscreteTime() if discrete else ContinuousTime(),
        )
        trajectory = simulate(system, until=5, dt=1.0, random_state=SEED)
        finals.append(trajectory.values('players.state'))
    np.testing.assert_array_equal(finals[1], finals[0])


@pytest.mark.parametrize('rule', ['random', 'best', 'fermi'])
def test_imitate_reads_a_renamed_target_and_probability_in_the_agent_form(rule):
    """Test that agent imitation that binds `peers` and names its probability `openness` runs as the plain one."""
    space = LatticeSpace((3, 3), boundary='periodic', metric='manhattan')
    finals = []
    for imitation in (
        ImitationInteraction('imitation', target='players', rule=rule, form='agent', probability=0.5),
        PeerImitationInteraction('imitation', peers='players', rule=rule, form='agent', openness=0.5),
    ):
        system = CompositeSystem(elements=[_grid_players()], interactions=[imitation], space=space, time=DiscreteTime())
        trajectory = simulate(system, until=10, random_state=SEED)
        finals.append(trajectory.agents('players', 10).values['state'])
    np.testing.assert_array_equal(finals[1], finals[0])


def _chain(schedule: str, seed: int, ties: str = 'first', payoffs: tuple[float, ...] = (3.0, 2.0, 1.0)) -> list[float]:
    """Return the states after one step of best imitation of agents on a chain, the first holding option 1."""
    group = GroupElement(
        'players',
        element=PlayerElement,
        size=len(payoffs),
        values={'state': [1.0] + [0.0] * (len(payoffs) - 1), 'payoff': list(payoffs)},
        location=[(cell,) for cell in range(len(payoffs))],
    )
    imitation = ImitationInteraction(
        'imitation',
        target='players',
        rule='best',
        form='agent',
        schedule=schedule,
        ties=ties,
    )
    space = LatticeSpace((len(payoffs),), boundary='fixed')
    system = CompositeSystem(elements=[group], interactions=[imitation], space=space, time=DiscreteTime())
    trajectory = simulate(system, until=1, random_state=seed)
    return trajectory.agents('players', 1).values['state'].tolist()


def test_agent_imitation_sequential_and_member_schedules_read_the_states_set_before():
    """Test that sequential and member imitation read the states earlier agents set, while synchronous does not."""
    assert _chain('synchronous', 0) == [1.0, 1.0, 0.0]
    assert _chain('sequential', 0) == [1.0, 1.0, 1.0]
    members = {tuple(_chain('member', seed)) for seed in range(20)}
    assert members == {(1.0, 1.0, 0.0), (1.0, 1.0, 1.0)}


def test_agent_imitation_best_rule_draws_tied_neighbours_by_the_random_ties():
    """Test that best imitation copies one fixed neighbour among ties by the first ties, and either by random ties."""
    payoffs = (3.0, 1.0, 3.0)
    assert len({_chain('synchronous', seed, 'first', payoffs)[1] for seed in range(20)}) == 1
    copies = [_chain('synchronous', seed, 'random', payoffs)[1] for seed in range(200)]
    assert np.mean(copies) == pytest.approx(0.5, abs=0.1)


def test_agent_imitation_reads_the_neighbours_of_its_layer():
    """Test that agent imitation on a layer copies the neighbours of that layer, not those of the network graph."""
    work = nx.Graph([(0, 2)])
    work.add_node(1)
    group = GroupElement(
        'players',
        element=PlayerElement,
        size=3,
        values={'state': [1.0, 0.0, 0.0]},
        location=[0, 1, 2],
    )
    imitation = ImitationInteraction('imitation', target='players', form='agent', layer='work')
    space = NetworkSpace(nx.path_graph(3), layers=[LayerSpec('work', work)])
    system = CompositeSystem(elements=[group], interactions=[imitation], space=space, time=DiscreteTime())
    trajectory = simulate(system, until=1, random_state=0)
    assert trajectory.agents('players', 1).values['state'].tolist() == [0.0, 0.0, 1.0]
