"""Test the social science models: opinion dynamics, segregation, cooperation, spreading, crowds, and organizations."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.base import ValidationError
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.models.social.cooperation import (
    play,
    prisoners_dilemma_tournament,
    public_goods_game,
    spatial_prisoners_dilemma,
)
from skcomplex.models.social.crowds import social_force
from skcomplex.models.social.opinion_dynamics import (
    cognitive_consistency,
    deffuant,
    galam_majority,
    influentials,
    naming_game,
    settle,
    social_impact,
    sznajd,
    threshold_adoption,
    voter_variants,
)
from skcomplex.models.social.organizations import (
    basal_metabolic_rate,
    bonabeau_hierarchies,
    employment_takeover,
    insider_sabotage,
    obesity,
    priority_queues,
    trafficking,
)
from skcomplex.models.social.segregation import stochastic_schelling
from skcomplex.models.social.spreading import maki_thompson, simon_yule, watts_cascades


def _run(system, until, **settings):
    """Return the trajectory of a model run with a fixed random state."""
    return Simulation(system, random_state=0, **settings).run(until=until).trajectory_


def _values(trajectory, group, name, frame=-1):
    """Return the values of an agent state at a frame."""
    frame = len(trajectory.times) - 1 if frame == -1 else frame
    return trajectory.agents(group, frame).values[name]


def _clusters(opinions, gap):
    """Return the sizes of the groups of sorted opinions separated by more than a gap."""
    ordered = np.sort(opinions)
    labels = np.concatenate([[0], np.cumsum(np.diff(ordered) > gap)])
    return np.bincount(labels)


def _cooperation(trajectory):
    """Return the share of cooperators at every frame of a lattice game."""
    strategies = trajectory['players.strategy']
    return strategies.reshape(len(strategies), -1).mean(axis=1)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_deffuant_bounded_confidence_clusters(seed):
    """Test that Deffuant opinions with confidence 0.2 converge to about `1 / (2 * 0.2)` clusters."""
    trajectory = _run(deffuant(500, confidence=0.2, random_state=seed), 100)
    opinions = _values(trajectory, 'people', 'opinion')
    sizes = _clusters(opinions, 0.05)
    major = sizes[sizes >= 0.05 * len(opinions)]
    assert len(major) in {2, 3}
    assert 2 <= len(sizes) <= 4
    initial = _clusters(_values(trajectory, 'people', 'opinion', 0), 0.05)
    assert len(initial) == 1


def test_deffuant_conserves_mean_opinion():
    """Test that compromises keep the mean opinion of the Deffuant model."""
    trajectory = _run(deffuant(100, random_state=0), 20)
    first, last = _values(trajectory, 'people', 'opinion', 0), _values(trajectory, 'people', 'opinion')
    assert last.mean() == pytest.approx(first.mean())


def test_sznajd_reaches_consensus_on_a_ring():
    """Test that the Sznajd model on a small ring ends with one opinion."""
    trajectory = _run(sznajd(nx.cycle_graph(30), random_state=0), 300)
    assert len(set(_values(trajectory, 'people', 'opinion').tolist())) == 1


def test_galam_minority_vanishes():
    """Test that a minority of 40% disappears after a few rounds of discussion in groups of three."""
    trajectory = _run(galam_majority(300, share=0.4, random_state=0), 10)
    assert _values(trajectory, 'citizens', 'opinion').sum() == 0


def test_social_impact_freezes_with_surviving_minority():
    """Test that social impact settles into a static pattern where clusters of the minority survive."""
    trajectory = _run(social_impact(random_state=1), 10)
    last, previous = _values(trajectory, 'people', 'opinion'), _values(trajectory, 'people', 'opinion', 9)
    assert np.array_equal(last, previous)
    assert 0 < (last > 0).mean() < 1


def test_naming_game_agrees_on_one_word():
    """Test that the naming game ends with every agent knowing the same single word."""
    words = _values(_run(naming_game(30), 200), 'namers', 'words')
    assert words.sum(axis=1).tolist() == [1.0] * 30
    assert len(set(words.argmax(axis=1).tolist())) == 1


def test_influentials_add_their_weight():
    """Test that beliefs grow by eight neighbours per agent and influentials pull agents to caring passionately."""
    plain = _run(influentials(n_influentials=0, random_state=0), 20)
    total = [_values(plain, 'people', 'belief', frame).sum() for frame in (0, 1)]
    assert total[1] - total[0] == 8 * 400
    swayed = _run(influentials(n_influentials=40, random_state=0), 20)
    passionate = [(_values(trajectory, 'people', 'caring') == 3).mean() for trajectory in (plain, swayed)]
    assert passionate[1] > passionate[0]


@pytest.mark.parametrize(('positive', 'sign'), [(0.8, 1.0), (0.2, -1.0)])
def test_cognitive_consistency_agrees_with_the_majority(positive, sign):
    """Test that a complete graph agrees on an attitude whose sign is that of the initial majority."""
    trajectory = _run(cognitive_consistency(positive=positive, random_state=0), 60)
    valences = _values(trajectory, 'thinkers', 'valences')
    assert np.sign(valences[:, 1]).tolist() == [sign] * len(valences)
    assert valences.std(axis=0).max() < 0.05
    assert np.abs(valences).max() <= 1


def test_settling_lowers_the_energy():
    """Test that settling the belief network does not raise its energy and keeps valences in range."""
    coherence = np.array([[0.0, 0.05, 0.0], [0.05, 0.0, -0.05], [0.0, -0.05, 0.0]])
    start = np.array([0.0, 0.5, 0.5])
    settled = settle(start)
    assert -settled @ coherence @ settled <= -start @ coherence @ start
    assert np.abs(settled).max() <= 1


def test_threshold_adoption_by_infection_spreads_to_everyone():
    """Test that adoption by infection with a reach of 15 reaches everyone and never decreases."""
    trajectory = _run(threshold_adoption(random_state=0), 25)
    shares = [_values(trajectory, 'people', 'adopted', frame).mean() for frame in range(26)]
    assert shares[-1] == 1.0
    assert np.all(np.diff(shares) >= 0)


def test_threshold_adoption_rejects_unknown_mode():
    """Test that an unknown adoption mode raises a validation error when the model compiles."""
    with pytest.raises(ValidationError, match='Unknown adoption mode'):
        _run(threshold_adoption(40, size=60.0, mode='contagion'), 1)


def test_voter_variants_contrarians_prevent_consensus():
    """Test that majority voters reach consensus, and contrarians keep a mixed state."""
    graph = nx.complete_graph(30)
    majority = _run(voter_variants(graph, contrarians=0.0, random_state=0), 50)
    assert len(set(_values(majority, 'voters', 'opinion').tolist())) == 1
    mixed = _run(voter_variants(graph, contrarians=0.2, random_state=0), 50)
    assert len(set(_values(mixed, 'voters', 'opinion').tolist())) == 2


def _block_densities(trajectory):
    """Return the sorted densities of the 16 blocks of 25 cells of the stochastic Schelling model."""
    blocks = _values(trajectory, 'dwellers', 'block').astype(int)
    return np.bincount(blocks, minlength=16) / 25


def test_stochastic_schelling_segregates_above_critical_beta():
    """Test that block densities split towards 0.07 and 0.93 at `beta = 3`, and stay near 0.5 at `beta = 1`."""
    hot = _block_densities(_run(stochastic_schelling(beta=1.0, random_state=0), 100))
    cold = _block_densities(_run(stochastic_schelling(beta=3.0, random_state=0), 100))
    assert cold.std() > 2 * hot.std()
    assert np.abs(hot - 0.5).max() < 0.4
    assert cold.min() < 0.2 and cold.max() > 0.8  # noqa: PT018


def test_spatial_prisoners_dilemma_regimes():
    """Test that cooperators and defectors coexist at temptation 1.85 and defectors win at temptation 3."""
    coexisting = _cooperation(_run(spatial_prisoners_dilemma(temptation=1.85), 50, dt=1))
    assert coexisting[-10:].min() > 0.1 and coexisting[-10:].max() < 0.6  # noqa: PT018
    assert _cooperation(_run(spatial_prisoners_dilemma(temptation=3.0), 20, dt=1))[-1] == 0
    fermi = _run(spatial_prisoners_dilemma(temptation=1.05, imitation='fermi', share=0.5), 50, dt=1)
    assert _cooperation(fermi)[-1] > 0.9


def test_spatial_prisoners_dilemma_rejects_unknown_imitation():
    """Test that an unknown imitation rule raises a validation error when the model compiles."""
    with pytest.raises(ValidationError, match='Unknown imitation'):
        _run(spatial_prisoners_dilemma((10, 10), imitation='copy'), 1, dt=1)


def test_public_goods_game_depends_on_synergy():
    """Test that cooperation dies out at synergy 2 and takes over at synergy 6."""
    assert _cooperation(_run(public_goods_game(synergy=2.0), 50, dt=1))[-1] == 0
    assert _cooperation(_run(public_goods_game(synergy=6.0), 50, dt=1))[-1] > 0.9


def test_iterated_games_and_a_cooperative_population():
    """Test the payoffs of iterated games and that a population of cooperators keeps fitness 3 without mutation."""
    tit_for_tat = [1, 1, 0, 1, 0, 1, 0]
    assert play(tit_for_tat, [0] * 7) == (5.0, 10.0)
    assert play([1] * 7, [1] * 7) == (18.0, 18.0)
    system = prisoners_dilemma_tournament(20, mutation=0.0, random_state=0)
    system.elements[0].values['genome'] = np.ones((20, 7))
    assert set(_values(_run(system, 5), 'agents', 'fitness').tolist()) == {3.0}


def test_maki_thompson_leaves_a_fifth_ignorant():
    """Test that a rumour stops with about 20% of the population never hearing it."""
    finals = [
        Simulation(maki_thompson(1000), dt=0.5, random_state=seed).run(until=40).trajectory_['ignorants.n'][-1]
        for seed in range(5)
    ]
    assert np.mean(finals) / 1000 == pytest.approx(0.203, abs=0.04)


def test_watts_cascades_depend_on_threshold():
    """Test that one seed on a ring activates every node at threshold 0.5 and only itself at threshold 0.6."""
    spread = _run(watts_cascades(nx.cycle_graph(20), threshold=0.5, random_state=0), 30)
    assert _values(spread, 'nodes', 'active').sum() == 20
    stuck = _run(watts_cascades(nx.cycle_graph(20), threshold=0.6, random_state=0), 30)
    assert _values(stuck, 'nodes', 'active').sum() == 1


def test_simon_yule_grows_new_words_at_the_innovation_rate():
    """Test that the text has its expected length and about `innovation` times as many distinct words."""
    counts = _values(_run(simon_yule(innovation=0.1), 100), 'words', 'count')
    assert counts.sum() == 10_001
    assert len(counts) == pytest.approx(1000, rel=0.1)
    assert counts.max() > 100 * np.median(counts)


def test_social_force_walkers_reach_desired_speed_between_walls():
    """Test that both crowds walk at nearly their desired speed in opposite directions and stay between the walls."""
    trajectory = Simulation(social_force(), dt=0.2, solver=RK4(dt=0.05)).run(until=20).trajectory_
    assert trajectory['rightward.v'][-1][:, 0].mean() > 1.0
    assert trajectory['leftward.v'][-1][:, 0].mean() < -1.0
    heights = np.concatenate([trajectory['rightward.x'][:, :, 1], trajectory['leftward.x'][:, :, 1]])
    assert heights.min() > 0 and heights.max() < 10  # noqa: PT018


def test_bonabeau_hierarchy_emerges_at_high_density():
    """Test that strengths spread far more at a high density than at a low one."""
    sparse = _values(_run(bonabeau_hierarchies(density=0.05, random_state=0), 100), 'contenders', 'strength')
    dense = _values(_run(bonabeau_hierarchies(density=0.8, random_state=0), 100), 'contenders', 'strength')
    assert dense.std() > 3 * sparse.std()


def _waits(trajectory, frames):
    """Return the waiting times of the tasks executed in a run of the priority queue."""
    waits = []
    for frame in range(1, frames + 1):
        values = trajectory.agents('tasks', frame).values
        waits += values['wait'][values['age'] == 0].tolist()
    return np.array(waits)


def test_priority_queue_waits():
    """Test that random execution gives a mean wait of two for two tasks, and greedy execution gives long waits."""
    random = _waits(_run(priority_queues(greed=0.0, random_state=0), 2000), 2000)
    assert random.mean() == pytest.approx(2.0, rel=0.1)
    greedy = _waits(_run(priority_queues(random_state=0), 2000), 2000)
    assert np.median(greedy) == 1
    assert greedy.max() > 100


def test_employment_takeover_is_deterministic_and_settles():
    """Test that the employment model gives the same run twice and ends in a fixed point or a short cycle."""
    first = _run(employment_takeover(), 60, dt=1)['cells.colour']
    second = _run(employment_takeover(), 60, dt=1)['cells.colour']
    assert np.array_equal(first, second)
    assert any(np.array_equal(first[-1], first[-1 - period]) for period in (1, 2))


def test_insider_risk_falls_with_turnover():
    """Test that a higher turnover keeps the mean risk of sabotage lower after one year."""
    risks = [
        _values(_run(insider_sabotage(turnover=turnover, random_state=0), 365), 'employees', 'risk').mean()
        for turnover in (0.1, 3.0)
    ]
    assert risks[1] < risks[0]


def test_obesity_weight_follows_meals():
    """Test that people stay within the grid and that heavy eaters near every restaurant gain weight."""
    trajectory = _run(obesity(30, n_places=10, size=50.0, reach=100.0, random_state=0), 28)
    locations = trajectory.agents('people', 28).locations
    assert locations.min() >= 0 and locations.max() <= 50  # noqa: PT018
    habits = _values(trajectory, 'people', 'habit')
    gain = _values(trajectory, 'people', 'weight') - _values(trajectory, 'people', 'weight', 0)
    assert gain[habits == 2].mean() > gain[habits == 0].mean()
    assert basal_metabolic_rate(70.0, 175.0, 30.0, 0.0) < basal_metabolic_rate(70.0, 175.0, 30.0, 1.0)


def test_trafficking_victims_stay_in_brothels():
    """Test that victims are always in brothels and that brothels run higher bills than houses."""
    trajectory = _run(trafficking(random_state=0), 180)
    premises, women = trajectory.agents('premises', 180), trajectory.agents('women', 180)
    brothels = premises.values['brothel'] > 0
    cells = {tuple(location) for location in premises.locations[brothels].tolist()}
    victims = women.locations[women.values['victim'] > 0].tolist()
    assert all(tuple(location) in cells for location in victims)
    assert premises.values['last_bills'][brothels].mean() > premises.values['last_bills'][~brothels].mean()
