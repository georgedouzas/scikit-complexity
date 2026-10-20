"""Test the ecology, epidemiology, and evolution models of populations."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.model import System
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models import catalog, describe, run
from skcomplex.models.biology.ecology import (
    agent_based_lotka_volterra,
    competition,
    food_webs,
    lotka_volterra,
    lotka_volterra_with_harvesting,
    mangrove_succession,
    meadow_capacity,
    mutualism,
    neutral_macroecology,
    nicholson_bailey,
    rosenzweig_macarthur,
    stigmergic_predator_prey,
    stochastic_predator_prey,
    varying_capacity_logistic,
)
from skcomplex.models.biology.epidemiology import (
    agent_based_sir,
    delayed_sirs,
    heterogeneous_mean_field_sis,
    metapopulation,
    network_sir,
    random_walker_epidemics,
    sirs,
    sis,
)
from skcomplex.models.biology.evolution import (
    adaptive_climbing,
    eigen_quasispecies,
    hypercycles,
    moran_process,
    nk_landscape,
    replicator_dynamics,
    rock_paper_scissors,
    tangled_nature,
)

PRECISE = ScipyODE(method='DOP853', rtol=1e-12, atol=1e-14)
ECOLOGY = {
    'competition',
    'mutualism',
    'food_webs',
    'rosenzweig_macarthur',
    'nicholson_bailey',
    'lotka_volterra_with_harvesting',
    'varying_capacity_logistic',
    'mangrove_succession',
    'agent_based_lotka_volterra',
    'stigmergic_predator_prey',
    'meadow_capacity',
    'neutral_macroecology',
    'stochastic_predator_prey',
}
EPIDEMIOLOGY = {
    'sis',
    'sirs',
    'agent_based_sir',
    'random_walker_epidemics',
    'network_sir',
    'heterogeneous_mean_field_sis',
    'metapopulation',
    'delayed_sirs',
}
EVOLUTION = {
    'replicator_dynamics',
    'moran_process',
    'eigen_quasispecies',
    'hypercycles',
    'nk_landscape',
    'tangled_nature',
    'adaptive_climbing',
    'rock_paper_scissors',
}


def _run(system, until, **settings):
    """Return the trajectory of a system run until a time."""
    return Simulation(system, **settings).run(until=until).trajectory_


def _health(trajectory, group, frame):
    """Return the health of the agents of a group at a frame."""
    return trajectory.agents(group, frame).values['health']


@pytest.mark.parametrize(
    ('field', 'names'),
    [('biology/ecology', ECOLOGY), ('biology/epidemiology', EPIDEMIOLOGY), ('biology/evolution', EVOLUTION)],
)
def test_catalog_lists_subfield(field, names):
    """Test that the catalog lists the models of a subfield with descriptions and references."""
    entries = {entry.name: entry for entry in catalog(field=field)}
    assert names <= set(entries)
    assert all(entries[name].description and entries[name].references for name in names)


def test_lotka_volterra_by_name_takes_predation_rate():
    """Test that the Lotka-Volterra model created with a predation rate oscillates around its center."""
    trajectory = _run(lotka_volterra(a=0.2), 50, dt=0.1)
    assert trajectory['prey.n'].min() < 0.4 / 0.1 < trajectory['prey.n'].max()


def test_competition_settles_at_coexistence():
    """Test that weak competitors coexist at the equilibrium of the competition model."""
    trajectory = _run(competition(), 100, dt=1.0)
    assert trajectory['first.n'][-1] == pytest.approx((100 - 0.5 * 80) / (1 - 0.3), rel=1e-3)
    assert trajectory['second.n'][-1] == pytest.approx((80 - 0.6 * 100) / (1 - 0.3), rel=1e-3)


def test_competition_excludes_weak_species():
    """Test that strong competition lets one species exclude the other."""
    trajectory = _run(competition(alpha12=0.5, alpha21=1.5), 200, dt=1.0)
    assert trajectory['second.n'][-1] < 1e-3
    assert trajectory['first.n'][-1] == pytest.approx(100, rel=1e-3)


def test_mutualism_with_handling_is_bounded_above_capacities():
    """Test that mutualism with a handling time settles above the capacities, and without it explodes."""
    bounded = _run(mutualism(), 100, dt=1.0)
    assert 50 < bounded['first.n'][-1] < 100
    assert bounded['first.n'][-1] == pytest.approx(bounded['first.n'][-2], rel=1e-4)
    explosive = _run(mutualism(handling=0.0, a=0.1, b=0.3), 2, dt=1.0)
    assert explosive['first.n'][-1] > 150


def test_food_web_species_coexist():
    """Test that the four species of the default food web coexist at a stable equilibrium."""
    trajectory = _run(food_webs(), 400, dt=1.0)
    final = [trajectory[f'{name}.n'][-1] for name in ('frogs', 'mice', 'snakes', 'hawks')]
    assert min(final) > 10
    assert trajectory['snakes.n'][-1] == pytest.approx(trajectory['snakes.n'][-20], rel=1e-3)


def test_rosenzweig_macarthur_tips_only_at_fast_rates():
    """Test that the prey track the coexistence point under a slow rise of crowding and collapse under a fast one."""
    solver = describe('rosenzweig_macarthur').defaults['solver']
    slow = _run(rosenzweig_macarthur(rate=1e-4), 100, dt=0.5, solver=solver)
    fast = _run(rosenzweig_macarthur(rate=0.004), 100, dt=0.5, solver=solver)
    assert slow['prey.n'].min() > 4.5
    assert fast['prey.n'].min() < 0.1


def test_rosenzweig_macarthur_prey_return_after_a_deep_collapse():
    """Test that the registered run follows the prey through a collapse far below any absolute tolerance and back."""
    trajectory = run('rosenzweig_macarthur').trajectory_
    reference = _run(
        rosenzweig_macarthur(),
        describe('rosenzweig_macarthur').defaults['until'],
        dt=describe('rosenzweig_macarthur').defaults['dt'],
        solver=ScipyODE(method='DOP853', rtol=1e-12, atol=1e-300),
    )
    assert trajectory['prey.n'].min() < 1e-20
    assert trajectory['prey.n'][-1] > 4.0
    np.testing.assert_allclose(trajectory['prey.n'], reference['prey.n'], rtol=1e-3, atol=1e-3)


def test_nicholson_bailey_equilibrium_is_unstable():
    """Test that the Nicholson-Bailey equilibrium is a fixed point that perturbations move away from."""
    k, a, c = 2.0, 0.05, 1.0
    hosts, parasitoids = k * np.log(k) / ((k - 1) * a * c), np.log(k) / a
    fixed = _run(nicholson_bailey(hosts, parasitoids, k=k, a=a, c=c), 10, dt=1)
    assert fixed['hosts.n'][-1] == pytest.approx(hosts)
    perturbed = _run(nicholson_bailey(hosts * 1.05, parasitoids, k=k, a=a, c=c), 20, dt=1)
    deviations = np.abs(perturbed['hosts.n'] - hosts)
    assert deviations[10:].max() > 3 * deviations[0]


def test_harvesting_averages_match_shifted_equilibrium():
    """Test Volterra's principle: the cycle averages of harvested Lotka-Volterra are its shifted equilibrium."""
    trajectory = _run(lotka_volterra_with_harvesting(), 100, dt=0.05)
    assert trajectory['prey.n'].mean() == pytest.approx((0.5 + 0.4) / 0.05, rel=0.03)
    assert trajectory['predators.n'].mean() == pytest.approx((1.0 - 0.4) / 0.1, rel=0.03)


def test_varying_capacity_settles_on_cycle_of_the_season():
    """Test that logistic growth under a periodic capacity settles on a cycle of the same period."""
    trajectory = _run(varying_capacity_logistic(), 20, dt=0.05)
    population = trajectory['population.n']
    assert population[-1] == pytest.approx(population[-21], rel=1e-3)
    assert population[-20:].max() - population[-20:].min() > 10


def test_mangrove_pioneers_lead_and_climax_dominates():
    """Test that the pioneer Keora leads early and the climax Sundari dominates the mature forest."""
    trajectory = _run(mangrove_succession(), 400, dt=1.0)
    early, late = 10, -1
    assert trajectory['keora.volume'][early] > trajectory['sundari.volume'][early]
    assert trajectory['sundari.volume'][late] > max(trajectory['keora.volume'][late], trajectory['gewa.volume'][late])
    assert min(trajectory[f'{name}.volume'].min() for name in ('keora', 'gewa', 'sundari')) >= -1e-8


def test_neutral_abundances_settle_at_mean():
    """Test that neutral birth, death, and immigration settle at the mean abundance `b0 / (d1 - b1)`."""
    trajectory = _run(neutral_macroecology(50), 60, dt=10.0, random_state=0)
    assert trajectory['species.n'][-3:].mean() == pytest.approx(10, rel=0.2)


def test_stochastic_predator_prey_fluctuates_around_deterministic_equilibrium():
    """Test that demographic noise keeps integer populations that fluctuate around the deterministic equilibrium."""
    trajectory = _run(stochastic_predator_prey(), 100, dt=1.0, random_state=0)
    prey = trajectory['prey.n']
    assert np.array_equal(prey, np.round(prey))
    assert prey[30:].mean() == pytest.approx(0.3 / (0.005 * (0.5 - 0.15)), rel=0.25)


def test_agent_based_lotka_volterra_cycles():
    """Test that agent-based prey and predators persist and cycle."""
    trajectory = _run(agent_based_lotka_volterra(random_state=0), 100, random_state=0)
    prey = trajectory['prey.count']
    assert trajectory['predators.count'].min() > 0
    assert prey.max() > 2 * prey[30:].min()


def test_stigmergic_marks_follow_emotions():
    """Test that only animals with emotions mark the ground."""
    marks = {}
    for emotions in (True, False):
        system = stigmergic_predator_prey(20, 20, plants=100, prey=40, predators=10, emotions=emotions, random_state=0)
        trajectory = _run(system, 10, random_state=0)
        marks[emotions] = trajectory['ground/marks.prey_happiness'][-1].sum()
        assert trajectory['life/prey.count'][-1] > 0
    assert marks[True] > 0
    assert marks[False] == 0


@pytest.mark.parametrize('growth', [0.0, 0.2])
def test_meadow_runs_nested_in_a_farm(growth):
    """Test that the meadow placed as the subsystem `farm` runs as the meadow at the root."""
    root = _run(meadow_capacity(growth=growth, random_state=0), 25, random_state=0)
    nested = meadow_capacity(growth=growth, random_state=0).set_params(label='farm')
    farm = _run(System(systems=[nested]), 25, random_state=0)
    assert farm['farm/cows.count'].tolist() == root['cows.count'].tolist()
    assert {interaction.pasture for interaction in nested.interactions} == {'pasture'}


def test_meadow_regrowth_raises_carrying_capacity():
    """Test that a meadow without regrowth starves its cows within the summer, and regrowth feeds them."""
    barren = _run(meadow_capacity(cows=80, growth=0.0, random_state=0), 25, random_state=0)
    growing = _run(meadow_capacity(cows=80, growth=0.2, random_state=0), 25, random_state=0)
    assert barren['cows.count'][-1] == 0
    assert growing['cows.count'][-1] == 80


@pytest.mark.parametrize('stochastic', [False, True])
def test_sis_settles_at_endemic_equilibrium(stochastic):
    """Test that the SIS model settles at `I = N - gamma / beta`, in mean for random events."""
    trajectory = _run(sis(stochastic=stochastic), 300, dt=1.0, random_state=0)
    assert trajectory['I.n'][100:].mean() == pytest.approx(1000 - 0.1 / 0.0003, rel=0.05)
    assert trajectory['S.n'][-1] + trajectory['I.n'][-1] == pytest.approx(1000)


def test_sirs_settles_at_endemic_equilibrium():
    """Test that the SIRS model settles at `S = gamma / beta` and `I = xi (N - S) / (gamma + xi)`."""
    trajectory = _run(sirs(), 1000, dt=1.0)
    susceptible = 0.1 / 0.0003
    assert trajectory['S.n'][-1] == pytest.approx(susceptible, rel=1e-3)
    assert trajectory['I.n'][-1] == pytest.approx(0.02 * (1000 - susceptible) / 0.12, rel=1e-3)


def test_delayed_sirs_without_immunity_is_the_logistic_map():
    """Test that the discrete SIRS model without immunity and visits reduces to the logistic map."""
    trajectory = _run(delayed_sirs(0.1, 0.0, a=2.5, recovery=0, e=0.0), 60, dt=1)
    assert trajectory['first.x'][-1] == pytest.approx(1 - 1 / 2.5, rel=1e-6)
    assert trajectory['second.x'][-1] == 0


def test_delayed_sirs_outbreaks_recur_and_spread():
    """Test that outbreaks recur in the first city and spread to the second through visits."""
    trajectory = _run(delayed_sirs(), 200, dt=1)
    first = trajectory['first.x']
    assert trajectory['second.x'].max() > 0.1
    assert np.sum((first[1:-1] > first[:-2]) & (first[1:-1] > first[2:]) & (first[1:-1] > 0.1)) >= 3


def test_heterogeneous_mean_field_threshold():
    """Test that the degree classes are group members and that SIS dies out below `<k> / <k^2>` and persists above."""
    degrees = np.arange(3, 31, dtype=float)
    weights = degrees**-2.5 / np.sum(degrees**-2.5)
    threshold = (degrees @ weights) / (degrees**2 @ weights)
    classes = heterogeneous_mean_field_sis(lam=0.5 * threshold).elements[0]
    assert classes.size == degrees.size
    np.testing.assert_allclose(classes.values['k'], degrees)
    np.testing.assert_allclose(classes.values['P'], weights)
    below = _run(heterogeneous_mean_field_sis(lam=0.5 * threshold), 100, dt=1.0)
    above = _run(heterogeneous_mean_field_sis(lam=2 * threshold), 100, dt=1.0)
    assert below['network.rho'][-1].max() < 1e-6
    assert above['network.rho'][-1].min() > 1e-3
    assert np.all(np.diff(above['network.rho'][-1]) > 0)


def test_metapopulation_conserves_individuals_and_spreads():
    """Test that travellers conserve the total population and carry the epidemic to every node."""
    trajectory = _run(metapopulation(nx.path_graph(5), D=0.05), 200, dt=1.0)
    totals = trajectory['cities.S'] + trajectory['cities.I'] + trajectory['cities.R']
    assert totals.sum(axis=1) == pytest.approx(np.full(len(totals), 5000.0))
    assert trajectory['cities.R'][-1].min() > 100


def test_agent_based_sir_spreads_more_with_higher_infection():
    """Test that the final size of the agent-based SIR model grows with the infection probability."""
    sizes = []
    for beta in (0.05, 0.8):
        system = agent_based_sir(30, 30, people=300, infected=5, beta=beta, gamma=0.05, random_state=0)
        trajectory = _run(system, 200, random_state=0)
        sizes.append(np.sum(_health(trajectory, 'people', 200) == 3))
    assert sizes[1] > 2 * sizes[0]


def test_agent_based_sir_extensions():
    """Test that mortality kills, incubation delays infectiousness, and immunity wanes."""
    extended = {'mu': 0.05, 'alpha': 0.05, 'incubation': 3}
    system = agent_based_sir(20, 20, people=200, infected=10, beta=0.8, random_state=0, **extended)
    trajectory = _run(system, 60, random_state=0)
    health = np.concatenate([_health(trajectory, 'people', frame) for frame in range(0, 61, 3)])
    assert {1.0, 2.0, 3.0, 4.0} <= set(np.unique(health).tolist())
    locked = agent_based_sir(20, 20, people=200, lockdown_length=50, lockdown_d=0.0, random_state=0)
    still = _run(locked, 5, random_state=0)
    assert np.array_equal(still.agents('people', 5).locations, still.agents('people', 0).locations)


def test_random_walkers_die_after_infection():
    """Test that sick walkers infect others and fall dead, on a grid and on a network."""
    grid = _run(random_walker_epidemics(32, 32, walkers=400, random_state=0), 100, random_state=0)
    assert np.sum(_health(grid, 'walkers', 100) == 2) > 1
    world = nx.watts_strogatz_graph(30, 4, 0.1, seed=0)
    system = random_walker_epidemics(walkers=300, graph=world, infection=0.5, random_state=0)
    network = _run(system, 60, random_state=0)
    assert np.sum(_health(network, 'walkers', 60) > 0) > 10


def test_network_sir_spreads_between_cities():
    """Test that travellers carry the infection to several cities, where the infected recover."""
    trajectory = _run(network_sir(tau=0.1, m=0.1, random_state=0), 150, random_state=0)
    final = trajectory.agents('travellers', 150)
    recovered = final.values['health'] == 3
    assert recovered.sum() > 50
    assert len({final.locations[index] for index in np.flatnonzero(recovered)}) > 5


def test_hawks_and_doves_settle_at_share_v_over_c():
    """Test that hawks settle at the share `V / C` when fights cost more than the resource is worth."""
    trajectory = _run(replicator_dynamics(V=2.0, C=3.0), 50, dt=1.0, solver=PRECISE)
    assert trajectory['population.x'][-1][0] == pytest.approx(2 / 3, abs=1e-4)
    assert trajectory['population.x'][-1].sum() == pytest.approx(1)


def test_replicator_rock_paper_scissors_conserves_product():
    """Test that rock, paper, and scissors, three members of a group, keep the simplex and the product of the shares."""
    game = ((0, -1, 1), (1, 0, -1), (-1, 1, 0))
    system = replicator_dynamics([0.5, 0.3, 0.2], payoff=game)
    assert system.elements[0].size == 3
    trajectory = _run(system, 100, dt=0.5, solver=PRECISE)
    shares = trajectory['population.x']
    assert np.abs(shares.sum(axis=1) - 1).max() < 1e-10
    assert np.abs(np.prod(shares, axis=1) - 0.03).max() < 1e-6


def test_rock_paper_scissors_lattice_keeps_three_strategies():
    """Test that cyclic invasion on a lattice keeps the three strategies at about a third each."""
    trajectory = _run(rock_paper_scissors((40, 40)), 30, dt=1, random_state=0)
    final = trajectory['lattice.strategy'][-1]
    shares = np.bincount(final.astype(int), minlength=3) / final.size
    assert shares.min() > 0.1
    assert not np.array_equal(final, trajectory['lattice.strategy'][0])


@pytest.mark.parametrize(('u', 'master'), [(0.3, 0.4), (0.6, 0.0)])
def test_quasispecies_error_threshold(u, master):
    """Test that the master sequence keeps the share `1 - u / sigma` below the error threshold and is lost above."""
    trajectory = _run(eigen_quasispecies(u=u, sigma=0.5), 200, dt=1)
    assert trajectory['master.x'][-1] == pytest.approx(master, abs=1e-3)
    total = sum(trajectory[f'{label}.x'][-1] for label in ['master', *(f'k{k}' for k in range(1, 13))])
    assert total == pytest.approx(1)


def test_hypercycle_keeps_all_and_network_selects():
    """Test that a hypercycle of 4 molecules keeps the simplex and every molecule, and a complete network selects."""
    system = hypercycles(4)
    assert system.elements[0].size == 4
    cycle = _run(system, 1500, dt=10.0, solver=PRECISE)
    concentrations = cycle['soup.x']
    assert np.abs(concentrations.sum(axis=1) - 1).max() < 1e-10
    assert concentrations[-1].min() > 0.01
    assert np.abs(concentrations[-1] - concentrations[-2]).max() < 1e-4
    complete = _run(hypercycles(20, alpha=1.0, omega=50.0, topology='complete'), 50, dt=1.0)
    assert abs(np.sum(complete['soup.x'][-1] > 1e-3) - 10) <= 1


def test_moran_fixation_probability():
    """Test that a fitter mutant fixes with the probability `(1 - 1 / r) / (1 - 1 / r^N)`."""
    fixed = []
    for seed in range(60):
        trajectory = _run(moran_process(10, selection=0.5), 60, random_state=seed)
        traits = trajectory.agents('population', 60).values['trait']
        assert len(np.unique(traits)) == 1
        fixed.append(traits[0] == 1)
    assert np.mean(fixed) == pytest.approx((1 - 1 / 1.5) / (1 - 1 / 1.5**10), abs=0.15)


def test_moran_mutation_diffuses_traits():
    """Test that mutation spreads the traits of a Moran population along the trait axis."""
    trajectory = _run(moran_process(30, mutants=0, mutation=0.5), 50, random_state=0)
    assert np.ptp(trajectory.agents('population', 50).values['trait']) > 0


def test_tangled_nature_population_persists_and_diversifies():
    """Test that the Tangled Nature population persists and mutants spread over several types."""
    trajectory = _run(tangled_nature(random_state=1), 20, random_state=0)
    final = trajectory.agents('organisms', 20).values['trait']
    assert trajectory['organisms.count'].min() > 0
    assert len(np.unique(final)) > 3


def test_adaptive_climbing_improves_and_small_populations_escape():
    """Test that large populations only climb, and a single always mutating individual escapes to lower fitness."""
    large = _run(adaptive_climbing(10, u=0.1), 100, random_state=0)
    fitness = [large.agents('population', frame).values['fitness'][0] for frame in range(101)]
    assert np.all(np.diff(fitness) >= 0)
    assert fitness[-1] > fitness[0]
    single = _run(adaptive_climbing(1, u=1.0), 100, random_state=0)
    wandering = [single.agents('population', frame).values['fitness'][0] for frame in range(101)]
    assert np.any(np.diff(wandering) < 0)


def test_nk_climbers_never_lose_fitness_and_find_single_peak():
    """Test that NK climbers never lose fitness, and all find the single peak of a landscape without epistasis."""
    rugged = _run(nk_landscape(20, k=4, random_state=0), 30, random_state=0)
    start, end = (rugged.agents('climbers', frame).values['fitness'] for frame in (0, 30))
    assert np.all(end >= start)
    smooth = _run(nk_landscape(20, k=0, random_state=0), 150, random_state=0)
    assert len(np.unique(smooth.agents('climbers', 150).values['fitness'])) == 1
