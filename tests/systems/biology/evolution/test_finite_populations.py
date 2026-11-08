"""Test the `_finite_populations` module of `skcomplex.systems.biology.evolution`."""

import numpy as np
import pytest

from skcomplex.systems import all_systems
from skcomplex.systems.biology.evolution import (
    AdaptiveClimbingSystem,
    MoranProcessSystem,
    NKLandscapeSystem,
    TangledNatureSystem,
)

from ..conftest import _run_until

ECOLOGY = {
    'CompetitionSystem',
    'MutualismSystem',
    'FoodWebsSystem',
    'RosenzweigMacarthurSystem',
    'NicholsonBaileySystem',
    'LotkaVolterraWithHarvestingSystem',
    'VaryingCapacityLogisticSystem',
    'MangroveSuccessionSystem',
    'AgentBasedLotkaVolterraSystem',
    'StigmergicPredatorPreySystem',
    'MeadowCapacitySystem',
    'NeutralMacroecologySystem',
    'StochasticPredatorPreySystem',
}


EPIDEMIOLOGY = {
    'SISSystem',
    'SIRSSystem',
    'AgentBasedSIRSystem',
    'RandomWalkerEpidemicsSystem',
    'NetworkSIRSystem',
    'HeterogeneousMeanFieldSISSystem',
    'MetapopulationSystem',
    'DelayedSIRSSystem',
}


EVOLUTION = {
    'ReplicatorDynamicsSystem',
    'MoranProcessSystem',
    'EigenQuasispeciesSystem',
    'HypercyclesSystem',
    'NKLandscapeSystem',
    'TangledNatureSystem',
    'AdaptiveClimbingSystem',
    'RockPaperScissorsSystem',
}


@pytest.mark.parametrize(
    ('field', 'names'),
    [('biology/ecology', ECOLOGY), ('biology/epidemiology', EPIDEMIOLOGY), ('biology/evolution', EVOLUTION)],
)
def test_all_systems_lists_subfield(field, names):
    """Test that `all_systems` lists the systems of a subfield with descriptions and references."""
    systems = dict(all_systems(field=field))
    assert names <= set(systems)
    assert all(systems[name].__doc__ and systems[name].references for name in names)


def test_moran_fixation_probability():
    """Test that a fitter mutant fixes with the probability `(1 - 1 / r) / (1 - 1 / r^N)`."""
    fixed = []
    for seed in range(60):
        trajectory = _run_until(MoranProcessSystem(10, selection=0.5), 60, random_state=seed)
        traits = trajectory.agents('population', 60).values['trait']
        assert len(np.unique(traits)) == 1
        fixed.append(traits[0] == 1)
    assert np.mean(fixed) == pytest.approx((1 - 1 / 1.5) / (1 - 1 / 1.5**10), abs=0.15)


def test_moran_mutation_diffuses_traits():
    """Test that mutation spreads the traits of a Moran population along the trait axis."""
    trajectory = _run_until(MoranProcessSystem(30, mutants=0, mutation=0.5), 50, random_state=0)
    assert np.ptp(trajectory.agents('population', 50).values['trait']) > 0


def test_tangled_nature_population_persists_and_diversifies():
    """Test that the Tangled Nature population persists and mutants spread over several types."""
    trajectory = _run_until(TangledNatureSystem(random_state=1), 20, random_state=0)
    final = trajectory.agents('organisms', 20).values['trait']
    assert trajectory.values('organisms.count').min() > 0
    assert len(np.unique(final)) > 3


def test_adaptive_climbing_improves_and_small_populations_escape():
    """Test that large populations only climb, and a single always mutating individual escapes to lower fitness."""
    large = _run_until(AdaptiveClimbingSystem(10, u=0.1), 100, random_state=0)
    fitness = [large.agents('population', frame).values['fitness'][0] for frame in range(101)]
    assert np.all(np.diff(fitness) >= 0)
    assert fitness[-1] > fitness[0]
    single = _run_until(AdaptiveClimbingSystem(1, u=1.0), 100, random_state=0)
    wandering = [single.agents('population', frame).values['fitness'][0] for frame in range(101)]
    assert np.any(np.diff(wandering) < 0)


def test_nk_climbers_never_lose_fitness_and_find_single_peak():
    """Test that NK climbers never lose fitness, and all find the single peak of a landscape without epistasis."""
    rugged = _run_until(NKLandscapeSystem(20, k=4, random_state=0), 30, random_state=0)
    start, end = (rugged.agents('climbers', frame).values['fitness'] for frame in (0, 30))
    assert np.all(end >= start)
    smooth = _run_until(NKLandscapeSystem(20, k=0, random_state=0), 150, random_state=0)
    assert len(np.unique(smooth.agents('climbers', 150).values['fitness'])) == 1
