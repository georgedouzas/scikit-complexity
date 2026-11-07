"""Test the `_predation` module of `skcomplex.systems.biology.common.populations`."""

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.biology.common.populations import (
    LogisticGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
)
from skcomplex.systems.biology.ecology import PredationInteraction


def test_lotka_volterra_oscillates():
    """Test that predators and prey cycle without dying out."""
    system = System(
        elements=[PopulationElement('rabbits', n=40), PopulationElement('foxes', n=9)],
        interactions=[
            LogisticGrowthInteraction('growth', target='rabbits', r=1.1, K=1e6),
            PredationInteraction('hunt', predator='foxes', prey='rabbits', a=0.4, b=0.1),
            MortalityInteraction('death', target='foxes', d=0.4),
        ],
    )
    trajectory = Simulation(system, dt=0.1).prepare().run(until=50).trajectory_
    rabbits = trajectory.values('rabbits.n')
    assert rabbits.min() > 0
    assert rabbits.max() > 2 * rabbits.min()
