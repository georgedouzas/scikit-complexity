"""Test the `_populations` module of `skcomplex.models.biology.ecology`."""

import pytest

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.common.populations import (
    LogisticGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
)
from skcomplex.models.biology.ecology import PredationInteraction


def test_logistic_growth_saturates():
    """Test that logistic growth approaches the carrying capacity."""
    system = System(
        elements=[PopulationElement('p', n=5)],
        interactions=[LogisticGrowthInteraction('g', target='p', r=1.0, K=200)],
    )
    assert Simulation(
        system,
        dt=1.0,
    ).check().prepare().run(
        until=30,
    ).trajectory_.values('p.n')[
        -1
    ] == pytest.approx(200, rel=1e-3)


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
    trajectory = Simulation(system, dt=0.1).check().prepare().run(until=50).trajectory_
    rabbits = trajectory.values('rabbits.n')
    assert rabbits.min() > 0
    assert rabbits.max() > 2 * rabbits.min()
