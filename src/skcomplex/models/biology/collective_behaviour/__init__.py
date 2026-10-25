"""Collective behaviour: animals and people that coordinate through signals, trails, and contact."""

from ._decisions import (
    BranchComponent,
    NervousnessInteraction,
    RatsComponent,
    TrailChoiceInteraction,
    binary_ant_decision,
    rats,
)
from ._flocks import BoidComponent, MorsePotentialInteraction, SelfPropulsionInteraction, bird_swarm, boids
from ._foraging import AntComponent, PheromoneComponent, pheromone_foraging
from ._leadership import ShoalMemberComponent, couzin_leadership
from ._trails import GroundComponent, WalkerComponent, active_walkers

__all__: list[str] = [
    'AntComponent',
    'BoidComponent',
    'BranchComponent',
    'GroundComponent',
    'MorsePotentialInteraction',
    'NervousnessInteraction',
    'PheromoneComponent',
    'RatsComponent',
    'SelfPropulsionInteraction',
    'ShoalMemberComponent',
    'TrailChoiceInteraction',
    'WalkerComponent',
    'active_walkers',
    'binary_ant_decision',
    'bird_swarm',
    'boids',
    'couzin_leadership',
    'pheromone_foraging',
    'rats',
]
