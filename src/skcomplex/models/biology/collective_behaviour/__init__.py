"""Collective behaviour: animals and people that coordinate through signals, trails, and contact."""

from ._decisions import (
    AlarmInteraction,
    BranchElement,
    RatsElement,
    TrailChoiceInteraction,
    binary_ant_decision,
    rats,
)
from ._flocks import BoidElement, MorsePotentialInteraction, SelfPropulsionInteraction, bird_swarm, boids
from ._foraging import AntElement, PheromoneElement, pheromone_foraging
from ._leadership import ShoalMemberElement, couzin_leadership
from ._trails import GroundElement, WalkerElement, active_walkers

__all__: list[str] = [
    'AlarmInteraction',
    'AntElement',
    'BoidElement',
    'BranchElement',
    'GroundElement',
    'MorsePotentialInteraction',
    'PheromoneElement',
    'RatsElement',
    'SelfPropulsionInteraction',
    'ShoalMemberElement',
    'TrailChoiceInteraction',
    'WalkerElement',
    'active_walkers',
    'binary_ant_decision',
    'bird_swarm',
    'boids',
    'couzin_leadership',
    'pheromone_foraging',
    'rats',
]
