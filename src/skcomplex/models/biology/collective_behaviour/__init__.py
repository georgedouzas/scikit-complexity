"""Collective behaviour: animals and people that coordinate through signals, trails, and contact."""

from ._decisions import (
    BranchElement,
    NervousnessInteraction,
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
    'AntElement',
    'BoidElement',
    'BranchElement',
    'GroundElement',
    'MorsePotentialInteraction',
    'NervousnessInteraction',
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
