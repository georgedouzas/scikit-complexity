"""Collective behaviour: animals and people that coordinate through signals, trails, and contact."""

from ._decisions import (
    AlarmInteraction,
    BranchElement,
    RatsElement,
    TrailChoiceInteraction,
    binary_ant_decision,
    rats,
)
from ._flight import FlightInteraction
from ._flocking import FlockingInteraction
from ._flocks import BoidElement, MorsePotentialInteraction, SelfPropulsionInteraction, bird_swarm, boids
from ._foraging import AntElement, PheromoneElement, pheromone_foraging
from ._leadership import ShoalMemberElement, couzin_leadership
from ._schooling import SchoolingInteraction
from ._swimming import SwimmingInteraction
from ._trails import GroundElement, WalkerElement, active_walkers

__all__: list[str] = [
    'AlarmInteraction',
    'AntElement',
    'BoidElement',
    'BranchElement',
    'FlightInteraction',
    'FlockingInteraction',
    'GroundElement',
    'MorsePotentialInteraction',
    'PheromoneElement',
    'RatsElement',
    'SchoolingInteraction',
    'SelfPropulsionInteraction',
    'ShoalMemberElement',
    'SwimmingInteraction',
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
