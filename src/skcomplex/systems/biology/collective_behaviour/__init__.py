"""Collective behaviour: animals and people that coordinate through signals, trails, and contact."""

from ._colony_optimization import ColonySearchInteraction, TrailSegmentElement, ant_colony_optimization
from ._decisions import AlarmInteraction, BranchElement, RatsElement, TrailChoiceInteraction, binary_ant_decision, rats
from ._flocks import (
    BoidElement,
    FlightInteraction,
    FlockingInteraction,
    HeadingInteraction,
    MorsePotentialInteraction,
    SelfPropulsionInteraction,
    SpatialBoidElement,
    SpatialFlightInteraction,
    bird_swarm,
    boids,
    boids_3d,
)
from ._foraging import AntElement, PheromoneElement, pheromone_foraging
from ._herding import HerdedSheepElement, ShepherdElement, ShepherdingInteraction, shepherding
from ._leadership import SchoolingInteraction, ShoalMemberElement, SwimmingInteraction, couzin_leadership
from ._trails import GroundElement, WalkerElement, active_walkers

__all__: list[str] = [
    'AlarmInteraction',
    'AntElement',
    'BoidElement',
    'BranchElement',
    'ColonySearchInteraction',
    'FlightInteraction',
    'FlockingInteraction',
    'GroundElement',
    'HeadingInteraction',
    'HerdedSheepElement',
    'MorsePotentialInteraction',
    'PheromoneElement',
    'RatsElement',
    'SchoolingInteraction',
    'SelfPropulsionInteraction',
    'ShepherdElement',
    'ShepherdingInteraction',
    'ShoalMemberElement',
    'SpatialBoidElement',
    'SpatialFlightInteraction',
    'SwimmingInteraction',
    'TrailChoiceInteraction',
    'TrailSegmentElement',
    'WalkerElement',
    'active_walkers',
    'ant_colony_optimization',
    'binary_ant_decision',
    'bird_swarm',
    'boids',
    'boids_3d',
    'couzin_leadership',
    'pheromone_foraging',
    'rats',
    'shepherding',
]
