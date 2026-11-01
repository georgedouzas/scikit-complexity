"""Collective behaviour: animals and people that coordinate through signals, trails, and contact."""

from ._alarm import AlarmInteraction
from ._ant import AntElement
from ._boid import BoidElement
from ._branch import BranchElement
from ._decisions import binary_ant_decision, rats
from ._flight import FlightInteraction
from ._flocking import FlockingInteraction
from ._flocks import bird_swarm, boids
from ._foraging import pheromone_foraging
from ._ground import GroundElement
from ._leadership import ShoalMemberElement, couzin_leadership
from ._morse_potential import MorsePotentialInteraction
from ._pheromone import PheromoneElement
from ._rats import RatsElement
from ._schooling import SchoolingInteraction
from ._self_propulsion import SelfPropulsionInteraction
from ._swimming import SwimmingInteraction
from ._trail_choice import TrailChoiceInteraction
from ._trails import active_walkers
from ._walker import WalkerElement

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
