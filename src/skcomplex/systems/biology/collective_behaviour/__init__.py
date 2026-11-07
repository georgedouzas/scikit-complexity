"""Collective behaviour: animals and people that coordinate through signals, trails, and contact."""

from ._alarm import AlarmInteraction
from ._ant import AntElement
from ._boid import BoidElement
from ._branch import BranchElement
from ._colony_optimization import ant_colony_optimization
from ._colony_search import ColonySearchInteraction
from ._decisions import binary_ant_decision, rats
from ._flight import FlightInteraction
from ._flocking import FlockingInteraction
from ._flocks import bird_swarm, boids, boids_3d
from ._foraging import pheromone_foraging
from ._ground import GroundElement
from ._heading import HeadingInteraction
from ._herded_sheep import HerdedSheepElement
from ._herding import shepherding
from ._leadership import ShoalMemberElement, couzin_leadership
from ._morse_potential import MorsePotentialInteraction
from ._pheromone import PheromoneElement
from ._rats import RatsElement
from ._schooling import SchoolingInteraction
from ._self_propulsion import SelfPropulsionInteraction
from ._shepherd import ShepherdElement
from ._shepherding import ShepherdingInteraction
from ._spatial_boid import SpatialBoidElement
from ._spatial_flight import SpatialFlightInteraction
from ._swimming import SwimmingInteraction
from ._trail_choice import TrailChoiceInteraction
from ._trail_segment import TrailSegmentElement
from ._trails import active_walkers
from ._walker import WalkerElement

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
