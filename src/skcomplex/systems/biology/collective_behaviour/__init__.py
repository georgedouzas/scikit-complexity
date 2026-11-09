"""Collective behaviour of animals and people.

Animals and people coordinate through signals, trails, and contact.
"""

from ._colony_optimization import AntColonyOptimizationSystem, ColonySearchInteraction, TrailSegmentElement
from ._decisions import (
    AlarmInteraction,
    BinaryAntDecisionSystem,
    BranchElement,
    RatsElement,
    RatsSystem,
    TrailChoiceInteraction,
)
from ._flocks import (
    BirdSwarmSystem,
    BoidElement,
    Boids3dSystem,
    BoidsSystem,
    FlightInteraction,
    FlockingInteraction,
    HeadingInteraction,
    MorsePotentialInteraction,
    SelfPropulsionInteraction,
    SpatialBoidElement,
    SpatialFlightInteraction,
)
from ._foraging import AntElement, PheromoneElement, PheromoneForagingSystem
from ._herding import HerdedSheepElement, ShepherdElement, ShepherdingInteraction, ShepherdingSystem
from ._leadership import CouzinLeadershipSystem, SchoolingInteraction, ShoalMemberElement, SwimmingInteraction
from ._many_body_swarms import (
    EncirclementInteraction,
    FlankingInteraction,
    ManyBodyAlignmentSwarmSystem,
    MorseAlignmentInteraction,
    SwarmAgentElement,
)
from ._trails import ActiveWalkersSystem, GroundElement, WalkerElement

__all__: list[str] = [
    'ActiveWalkersSystem',
    'AlarmInteraction',
    'AntColonyOptimizationSystem',
    'AntElement',
    'BinaryAntDecisionSystem',
    'BirdSwarmSystem',
    'BoidElement',
    'Boids3dSystem',
    'BoidsSystem',
    'BranchElement',
    'ColonySearchInteraction',
    'CouzinLeadershipSystem',
    'EncirclementInteraction',
    'FlankingInteraction',
    'FlightInteraction',
    'FlockingInteraction',
    'GroundElement',
    'HeadingInteraction',
    'HerdedSheepElement',
    'ManyBodyAlignmentSwarmSystem',
    'MorseAlignmentInteraction',
    'MorsePotentialInteraction',
    'PheromoneElement',
    'PheromoneForagingSystem',
    'RatsElement',
    'RatsSystem',
    'SchoolingInteraction',
    'SelfPropulsionInteraction',
    'ShepherdElement',
    'ShepherdingInteraction',
    'ShepherdingSystem',
    'ShoalMemberElement',
    'SpatialBoidElement',
    'SpatialFlightInteraction',
    'SwarmAgentElement',
    'SwimmingInteraction',
    'TrailChoiceInteraction',
    'TrailSegmentElement',
    'WalkerElement',
]
