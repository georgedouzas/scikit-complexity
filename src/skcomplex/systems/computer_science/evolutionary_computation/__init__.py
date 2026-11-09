"""Evolutionary computation: genetic algorithms and evolving bit strings.

The bit strings evolve on fitness landscapes.
"""

from ._genetic_algorithms import (
    AgedStrategyElement,
    CircumstanceInteraction,
    RobbyGASystem,
    RoomCleaningInteraction,
    StrategyElement,
)
from ._landscapes import BitStringElement, BitStringEvolutionSystem, ReproductionInteraction
from ._optimisers import (
    AnnealedCandidateElement,
    CovarianceAdaptationSpec,
    DifferentialEvolutionSpec,
    EvolutionaryOptimisersSystem,
    EvolutionStrategySpec,
    OptimizationInteraction,
    ParticleSwarmSpec,
    RunningBestElement,
    RunningBestInteraction,
    SearchCandidateElement,
)

__all__: list[str] = [
    'AgedStrategyElement',
    'AnnealedCandidateElement',
    'BitStringElement',
    'BitStringEvolutionSystem',
    'CircumstanceInteraction',
    'CovarianceAdaptationSpec',
    'DifferentialEvolutionSpec',
    'EvolutionStrategySpec',
    'EvolutionaryOptimisersSystem',
    'OptimizationInteraction',
    'ParticleSwarmSpec',
    'ReproductionInteraction',
    'RobbyGASystem',
    'RoomCleaningInteraction',
    'RunningBestElement',
    'RunningBestInteraction',
    'SearchCandidateElement',
    'StrategyElement',
]
