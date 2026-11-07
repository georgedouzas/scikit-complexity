"""Evolutionary computation: genetic algorithms and bit strings on fitness landscapes."""

from ._annealed_candidate import AnnealedCandidateElement
from ._bit_string import BitStringElement
from ._genetic_algorithms import (
    AgedStrategyElement,
    CircumstanceInteraction,
    RoomCleaningInteraction,
    StrategyElement,
    robby_ga,
)
from ._landscapes import bit_string_evolution
from ._optimisers import evolutionary_optimisers
from ._reproduction import ReproductionInteraction
from ._running_best import RunningBestElement
from ._running_best_update import RunningBestInteraction
from ._search_candidate import SearchCandidateElement

__all__: list[str] = [
    'AgedStrategyElement',
    'AnnealedCandidateElement',
    'BitStringElement',
    'CircumstanceInteraction',
    'ReproductionInteraction',
    'RoomCleaningInteraction',
    'RunningBestElement',
    'RunningBestInteraction',
    'SearchCandidateElement',
    'StrategyElement',
    'bit_string_evolution',
    'evolutionary_optimisers',
    'robby_ga',
]
