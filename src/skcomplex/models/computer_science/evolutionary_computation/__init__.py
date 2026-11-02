"""Evolutionary computation: genetic algorithms and bit strings on fitness landscapes."""

from ._annealed_candidate import AnnealedCandidateElement
from ._bit_string import BitStringElement
from ._genetic_algorithms import robby_ga
from ._landscapes import bit_string_evolution
from ._optimisers import evolutionary_optimisers
from ._reproduction import ReproductionInteraction
from ._room_cleaning import RoomCleaningInteraction
from ._running_best import RunningBestElement
from ._running_best_update import RunningBestInteraction
from ._search_candidate import SearchCandidateElement
from ._strategy import StrategyElement

__all__: list[str] = [
    'AnnealedCandidateElement',
    'BitStringElement',
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
